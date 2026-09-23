from __future__ import annotations
import argparse
import json
from typing import Callable
import torch
from torch import nn
from torch.nn import functional as F
from einops import repeat, rearrange
import logging
import random
from collections import defaultdict, deque
import math
from typing import Dict, Optional, Union
from src.baseline_attention_reg import AttentionEntropyMaximization
from vanilla_supcon import VanillaSupConLoss as Vsupcon


class NeedleProportionBCE(nn.Module):
    """
    Proportion BCE loss on per-patch cancer logits within the needle region.
    Forces encoder to produce spatially heterogeneous features by supervising
    what fraction of needle patches should be cancer-positive.

    Uses the same MaskedPredictionModule logic as ProstNFound but operates
    on per-patch cancer scores from a lightweight head on top of H.

    Parameters
    ----------
    patch_cancer_head : nn.Linear(proj_dim, 1) — predicts per-patch cancer score
    treat_gg1_as_benign : if True, GG1 cores are treated as benign (involvement=0)
    pos_weight : upweight cancer patches
    """
    def __init__(
        self,
        proj_dim:           int   = 768,
        treat_gg1_as_benign: bool = False,
        pos_weight:         float = 8.0,
    ):
        super().__init__()
        self.treat_gg1_as_benign = treat_gg1_as_benign
        self.pos_weight          = pos_weight

        # Lightweight per-patch cancer scoring head
        # Takes projected patch features H: (B, N, proj_dim) → (B, N, 1)
        self.patch_cancer_head = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, data: dict) -> torch.Tensor:
        """
        Expects:
            data["patch_feats_needle"] : (B, N, proj_dim) projected patch features H
            data["involvement"]        : (B,) float, cancer involvement fraction [0,1]
            data["grade_group"]        : (B,) int ISUP grade
        """
        device      = data["patch_feats_needle"].device
        H           = data["patch_feats_needle"]          # (B, N, D) ON graph
        involvement = data["involvement"].float().to(device)  # (B,)
        grades      = data["grade_group"].to(device)

        if self.treat_gg1_as_benign:
            involvement = involvement.clone()
            involvement[grades == 1] = 0.0

        B, N, D = H.shape

        # Per-patch cancer logits: (B, N, 1) → (B, N)
        patch_logits = self.patch_cancer_head(H).squeeze(-1)  # (B, N)

        grade_mean_scores = {}
        for grade in range(6):
            mask = (grades == grade)
            if mask.sum() > 0:
                grade_mean_scores[grade] = patch_logits[mask].sigmoid().mean(dim=1).mean()
        data["grade_mean_cancer_scores"] = grade_mean_scores
        # grade_weights = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.1, 5: 0.1}
        grade_weights = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}
        loss = torch.tensor(0.0, device=device)
        for i in range(B):
            grade_i      = int(grades[i].item())
            weight_i     = grade_weights.get(grade_i, 1.0)
            bag_logits   = patch_logits[i]
            involvement_i = involvement[i].item()
            probs     = bag_logits.sigmoid()
            pred_prop = probs.mean()
            sample_loss = (
                -involvement_i * (pred_prop + 1e-8).log()
                - (1 - involvement_i) * (1 - pred_prop + 1e-8).log()
            )
            if involvement_i > 0:
                sample_loss = sample_loss * self.pos_weight
            loss = loss + weight_i * sample_loss
        return loss / B


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, p=2, mode='hist'):
        """
        mode: 'hist' → reads from positive_hist/negative_hist (all modes except joint)
              'mri'  → reads from joint_positive_mri/joint_negative_mri (us_mri_histo only)
        """
        super().__init__()
        self.criterion = nn.TripletMarginLoss(margin=margin, p=p)
        self.mode = mode
        self.num_skipped = 0
        self.num_computed = 0
        self.num_true_adjacent = 0
        self.num_total_mined = 0
        self.anchor_neg_gap = []
        self.per_grade_adjacent = {grade: [0, 0] for grade in [0, 2, 3, 4, 5]}
        self.GRADE_SEQUENCE = [0, 2, 3, 4, 5]

    def _get_keys(self):
        """Return the correct data keys based on mode."""
        if self.mode == 'mri':
            return (
                "joint_positive_mri",
                "joint_negative_mri",
                "joint_neg_grade_mri",
            )
        else:  # hist — default for all non-joint modes
            return (
                "positive_hist",
                "negative_hist",
                "negative_grade",
            )

    def _is_adjacent(self, anchor_label, neg_label):
        if anchor_label not in self.GRADE_SEQUENCE or neg_label not in self.GRADE_SEQUENCE:
            return False
        idx = self.GRADE_SEQUENCE.index(anchor_label)
        neighbors = []
        if idx > 0:
            neighbors.append(self.GRADE_SEQUENCE[idx - 1])
        if idx < len(self.GRADE_SEQUENCE) - 1:
            neighbors.append(self.GRADE_SEQUENCE[idx + 1])
        return neg_label in neighbors

    def _track_mining(self, anchor_grades, neg_grades):
        for anchor_label, neg_label in zip(anchor_grades, neg_grades):
            anchor_label = int(anchor_label)
            neg_label = int(neg_label)

            is_adjacent = self._is_adjacent(anchor_label, neg_label)
            self.num_true_adjacent += int(is_adjacent)
            self.num_total_mined += 1
            self.anchor_neg_gap.append(abs(neg_label - anchor_label))

            if anchor_label in self.per_grade_adjacent:
                self.per_grade_adjacent[anchor_label][1] += 1
                if is_adjacent:
                    self.per_grade_adjacent[anchor_label][0] += 1

        if self.num_total_mined % 2000 == 0:
            rate = self.num_true_adjacent / self.num_total_mined
            avg_gap = sum(self.anchor_neg_gap) / len(self.anchor_neg_gap)
            per_grade_rates = {
                g: f"{v[0]/v[1]:.2%} ({v[0]}/{v[1]})"
                for g, v in sorted(self.per_grade_adjacent.items())
                if v[1] > 0
            }
            logging.info(
                f"[TripletLoss:{self.mode}] Adjacent rate: {rate:.2%} | "
                f"Avg grade gap: {avg_gap:.2f} | "
                f"Gap dist: { {g: self.anchor_neg_gap.count(g) for g in sorted(set(self.anchor_neg_gap))} } | "
                f"Per-grade rates: {per_grade_rates}"
            )

    def forward(self, data):
        pos_key, neg_key, grade_key = self._get_keys()

        required_keys = ["image_feats_needle", pos_key, neg_key]
        missing = [k for k in required_keys if k not in data or data[k] is None]

        if missing:
            self.num_skipped += 1
            logging.warning(f"[TripletLoss:{self.mode}] Skipped (count={self.num_skipped}). Missing: {missing}")
            return torch.tensor(0.0, device=next(iter(data.values())).device, requires_grad=True)

        self.num_computed += 1
        embed_us = data["image_feats_needle"].to('cuda')
        X_pos    = data[pos_key].to('cuda')
        X_neg    = data[neg_key].to('cuda')

        # Track mining quality
        if (
            "bucket_label" in data and
            grade_key in data and
            data[grade_key] is not None
        ):
            self._track_mining(
                data["bucket_label"].cpu().tolist(),
                data[grade_key].cpu().tolist(),
            )

        loss = self.criterion(embed_us, X_pos, X_neg)
        return loss


class TripletLossUS(nn.Module):
    def __init__(self, margin=1.0, p=2, hard_mining_prob=0.8, bank_size=64, num_grades=6):
        super().__init__()
        self.criterion = nn.TripletMarginLoss(margin=margin, p=p)
        self.hard_mining_prob = hard_mining_prob
        self.bank_size = bank_size
        self.num_skipped = 0
        self.num_computed = 0
        self.num_true_adjacent = 0
        self.num_total_mined = 0
        self.anchor_neg_gap = []
        self.bank = {grade: deque(maxlen=bank_size) for grade in [0, 2, 3, 4, 5]}
        self.per_grade_adjacent = {grade: [0, 0] for grade in [0, 2, 3, 4, 5]}

    def _update_bank(self, embeds, labels_list):
        for emb, lbl in zip(embeds, labels_list):
            self.bank[lbl].append(emb.detach().clone())

    def _is_adjacent(self, anchor_label, neg_label):
        """True if neg_label is the nearest neighbor of anchor_label in the actual grade sequence."""
        GRADE_SEQUENCE = [0, 2, 3, 4, 5]
        if anchor_label not in GRADE_SEQUENCE or neg_label not in GRADE_SEQUENCE:
            return False
        idx = GRADE_SEQUENCE.index(anchor_label)
        neighbors = []
        if idx > 0:
            neighbors.append(GRADE_SEQUENCE[idx - 1])
        if idx < len(GRADE_SEQUENCE) - 1:
            neighbors.append(GRADE_SEQUENCE[idx + 1])
        return neg_label in neighbors

    def _get_bank_sample(self, grade):
        pool = list(self.bank[grade])
        return random.choice(pool) if pool else None

    def _get_hard_label(self, label, valid_classes):
        candidates = [c for c in valid_classes if c != label]
        if not candidates:
            return None

        if random.random() < self.hard_mining_prob:
            adjacent = []
            if label > min(valid_classes):
                lower = [c for c in valid_classes if c < label]
                if lower:
                    adjacent.append(max(lower))
            if label < max(valid_classes):
                upper = [c for c in valid_classes if c > label]
                if upper:
                    adjacent.append(min(upper))
            return random.choice(adjacent) if adjacent else random.choice(candidates)
        else:
            return random.choice(candidates)

    def forward(self, data):
        required_keys = ["image_feats_needle", "grade_group"]
        missing = [k for k in required_keys if k not in data or data[k] is None]

        if missing:
            self.num_skipped += 1
            logging.warning(f"Triplet loss skipped (count={self.num_skipped}). Missing keys: {missing}")
            return torch.tensor(0.0, device=next(iter(data.values())).device, requires_grad=True)

        embeds = data["image_feats_needle"]   # (B, D)
        labels = data["grade_group"]          # (B,)
        device = embeds.device

        labels_list = labels.cpu().tolist()

        # Update bank BEFORE mining so current batch is available as candidates
        self._update_bank(embeds, labels_list)

        # Valid classes: only grades with non-empty banks
        valid_classes = [g for g in sorted(self.bank.keys()) if len(self.bank[g]) > 0]
        anchors, positives, negatives = [], [], []

        for i in range(len(embeds)):
            anchor_label = labels_list[i]

            # Positive: same class from bank, exclude current embedding
            pos_pool = [
                e for e in self.bank[anchor_label]
                if not torch.equal(e, embeds[i].detach())
            ]
            if len(pos_pool) == 0:
                continue

            # Hard negative: adjacent grade from bank
            hard_neg_label = self._get_hard_label(anchor_label, valid_classes)
            if hard_neg_label is None:
                continue

            pos_emb = random.choice(pos_pool)
            neg_emb = self._get_bank_sample(hard_neg_label)
            if neg_emb is None:
                continue

            # Mining quality tracking
            is_adjacent = self._is_adjacent(anchor_label, hard_neg_label)
            self.num_true_adjacent += int(is_adjacent)
            self.num_total_mined += 1
            self.anchor_neg_gap.append(abs(hard_neg_label - anchor_label))

            self.per_grade_adjacent[anchor_label][1] += 1
            if is_adjacent:
                self.per_grade_adjacent[anchor_label][0] += 1

            if self.num_total_mined % 2000 == 0:
                rate = self.num_true_adjacent / self.num_total_mined
                avg_gap = sum(self.anchor_neg_gap) / len(self.anchor_neg_gap)
                per_grade_rates = {
                    g: f"{v[0]/v[1]:.2%} ({v[0]}/{v[1]})"
                    for g, v in sorted(self.per_grade_adjacent.items())
                    if v[1] > 0
                }
                logging.info(
                    f"[TripletLossUS] Adjacent rate: {rate:.2%} | "
                    f"Avg grade gap: {avg_gap:.2f} | "
                    f"Gap dist: { {g: self.anchor_neg_gap.count(g) for g in sorted(set(self.anchor_neg_gap))} } | "
                    f"Per-grade rates: {per_grade_rates}"
                )

            anchors.append(embeds[i])
            positives.append(pos_emb)
            negatives.append(neg_emb)

        if len(anchors) == 0:
            self.num_skipped += 1
            logging.warning(f"Triplet loss skipped (count={self.num_skipped}). No valid triplets in batch.")
            return embeds.sum() * 0.0

        self.num_computed += 1
        anchors   = torch.stack(anchors)
        positives = torch.stack(positives).to(device)
        negatives = torch.stack(negatives).to(device)

        return self.criterion(anchors, positives, negatives)


class SumLoss(nn.Module):
    def __init__(self, losses, weights=None, names=None, log_every=10, normalize=False):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights if weights is not None else [1.0] * len(losses)
        self.names = names if names is not None else [f'loss_{i}' for i in range(len(losses))]
        self.log_every = log_every
        self.normalize = normalize
        self._step = 0
        self._loss_scale = None  # set on first forward pass

    def set_epoch(self, epoch: int):
        """Propagate epoch to any child loss that needs it."""
        for loss_fn in self.losses:
            if hasattr(loss_fn, 'set_epoch'):
                loss_fn.set_epoch(epoch)

    def forward(self, data):
        total_loss = None
        individual_losses = {}

        raw_losses = {}
        for loss_fn, name in zip(self.losses, self.names):
            raw_losses[name] = loss_fn(data)

        # on first step, record initial magnitudes as scale factors
        if self.normalize and self._loss_scale is None:
            self._loss_scale = {
                name: max(raw_losses[name].item(), 1e-8) 
                for name in self.names
            }
            logging.info(f"Loss scales initialized: { {k: f'{v:.4f}' for k, v in self._loss_scale.items()} }")

        for name, raw, weight in zip(self.names, raw_losses.values(), self.weights):
            scale = self._loss_scale[name] if (self.normalize and self._loss_scale) else 1.0
            component = (raw / scale) * weight
            individual_losses[name] = raw.item()  # log raw for interpretability
            total_loss = component if total_loss is None else total_loss + component

        individual_losses['total'] = total_loss.item()
        data['_loss_components'] = individual_losses

        if self._step % self.log_every == 0:
            loss_str = " | ".join(f"{name}={val:.4f}" for name, val in individual_losses.items())
            logging.info(f"[step {self._step}] {loss_str}")

        self._step += 1
        return total_loss


class ISUPLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 6,
        label_smoothing: float = 0.1,
        weight_strategy: str = 'custom',  # 'sqrt', 'inverse', 'custom'
    ):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.weight_strategy = weight_strategy

        # Weights indexed by grade: [grade0, grade1, grade2, grade3, grade4, grade5]
        # Grade 1 is 0.0 since it doesn't exist in the dataset
        if weight_strategy == 'inverse':
            weights = torch.tensor([0.0159, 0.0, 0.1634, 0.3791, 0.4793, 1.0])
        elif weight_strategy == 'sqrt':
            weights = torch.tensor([0.1261, 0.0, 0.4042, 0.6157, 0.6923, 1.0])
        elif weight_strategy == 'custom':
            # Clinically motivated: benign gets moderate weight,
            # cancer grades get full weight to prioritize sensitivity
            weights = torch.tensor([0.6, 0.0, 1.0, 1.0, 1.0, 1.0])
        else:
            raise ValueError(f"Unknown weight_strategy: {weight_strategy}")

        self.register_buffer('weight', weights)

    def forward(self, data):
        if "isup_logits" not in data:
            logging.warning("ISUP logits not in data")
            return torch.tensor(0.0, device=data["grade_group"].device, requires_grad=True)

        logits = data["isup_logits"]
        labels = data["grade_group"].long().to(logits.device)

        assert (labels != 1).all(), "Unexpected grade 1 sample found in batch"

        loss = F.cross_entropy(
            logits,
            labels,
            weight=self.weight.to(logits.device),
            label_smoothing=self.label_smoothing,
        )
        return loss

class LossScaleNormalizer:
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self._scale = None
    def normalize(self, loss: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            current = loss.item()
            if self._scale is None:
                self._scale = current
            else:
                self._scale = (
                    self.momentum * self._scale 
                    + (1 - self.momentum) * current
                )
        return loss / (self._scale + 1e-8)


class ConditionalMMDLoss(nn.Module):
    def __init__(self, kernel_bandwidths=None, grade_sequence=None):
        super().__init__()
        self.bandwidths = kernel_bandwidths or [0.5, 1.0, 2.0, 5.0]
        self.grade_sequence = grade_sequence or [0, 1, 2, 3, 4, 5]

    def _rbf_kernel(self, X, Y):
        """Gaussian RBF kernel matrix between X and Y."""
        XX = (X ** 2).sum(dim=1, keepdim=True)
        YY = (Y ** 2).sum(dim=1, keepdim=True)
        dist = XX + YY.t() - 2 * torch.mm(X, Y.t())
        K = sum(torch.exp(-dist/(2 * bw ** 2)) for bw in self.bandwidths)
        return K / len(self.bandwidths)

    def _mmd(self, X, Y):
        """Unbiased MMD^2 between X and Y."""
        n, m = X.size(0), Y.size(0)
        Kxx = self._rbf_kernel(X, X)
        Kyy = self._rbf_kernel(Y, Y)
        Kxy = self._rbf_kernel(X, Y)
        # Unbiased: exclude diagonal for within-set terms
        Kxx = Kxx * (1 - torch.eye(n, device=X.device)) / max(n * (n - 1), 1)
        Kyy = Kyy * (1 - torch.eye(m, device=Y.device)) / max(m * (m - 1), 1)
        return Kxx.sum() + Kyy.sum() - 2 * Kxy.mean()

    def forward(self, us_embeds, histo_embeds, labels_list, device, us_bank, histo_bank):
        labels = torch.tensor(labels_list, device=device)
        mmd_terms = []
        for grade in self.grade_sequence:
            # Current batch samples for this grade
            mask = labels == grade
            us_g   = us_embeds[mask]
            hist_g = histo_embeds[mask]
            # Augment with bank entries
            us_bank_g   = list(us_bank[grade])
            hist_bank_g = list(histo_bank[grade])
            if us_bank_g:
                us_g = torch.cat([us_g, torch.stack(us_bank_g).to(device)], dim=0)
            if hist_bank_g:
                hist_g = torch.cat([hist_g, torch.stack(hist_bank_g).to(device)], dim=0)
            if us_g.size(0) < 2 or hist_g.size(0) < 2:
                continue
            mmd_terms.append(self._mmd(us_g, hist_g))
        if not mmd_terms:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return torch.stack(mmd_terms).mean()


class OrdinalNegWeightedSupConLossv2(nn.Module):
    """
    Supervised contrastive loss with ordinal negative weighting.

    Assumptions:
      - embeddings are L2-normalized before being passed in (caller's responsibility)
      - labels are contiguous integer indices (0, 1, 2, ...) not raw ISUP grades
      - when domain_aware=True, domain_labels must be provided (0=US, 1=histo)

    Denominator construction (corrected):
      For anchor i:
        denom_i = Σ_{j≠i, same grade}   exp(sim_ij)          # positives, weight 1
                + Σ_{j≠i, diff grade}   exp(sim_ij) * w_ij   # negatives, ordinal weight
      where w_ij = 1 + neg_strength * |label_i - label_j| / max_dist
                     [+ domain_boost if cross-domain negative]

    Numerator:
      num_i = mean_{p in P(i)} sim_ip     (unweighted, standard SupCon)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        neg_strength: float = 1.0,
        max_label_distance: int = 5,
        domain_aware: bool = False,
        domain_boost: float = 0.5,   # additive boost on cross-domain negatives
    ):
        super().__init__()
        self.temperature = max(float(temperature), 1e-6)
        self.neg_strength = float(neg_strength)
        self.max_label_distance = max(1, int(max_label_distance))
        self.domain_aware = bool(domain_aware)
        self.domain_boost = float(domain_boost)

    def forward(
        self,
        embeddings: torch.Tensor,          # (N, D), L2-normalized
        labels: torch.Tensor,              # (N,)   integer grade indices
        domain_labels: Optional[torch.Tensor] = None,  # (N,) 0=US, 1=histo
    ) -> torch.Tensor:
        device = embeddings.device
        n = embeddings.size(0)
        labels = labels.view(-1).to(device)
        # ------------------------------------------------------------------ #
        # Similarity matrix                                                    #
        # ------------------------------------------------------------------ #
        sim = torch.matmul(embeddings, embeddings.t()) / self.temperature   # (N, N)
        # Exclude self-pairs everywhere
        self_mask = torch.eye(n, dtype=torch.bool, device=device)
        off_diag  = ~self_mask                                              # (N, N)
        # ------------------------------------------------------------------ #
        # Positive / negative masks                                            #
        # ------------------------------------------------------------------ #
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)           # (N, N)
        pos_mask = labels_equal & off_diag                                  # same grade, not self
        neg_mask = (~labels_equal) & off_diag                               # diff grade, not self
        # Guard: if no positive pairs exist for any anchor, loss is undefined
        if pos_mask.sum() == 0:
            return embeddings.sum() * 0.0
        # ------------------------------------------------------------------ #
        # Ordinal negative weights                                             #
        # ------------------------------------------------------------------ #
        yi   = labels.unsqueeze(1).float()
        yj   = labels.unsqueeze(0).float()
        dist = torch.abs(yi - yj)
        d_norm = (dist / self.max_label_distance).clamp(0.0, 1.0)          # (N, N)
        # Base negative weight: 1 for adjacent grades, up to (1+neg_strength) for max distance
        w_neg = (1.0 + self.neg_strength * d_norm) * neg_mask.float()      # (N, N), 0 on pos/self
        # Optional domain boost: cross-domain negatives get extra push
        if self.domain_aware:
            if domain_labels is None:
                raise ValueError("domain_labels required when domain_aware=True")
            domain_labels = domain_labels.view(-1).to(device)
            cross_domain = (
                (domain_labels.unsqueeze(0) != domain_labels.unsqueeze(1)).float()
            )                                                               # (N, N)
            # Additive boost only on negative pairs — does not affect positives
            w_neg = w_neg + self.domain_boost * cross_domain * neg_mask.float()
        # ------------------------------------------------------------------ #
        # Denominator: positives (weight=1) + weighted negatives              #
        # ------------------------------------------------------------------ #
        # pos_mask entries get weight 1.0, neg_mask entries get w_neg,
        # diagonal entries get 0.0 (excluded).
        denom_weights = pos_mask.float() + w_neg                            # (N, N)
        sim_stable = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp_sim    = torch.exp(sim_stable)                                  # (N, N)
        denom = (exp_sim * denom_weights).sum(dim=1, keepdim=True) + 1e-8  # (N, 1)
        log_prob = sim_stable - torch.log(denom)                            # (N, N)
        pos_count = pos_mask.sum(dim=1).clamp(min=1).float()               # (N,)
        loss_per_anchor = -(log_prob * pos_mask.float()).sum(dim=1) / pos_count

        return loss_per_anchor.mean()


# ---------------------------------------------------------------------------
# Outer loss: within-US + cross-modal, with memory banks
# ---------------------------------------------------------------------------

class WithinModalSupConLossv2(nn.Module):
    """
    Two-term contrastive loss for US/histopathology cross-modal alignment.

    Term 1 — within_loss (primary):
        US-only SupCon augmented from a US memory bank.
        Directly optimizes grade separation in US embedding space.

    Term 2 — cross_loss (secondary):
        Cross-modal SupCon over a mixed US+histo batch augmented from both banks.
        Histo embeddings (privileged, train-time only) supervise US grade structure.
        domain_aware=True boosts cross-domain negative weights to counteract
        the modality gap pulling same-grade US/histo embeddings apart.

    Memory bank:
        Embeddings are stored post-normalization and detached.
        Optional momentum update (momentum > 0) maintains a running average
        of embeddings per grade, reducing staleness across training steps.

    Args:
        temperature:    SupCon temperature (default 0.07)
        neg_strength:   Scale of ordinal negative boost (default 1.0)
        domain_boost:   Additive weight on cross-domain negatives in Term 2 (default 0.5)
        lambda_cross:   Weight of cross-modal term relative to within-US (default 0.5)
        lambda_cross:   Weight of cross-modal term relative to corss-modal (default 0.5)
        bank_size:      Max embeddings stored per grade per modality (default 128)
        grade_sequence: List of raw ISUP grade values present in data
        within_bank_n:  Bank samples per grade for within-US batch (default 16)
        cross_bank_n:   Bank samples per grade per modality for cross-modal batch (default 8)
        momentum:       EMA momentum for bank updates; 0 = no momentum (default 0.0)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        neg_strength: float = 1.0,
        domain_boost: float = 0.5,
        lambda_cross: float = 1, #0.5,
        lambda_within: float = 1,
        bank_size: int = 128,
        grade_sequence: list = [0, 1, 2, 3, 4, 5], #TODO
        # grade_sequence: list = [0, 1, 2],
        within_bank_n: int = 16,
        cross_bank_n: int = 16,
        momentum: float = 0.0,
        lambda_mri = 0.5,
        lambda_mmd = 0.0,
        predictor: Optional[nn.Module] = None,  # JEPAPredictor instance, owned externally
        use_predictor_for_cross: bool = False, 
        use_mmd = False,
    ):
        super().__init__()
        self.lambda_cross   = lambda_cross
        self.lambda_within  = lambda_within
        self.grade_sequence = grade_sequence
        self.grade_to_idx   = {g: i for i, g in enumerate(grade_sequence)}
        self.within_bank_n  = within_bank_n
        self.cross_bank_n   = cross_bank_n
        self.momentum       = momentum
        self.bank_size      = bank_size
        self.lambda_mri  = lambda_mri   # new arg, e.g. default 0.2
        self.mri_bank    = {g: deque(maxlen=bank_size) for g in grade_sequence}
        self._mri_ema:   dict[int, Optional[torch.Tensor]] = {g: None for g in grade_sequence}
        self.lambda_mmd  = lambda_mmd
        self.use_mmd = use_mmd
        self.predictor = predictor
        self.use_predictor_for_cross = use_predictor_for_cross


        self._num_skipped  = 0
        self._num_computed = 0

        # Per-grade deque banks (store normalized embeddings, detached)
        self.us_bank    = {g: deque(maxlen=bank_size) for g in grade_sequence}
        self.histo_bank = {g: deque(maxlen=bank_size) for g in grade_sequence}

        # Optional momentum buffers: grade -> running mean embedding
        self._us_ema:    dict[int, Optional[torch.Tensor]] = {g: None for g in grade_sequence}
        self._histo_ema: dict[int, Optional[torch.Tensor]] = {g: None for g in grade_sequence}

        if self.use_mmd:
            self.mmd_loss = ConditionalMMDLoss(grade_sequence=grade_sequence)
            self.lambda_mmd = lambda_mmd

        self._supcon_normalizer = LossScaleNormalizer()
        self._mmd_normalizer    = LossScaleNormalizer()

        # Term 1: within-US, no domain labels
        self.within_supcon = OrdinalNegWeightedSupConLossv2(
            temperature=temperature,
            neg_strength=neg_strength,
            max_label_distance=len(grade_sequence) - 1,
            domain_aware=False,
        )

        # Term 2: cross-modal, domain-aware
        self.cross_supcon = OrdinalNegWeightedSupConLossv2(
            temperature=temperature,
            neg_strength=neg_strength,
            max_label_distance=len(grade_sequence) - 1,
            domain_aware=True,
            domain_boost=domain_boost,
        )

        self.cross_supcon_mri = OrdinalNegWeightedSupConLossv2(
            temperature=temperature,
            neg_strength=neg_strength,
            max_label_distance=len(grade_sequence) - 1,
            domain_aware=True,
            domain_boost=domain_boost,
        )

    # ---------------------------------------------------------------------- #
    # Bank management                                                          #
    # ---------------------------------------------------------------------- #

    def _update_banks(
        self,
        us_embeds: torch.Tensor,       # (B, D), normalized
        histo_embeds: torch.Tensor,    # (B, D), normalized
        labels_list: list[int],
    ) -> None:
        for us_e, hist_e, lbl in zip(us_embeds, histo_embeds, labels_list):
            lbl = int(lbl)
            if lbl not in self.us_bank:
                continue

            us_e_cpu   = us_e.detach().cpu()
            hist_e_cpu = hist_e.detach().cpu()

            if self.momentum > 0:
                # EMA update: running mean stored in _ema buffers
                if self._us_ema[lbl] is None:
                    self._us_ema[lbl]    = us_e_cpu.clone()
                    self._histo_ema[lbl] = hist_e_cpu.clone()
                else:
                    m = self.momentum
                    self._us_ema[lbl]    = m * self._us_ema[lbl]    + (1 - m) * us_e_cpu
                    self._histo_ema[lbl] = m * self._histo_ema[lbl] + (1 - m) * hist_e_cpu
                    # Re-normalize after EMA update
                    self._us_ema[lbl]    = F.normalize(self._us_ema[lbl], dim=0)
                    self._histo_ema[lbl] = F.normalize(self._histo_ema[lbl], dim=0)

                self.us_bank[lbl].append(self._us_ema[lbl].clone())
                self.histo_bank[lbl].append(self._histo_ema[lbl].clone())
            else:
                self.us_bank[lbl].append(us_e_cpu)
                self.histo_bank[lbl].append(hist_e_cpu)

    # ---------------------------------------------------------------------- #
    # Batch construction helpers                                               #
    # ---------------------------------------------------------------------- #

    def _build_within_us_batch(
        self,
        us_embeds: torch.Tensor,
        labels_list: list[int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Current-batch US embeddings + bank augmentation → (embeds, label_indices)."""
        all_embeds: list[torch.Tensor] = list(us_embeds)
        all_labels: list[int]          = [int(l) for l in labels_list]

        for grade in self.grade_sequence:
            pool = list(self.us_bank[grade])
            n    = min(self.within_bank_n, len(pool))
            if n > 0:
                sampled = random.sample(pool, n)
                all_embeds.extend(sampled)
                all_labels.extend([grade] * n)

        embeds = torch.stack([
            e.to(device) if not e.is_cuda else e for e in all_embeds
        ])
        label_idx = torch.tensor(
            [self.grade_to_idx[g] for g in all_labels],
            device=device,
        )
        return embeds, label_idx

    def _build_cross_modal_batch(
        self,
        us_embeds: torch.Tensor,
        histo_embeds: torch.Tensor,
        labels_list: list[int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Current-batch US + histo + bank augmentation for both modalities.
        Returns (embeds, label_indices, domain_labels) where domain 0=US, 1=histo.
        """
        all_embeds:  list[torch.Tensor] = []
        all_labels:  list[int]          = []
        all_domains: list[int]          = []

        # Current batch
        for e, l in zip(us_embeds, labels_list):
            all_embeds.append(e)
            all_labels.append(int(l))
            all_domains.append(0)

        for e, l in zip(histo_embeds, labels_list):
            all_embeds.append(e)
            all_labels.append(int(l))
            all_domains.append(1)

        # Bank augmentation
        for grade in self.grade_sequence:
            us_pool    = list(self.us_bank[grade])
            histo_pool = list(self.histo_bank[grade])

            n_us    = min(self.cross_bank_n, len(us_pool))
            n_histo = min(self.cross_bank_n, len(histo_pool))

            for e in random.sample(us_pool, n_us):
                all_embeds.append(e)
                all_labels.append(grade)
                all_domains.append(0)

            for e in random.sample(histo_pool, n_histo):
                all_embeds.append(e)
                all_labels.append(grade)
                all_domains.append(1)

        embeds = torch.stack([
            e.to(device) if not e.is_cuda else e for e in all_embeds
        ])
        label_idx = torch.tensor(
            [self.grade_to_idx[g] for g in all_labels],
            device=device,
        )
        domain_t = torch.tensor(all_domains, device=device)
        return embeds, label_idx, domain_t

    def _update_mri_bank(self, mri_embeds: torch.Tensor, labels_list: list[int]) -> None:
        for mri_e, lbl in zip(mri_embeds, labels_list):
            lbl = int(lbl)
            if lbl not in self.mri_bank:
                continue
            mri_e_cpu = mri_e.detach().cpu()
            if self.momentum > 0:
                if self._mri_ema[lbl] is None:
                    self._mri_ema[lbl] = mri_e_cpu.clone()
                else:
                    self._mri_ema[lbl] = (
                        self.momentum * self._mri_ema[lbl]
                        + (1 - self.momentum) * mri_e_cpu
                    )
                    self._mri_ema[lbl] = F.normalize(self._mri_ema[lbl], dim=0)
                self.mri_bank[lbl].append(self._mri_ema[lbl].clone())
            else:
                self.mri_bank[lbl].append(mri_e_cpu)


    def _build_cross_modal_batch_mri(
        self,
        us_embeds: torch.Tensor,
        mri_embeds: torch.Tensor,
        labels_list: list[int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        all_embeds:  list[torch.Tensor] = []
        all_labels:  list[int]          = []
        all_domains: list[int]          = []

        for e, l in zip(us_embeds, labels_list):
            all_embeds.append(e); all_labels.append(int(l)); all_domains.append(0)
        for e, l in zip(mri_embeds, labels_list):
            all_embeds.append(e); all_labels.append(int(l)); all_domains.append(1)

        for grade in self.grade_sequence:
            us_pool  = list(self.us_bank[grade])
            mri_pool = list(self.mri_bank[grade])
            for e in random.sample(us_pool,  min(self.cross_bank_n, len(us_pool))):
                all_embeds.append(e); all_labels.append(grade); all_domains.append(0)
            for e in random.sample(mri_pool, min(self.cross_bank_n, len(mri_pool))):
                all_embeds.append(e); all_labels.append(grade); all_domains.append(1)

        embeds    = torch.stack([e.to(device) if not e.is_cuda else e for e in all_embeds])
        label_idx = torch.tensor([self.grade_to_idx[g] for g in all_labels], device=device)
        domain_t  = torch.tensor(all_domains, device=device)
        return embeds, label_idx, domain_t

    # ---------------------------------------------------------------------- #
    # Diagnostics                                                              #
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def _log_diagnostics(
        self,
        within_embeds: torch.Tensor,
        within_labels: torch.Tensor,
        within_loss: torch.Tensor,
        cross_loss: torch.Tensor,
        total_loss: torch.Tensor,
        mmd: torch.Tensor,
    ) -> None:
        """
        Log pos/neg similarity gap to verify the encoder is responding to the signal.
        A healthy run shows pos_sim > neg_sim and the gap widening over training.
        """
        sim = torch.matmul(within_embeds, within_embeds.t())
        n   = within_embeds.size(0)
        self_mask = torch.eye(n, dtype=torch.bool, device=within_embeds.device)

        labels_equal = within_labels.unsqueeze(0) == within_labels.unsqueeze(1)
        pos_mask     = labels_equal & ~self_mask
        neg_mask     = (~labels_equal) & ~self_mask

        pos_sim = (sim * pos_mask.float()).sum() / pos_mask.float().sum().clamp(min=1)
        neg_sim = (sim * neg_mask.float()).sum() / neg_mask.float().sum().clamp(min=1)

        bank_sizes_us    = {g: len(self.us_bank[g])    for g in self.grade_sequence}
        bank_sizes_histo = {g: len(self.histo_bank[g]) for g in self.grade_sequence}

        logging.info(
            f"[SupConLoss] step={self._num_computed} | "
            f"within={within_loss.item():.4f} | "
            f"cross={cross_loss.item():.4f} | "
            f"total={total_loss.item():.4f} | "
            f"mmd_loss={mmd.item():.4f} | "
            f"pos_sim={pos_sim.item():.4f} | "
            f"neg_sim={neg_sim.item():.4f} | "
            f"gap={pos_sim.item() - neg_sim.item():.4f} | "
            f"us_bank={bank_sizes_us} | "
            f"histo_bank={bank_sizes_histo}"
        )

    def forward(self, data: dict) -> torch.Tensor:
            three_way = (
                "joint_positive_hist" in data and data["joint_positive_hist"] is not None
                and "joint_positive_mri" in data and data["joint_positive_mri"] is not None
            )
            hist_key  = "joint_positive_hist" if three_way else "positive_hist"
            label_key = "grade_group"

            required_keys = ["image_feats_needle", hist_key, label_key]
            missing = [k for k in required_keys if k not in data or data[k] is None]
            if missing:
                self._num_skipped += 1
                logging.warning(
                    f"[WithinModalSupConLoss] skipped (total={self._num_skipped}). "
                    f"Missing keys: {missing}"
                )
                return torch.tensor(
                    0.0,
                    device=next(iter(v for v in data.values() if v is not None)).device,
                    requires_grad=True,
                )

            device       = data["image_feats_needle"].device
            us_embeds    = F.normalize(data["image_feats_needle"], dim=1)       # (B, D)
            histo_embeds = F.normalize(data[hist_key].to(device),  dim=1)       # (B, D)
            labels_list  = [int(g) for g in data[label_key].cpu().tolist()]
            z_cross = us_embeds

            # Bank updates always use raw us_embeds, not predicted,
            # so the within-US bank reflects the actual encoder state
            self._update_banks(us_embeds, histo_embeds, labels_list)
            if three_way:
                mri_embeds = F.normalize(data["joint_positive_mri"].to(device), dim=1)
                self._update_mri_bank(mri_embeds, labels_list)

            # ------------------------------------------------------------------ #
            # Term 1: within-US SupCon — operates on raw us_embeds               #
            # ------------------------------------------------------------------ #
            within_embeds, within_labels = self._build_within_us_batch(
                us_embeds, labels_list, device
            )
            within_loss = self.within_supcon(
                embeddings=within_embeds,
                labels=within_labels,
            )

            # ------------------------------------------------------------------ #
            # Term 2: cross-modal SupCon — operates on z_cross      #
            # z_cross is in translated histo space so SupCon operates on         #
            # geometrically compatible embeddings from both modalities            #
            # ------------------------------------------------------------------ #
            cross_embeds, cross_labels, cross_domains = self._build_cross_modal_batch(
                z_cross, histo_embeds, labels_list, device
            )
            cross_loss_histo = self.cross_supcon(
                embeddings=cross_embeds,
                labels=cross_labels,
                domain_labels=cross_domains,
            )

            # ------------------------------------------------------------------ #
            # Term 3: cross-modal MRI (three-way)                                 #
            # ------------------------------------------------------------------ #
            cross_loss_mri = torch.tensor(0.0, device=device)
            if three_way:
                if self.predictor is not None and self.use_predictor_for_cross:
                    z_cross_mri = self.predictor(us_embeds)
                else:
                    z_cross_mri = us_embeds
                cross_embeds_mri, cross_labels_mri, cross_domains_mri = (
                    self._build_cross_modal_batch_mri(
                        z_cross_mri, mri_embeds, labels_list, device
                    )
                )
                cross_loss_mri = self.cross_supcon_mri(
                    embeddings=cross_embeds_mri,
                    labels=cross_labels_mri,
                    domain_labels=cross_domains_mri,
                )

            # ------------------------------------------------------------------ #
            # Term 4: MMD (optional, now operates on z_cross if predictor active) #
            # ------------------------------------------------------------------ #
            mmd = torch.tensor(0.0, device=device)
            if self.use_mmd:
                mmd = self.mmd_loss(
                    z_cross, histo_embeds, labels_list, device=device,
                    us_bank=self.us_bank,
                    histo_bank=self.histo_bank,
                )
            # ------------------------------------------------------------------ #
            # Normalize and combine                                                #
            # ------------------------------------------------------------------ #
            within_loss_n      = self._supcon_normalizer.normalize(within_loss)
            cross_loss_histo_n = self._supcon_normalizer.normalize(cross_loss_histo)
            mmd_n             = self._mmd_normalizer.normalize(mmd)

            loss = (
                self.lambda_within * within_loss_n
                + self.lambda_cross * cross_loss_histo_n
                + self.lambda_mri   * cross_loss_mri
                + self.lambda_mmd   * mmd_n
            )

            self._num_computed += 1
            if self._num_computed % 100 == 0:
                self._log_diagnostics(
                    within_embeds, within_labels,
                    within_loss, cross_loss_histo, loss, mmd,
                )
                if three_way:
                    logging.info(
                        f"[WithinModalSupConLoss] cross_loss_mri={cross_loss_mri.item():.4f}"
                    )
            return loss



def build_loss(args):
    losses  = []
    weights = []
    names   = []

    if args.loss_type == 'isuploss':
        losses.append(ISUPLoss())
        weights.append(1.0)
        names.append('isuploss')

    elif args.loss_type == 'vsupcon':
        losses.append(Vsupcon())
        weights.append(1.0)
        names.append('vsupcon')

    elif args.loss_type == 'mmd_supcon':
        if args.train_mode == 'us_only':
            losses.append(OrdinalNegWeightedSupConLossv2())
            weights.append(1.0)
            names.append('us_supcon')
        else:
            losses.append(WithinModalSupConLossv2(
                lambda_cross = args.lambda_cross,
                lambda_mri = args.lambda_mri,
                lambda_within = args.lambda_within,
                lambda_mmd = args.lambda_mmd,
                neg_strength = args.neg_strength,
                domain_boost = args.domain_boost,
                within_bank_n = args.within_bank_n,
                cross_bank_n = args.cross_bank_n,
                bank_size = args.bank_size,
                use_mmd = args.use_mmd))
            weights.append(1.0)
            names.append('withinsupcon')

            if args.loss_type_reg == 'propbce':
                prop_bce = NeedleProportionBCE(
                    proj_dim            = args.proj_dim,
                    treat_gg1_as_benign = getattr(args, 'treat_gg1_as_benign', False),
                    pos_weight          = getattr(args, 'prop_pos_weight', 1.0),
                ).to(args.device)
                losses.append(prop_bce)
                weights.append(getattr(args, 'lambda_prop', 1.0))
                names.append('prop_bce')

            elif args.loss_type_reg == 'aem':
                aem_loss = AttentionEntropyMaximization(lambda_aem=args.lambda_aem)  # sweep if time allows
                losses.append(aem_loss)
                weights.append(1.0)
                names.append('aem')

    elif args.loss_type == 'triplet':
        if args.train_mode == 'us_only':
            losses.append(TripletLossUS(margin=5.0, p=2))
            weights.append(1.0)
            names.append('tripletus')

        elif args.train_mode == 'us_histo':
            losses.append(TripletLoss(margin=5.0, p=2, mode='hist'))
            weights.append(1.0)
            names.append('triplet_hist')

            if args.loss_type_reg == 'propbce':
                prop_bce = NeedleProportionBCE(
                    proj_dim            = args.proj_dim,
                    treat_gg1_as_benign = getattr(args, 'treat_gg1_as_benign', False),
                    pos_weight          = getattr(args, 'prop_pos_weight', 1.0),
                ).to(args.device)
                losses.append(prop_bce)
                weights.append(getattr(args, 'lambda_prop', 1.0))
                names.append('prop_bce')

    return SumLoss(losses, weights, names, normalize=args.normalize)

def get_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--loss", default="needle_region_ce")
    parser.add_argument(
        "--outside_prostate_penalty",
        action="store_true",
        default=False,
        help="Whether to penalize the model for making predictions outside the prostate region.",
    )
    return parser


