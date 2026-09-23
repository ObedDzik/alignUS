"""
supcon_vanilla.py — plain supervised contrastive loss (Khosla et al., 2020).

Self-contained baseline. Takes the data dict, same as WithinModalSupConLossv2:

    criterion = VanillaSupConLoss(temperature=0.07, lambda_within=1.0, lambda_cross=1.0)
    loss = criterion(data)

No memory bank. No ordinal negative weighting. No domain boost. No MMD, no
predictor, no MRI term. Positives are same-grade pairs, negatives are
different-grade pairs, every off-diagonal pair carries weight 1.

    L_i = -1/|P(i)| * sum_{p in P(i)} log[ exp(s_ip) / sum_{a != i} exp(s_ia) ]
    s_ij = <z_i, z_j> / T

Reads from data:
    image_feats_needle              (B, D)  US bag embeddings
    positive_hist / joint_positive_hist  (B, D)  grade-matched histo embeddings
    grade_group                     (B,)    ISUP grade
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


def supcon(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """Standard SupCon (L_out form) over one batch. Embeddings must be L2-normalized."""
    n = embeddings.size(0)
    device = embeddings.device
    labels = labels.view(-1).to(device)

    sim = torch.matmul(embeddings, embeddings.t()) / temperature

    off_diag = ~torch.eye(n, dtype=torch.bool, device=device)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & off_diag

    if pos_mask.sum() == 0:
        return embeddings.sum() * 0.0

    sim_stable = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim_stable)
    denom = (exp_sim * off_diag.float()).sum(dim=1, keepdim=True) + 1e-8

    log_prob = sim_stable - torch.log(denom)
    pos_count = pos_mask.sum(dim=1).clamp(min=1).float()
    return -((log_prob * pos_mask.float()).sum(dim=1) / pos_count).mean()


class VanillaSupConLoss(nn.Module):
    """
    Two-term plain SupCon, mirroring the term structure of your method so the
    comparison is on the objective rather than on which terms exist:

        within : SupCon over the current batch of US embeddings
        cross  : SupCon over the concatenated US + histo batch

    Set lambda_cross=0 for US-only rows — ProstNFoundMeta zeroes positive_hist in
    us_only mode, and the cross term would silently run on all-zero embeddings
    rather than raising.

    Note for the paper: this baseline has no memory bank, so it differs from your
    method in two ways at once (weighting AND bank). If you want the weighting
    isolated on its own, that is the neg_strength=0 / domain_boost=0 run of your
    own class, not this one. Both rows are worth having and they answer different
    questions.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        lambda_within: float = 1.0,
        lambda_cross: float = 1.0,
        log_every: int = 100,
    ):
        super().__init__()
        self.temperature = max(float(temperature), 1e-6)
        self.lambda_within = float(lambda_within)
        self.lambda_cross = float(lambda_cross)
        self.log_every = int(log_every)
        self._num_computed = 0
        self._num_skipped = 0

    def forward(self, data: dict) -> torch.Tensor:
        hist_key = (
            "joint_positive_hist"
            if data.get("joint_positive_hist") is not None
            else "positive_hist"
        )

        required = ["image_feats_needle", "grade_group"]
        if self.lambda_cross != 0:
            required.append(hist_key)
        missing = [k for k in required if k not in data or data[k] is None]
        if missing:
            self._num_skipped += 1
            logging.warning(
                f"[VanillaSupConLoss] skipped (total={self._num_skipped}). "
                f"Missing keys: {missing}"
            )
            ref = next(v for v in data.values() if isinstance(v, torch.Tensor))
            return torch.tensor(0.0, device=ref.device, requires_grad=True)

        device = data["image_feats_needle"].device
        us = F.normalize(data["image_feats_needle"], dim=1)          # (B, D)
        labels = data["grade_group"].view(-1).to(device)             # (B,)

        # --- Term 1: within-US ---
        within_loss = supcon(us, labels, self.temperature)

        # --- Term 2: cross-modal, US and histo in one batch ---
        cross_loss = torch.tensor(0.0, device=device)
        if self.lambda_cross != 0:
            histo = F.normalize(data[hist_key].to(device), dim=1)    # (B, D)
            embeds = torch.cat([us, histo], dim=0)                   # (2B, D)
            cross_labels = torch.cat([labels, labels], dim=0)        # (2B,)
            cross_loss = supcon(embeds, cross_labels, self.temperature)

        loss = self.lambda_within * within_loss + self.lambda_cross * cross_loss

        self._num_computed += 1
        if self.log_every and self._num_computed % self.log_every == 0:
            self._log(us, labels, within_loss, cross_loss, loss)
        return loss

    @torch.no_grad()
    def _log(self, us, labels, within_loss, cross_loss, total):
        sim = us @ us.t()
        n = us.size(0)
        off = ~torch.eye(n, dtype=torch.bool, device=us.device)
        pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & off
        neg = (~(labels.unsqueeze(0) == labels.unsqueeze(1))) & off
        pos_sim = (sim * pos.float()).sum() / pos.float().sum().clamp(min=1)
        neg_sim = (sim * neg.float()).sum() / neg.float().sum().clamp(min=1)
        logging.info(
            f"[VanillaSupCon] step={self._num_computed} | "
            f"within={within_loss.item():.4f} | cross={cross_loss.item():.4f} | "
            f"total={total.item():.4f} | pos_sim={pos_sim.item():.4f} | "
            f"neg_sim={neg_sim.item():.4f} | gap={(pos_sim - neg_sim).item():.4f}"
        )