"""
attention_regularization_baselines.py
=======================================
Faithful implementations of two attention-regularization baselines from the
weakly-supervised MIL literature, for comparison against PropBCE-induced
heterogeneity in this work.

1. ACMIL (Zhang et al., ECCV 2024) — "Attention-Challenging Multiple
   Instance Learning for Whole Slide Image Classification"
   https://arxiv.org/abs/2311.07125
   Combines:
     (a) Multiple Branch Attention (MBA) — M parallel attention heads/branches
         instead of one, each encouraged to attend to different instances.
     (b) Stochastic Top-K Instance Masking (STKIM) — randomly masks a subset
         of the top-K attended instances during training, forcing the model
         to spread attention to previously-ignored instances.

2. ADR / AEM (Zhang et al., 2024) — "Attention Entropy Maximization for
   Multiple Instance Learning based Whole Slide Image Classification"
   https://arxiv.org/abs/2406.15303
   A single plug-and-play regularization term: a negative entropy penalty
   added to the training loss, directly rewarding higher attention entropy
   (i.e. less concentrated attention), with one hyperparameter (lambda).

Both methods were designed to counter attention OVER-CONCENTRATION
(collapse onto too few instances, linked to overfitting in supervised WSI
classification). This is the opposite failure mode from the one studied in
this work (collapse toward near-UNIFORM attention under weak/unpaired
cross-modal supervision). They are included as baselines to test whether
entropy-increasing regularization — designed for a different degeneracy —
also resolves attention collapse-to-uniformity, or whether it is
orthogonal/insufficient compared to supervision-induced heterogeneity
(PropBCE).

Both modules are designed to attach to the existing ABMILISUP /
NeedleABMILWrapper pipeline with minimal changes.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
#  1. ACMIL — Multiple Branch Attention + Stochastic Top-K Instance Masking
# ══════════════════════════════════════════════════════════════════════════

class MultiBranchGatedAttention(nn.Module):
    """
    Multiple Branch Attention (MBA), as used in ACMIL.

    Replaces the single gated-attention head in ABMILISUP with M parallel
    attention branches. Each branch produces its own attention distribution
    over the N instances; branch outputs are averaged (post hoc, after
    pooling) to form the final bag representation, encouraging different
    branches to specialize on different discriminative instances.

    This mirrors the ACMIL reference implementation's `arch="ga"` (gated
    attention) variant with `n_token=M` branches.

    Parameters
    ----------
    proj_dim    : input feature dimension (matches ABMILISUP's H)
    attn_hidden : hidden dimension for the gated attention MLP
    n_branches  : number of attention branches (M). ACMIL's paper finds
                  M=5 near-optimal across datasets; default follows this.
    p_attn_dropout : dropout applied to the attention gate, matching
                  ABMILISUP's existing attn_do for a fair comparison.
    """

    def __init__(
        self,
        proj_dim: int = 768,
        attn_hidden: int = 512,
        n_branches: int = 5,
        p_attn_dropout: float = 0.15,
    ):
        super().__init__()
        self.n_branches = n_branches

        # One (V, U, w) gated-attention triple per branch
        self.attn_V = nn.ModuleList(
            [nn.Linear(proj_dim, attn_hidden) for _ in range(n_branches)]
        )
        self.attn_U = nn.ModuleList(
            [nn.Linear(proj_dim, attn_hidden) for _ in range(n_branches)]
        )
        self.attn_w = nn.ModuleList(
            [nn.Linear(attn_hidden, 1) for _ in range(n_branches)]
        )
        self.attn_do = nn.Dropout(p_attn_dropout)

    def forward(
        self,
        H: torch.Tensor,                  # (B, N, D)
        mask: Optional[torch.Tensor] = None,  # (B, N) bool, True = valid
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        A_branches : (B, M, N) attention weights per branch (softmax'd,
                     masked positions set to 0)
        z_branches : (B, M, D) bag embedding per branch
        """
        B, N, D = H.shape
        device = H.device
        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=device)

        A_list, z_list = [], []
        for m in range(self.n_branches):
            Vh = torch.tanh(self.attn_V[m](H))
            Uh = torch.sigmoid(self.attn_U[m](H))
            gate = self.attn_do(Vh * Uh)
            A_m = self.attn_w[m](gate).squeeze(-1)             # (B, N)
            A_m = A_m.masked_fill(~mask, torch.finfo(A_m.dtype).min)
            A_m = F.softmax(A_m, dim=1)
            z_m = torch.bmm(A_m.unsqueeze(1), H).squeeze(1)    # (B, D)
            A_list.append(A_m)
            z_list.append(z_m)

        A_branches = torch.stack(A_list, dim=1)   # (B, M, N)
        z_branches = torch.stack(z_list, dim=1)   # (B, M, D)
        return A_branches, z_branches


def stochastic_topk_instance_masking(
    A: torch.Tensor,           # (B, N) attention weights, pre- or post-softmax-logits
    k: int = 10,
    mask_drop_prob: float = 0.6,
    apply_to_logits: bool = True,
) -> torch.Tensor:
    """
    Stochastic Top-K Instance Masking (STKIM), as used in ACMIL.

    Identifies the top-K attended instances per sample, then randomly masks
    out a fraction (`mask_drop_prob`) of them — forcing their attention mass
    to be redistributed to the remaining (previously lower-attended)
    instances on the next forward pass.

    This should be applied to attention LOGITS (pre-softmax) for correct
    redistribution; if `A` passed in is already post-softmax, set
    `apply_to_logits=False` and re-normalize externally.

    Parameters
    ----------
    A : (B, N) attention logits (if apply_to_logits=True) or weights
    k : number of top instances considered for masking (ACMIL default ~10)
    mask_drop_prob : probability that any given one of the top-K instances
                     is masked out this step (ACMIL default 0.6; the 2024.10
                     update note in the reference repo suggests trying 0.0
                     if training is unstable)

    Returns
    -------
    masked_A : (B, N) with masked positions set to -inf (if logits) so a
               subsequent softmax redistributes their mass.
    """
    B, N = A.shape
    k = min(k, N)
    device = A.device

    masked_A = A.clone()

    topk_vals, topk_idx = A.topk(k, dim=1)  # (B, k)
    drop_mask = (torch.rand(B, k, device=device) < mask_drop_prob)  # (B, k)

    fill_value = float("-inf") if apply_to_logits else 0.0
    for b in range(B):
        idx_to_drop = topk_idx[b][drop_mask[b]]
        if idx_to_drop.numel() > 0:
            masked_A[b, idx_to_drop] = fill_value

    return masked_A


class ACMILHead(nn.Module):
    """
    Drop-in replacement for ABMILISUP's single-branch attention + pooling,
    combining Multiple Branch Attention and Stochastic Top-K Instance
    Masking, matching the ACMIL reference design.

    Use in place of the attn_V/attn_U/attn_w + softmax + bmm block inside
    ABMILISUP, keeping the surrounding proj/proj_bn and classification head
    unchanged for a fair comparison.

    Parameters
    ----------
    proj_dim       : projected patch feature dimension
    attn_hidden    : hidden dim for gated attention MLP
    n_branches     : M, number of attention branches (default 5, ACMIL paper)
    n_masked_patch : K, number of top instances considered for STKIM masking
                     (default 10, ACMIL paper)
    mask_drop_prob : STKIM masking probability (default 0.6, ACMIL paper;
                     try 0.0 if unstable, per ACMIL repo 2024.10 update note)
    p_attn_dropout : attention dropout, matches ABMILISUP for fair comparison
    training_only_masking : STKIM is only applied during training, never at
                     eval/inference (matches reference implementation)
    """

    def __init__(
        self,
        proj_dim: int = 768,
        attn_hidden: int = 512,
        n_branches: int = 5,
        n_masked_patch: int = 10,
        mask_drop_prob: float = 0.6,
        p_attn_dropout: float = 0.15,
        training_only_masking: bool = True,
    ):
        super().__init__()
        self.mba = MultiBranchGatedAttention(
            proj_dim=proj_dim,
            attn_hidden=attn_hidden,
            n_branches=n_branches,
            p_attn_dropout=p_attn_dropout,
        )
        self.n_masked_patch = n_masked_patch
        self.mask_drop_prob = mask_drop_prob
        self.training_only_masking = training_only_masking
        self.n_branches = n_branches

    def forward(
        self,
        H: torch.Tensor,                      # (B, N, D)
        mask: Optional[torch.Tensor] = None,  # (B, N) bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        z        : (B, D) final bag embedding — mean over branches
        A_mean   : (B, N) attention averaged over branches, for diagnostics
                   (entropy, effective-k, visualization) under the same
                   logging convention used elsewhere in this work
        A_branches : (B, M, N) raw per-branch attention, kept for inspection
        """
        B, N, D = H.shape
        device = H.device
        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=device)

        if self.training and self.n_masked_patch > 0:
            # Apply STKIM per branch on pre-softmax logits, then re-softmax.
            # We recompute branch logits explicitly here for masking control.
            A_list, z_list = [], []
            for m in range(self.mba.n_branches):
                Vh = torch.tanh(self.mba.attn_V[m](H))
                Uh = torch.sigmoid(self.mba.attn_U[m](H))
                gate = self.mba.attn_do(Vh * Uh)
                logits_m = self.mba.attn_w[m](gate).squeeze(-1)  # (B, N)
                logits_m = logits_m.masked_fill(~mask, torch.finfo(logits_m.dtype).min)

                logits_m = stochastic_topk_instance_masking(
                    logits_m,
                    k=self.n_masked_patch,
                    mask_drop_prob=self.mask_drop_prob,
                    apply_to_logits=True,
                )
                A_m = F.softmax(logits_m, dim=1)
                z_m = torch.bmm(A_m.unsqueeze(1), H).squeeze(1)
                A_list.append(A_m)
                z_list.append(z_m)
            A_branches = torch.stack(A_list, dim=1)  # (B, M, N)
            z_branches = torch.stack(z_list, dim=1)  # (B, M, D)
        else:
            A_branches, z_branches = self.mba(H, mask=mask)

        z = z_branches.mean(dim=1)        # (B, D) — average bag embedding across branches
        A_mean = A_branches.mean(dim=1)   # (B, N) — average attention across branches
        # Renormalize A_mean so it sums to 1 (mean of M softmax distributions
        # already sums to 1, but guard against numerical drift)
        A_mean = A_mean / A_mean.sum(dim=1, keepdim=True).clamp(min=1e-8)

        return z, A_mean, A_branches


@torch.no_grad()
def branch_diversity_stats(A_branches, mask=None, eps=1e-8):
    """Measures how much the M attention branches actually differ.

    A_branches : (B, M, N) per-branch attention distributions (post-softmax)
    mask       : (B, N) bool, True = valid. Padded positions are excluded and
                 distributions renormalized over valid elements only, so
                 padding doesn't inflate apparent agreement.

    Returns:
      mean_pairwise_l1    : mean L1 between branch pairs, [0, 2].
                            0 = collapsed, 2 = disjoint support.
      mean_pairwise_js    : mean Jensen-Shannon divergence, [0, ln2].
      mean_branch_entropy : mean entropy of individual branches. Low = peaked.
      mean_attn_entropy   : entropy of the branch-averaged distribution.
                            Much higher than branch_H = branches peaked on
                            DIFFERENT elements (desired). Nearly equal = agreement.
      argmax_agreement    : fraction of branch pairs sharing a top element.
                            1.0 = all branches pick the same frame.
    """
    B, M, N = A_branches.shape
    A = A_branches.float()

    if mask is not None:
        m = mask.unsqueeze(1).expand(B, M, N)
        A = A * m
        A = A / A.sum(dim=-1, keepdim=True).clamp(min=eps)

    stats = {}

    if M > 1:
        i, j = torch.triu_indices(M, M, offset=1)
        Ai, Aj = A[:, i], A[:, j]
        stats["mean_pairwise_l1"] = (Ai - Aj).abs().sum(-1).mean().item()

        Am = 0.5 * (Ai + Aj)
        def _kl(p, q):
            return (p * ((p + eps).log() - (q + eps).log())).sum(-1)
        js = 0.5 * _kl(Ai, Am) + 0.5 * _kl(Aj, Am)
        stats["mean_pairwise_js"] = js.mean().item()

        top = A.argmax(dim=-1)
        stats["argmax_agreement"] = (top[:, i] == top[:, j]).float().mean().item()
    else:
        stats["mean_pairwise_l1"] = 0.0
        stats["mean_pairwise_js"] = 0.0
        stats["argmax_agreement"] = 1.0

    ent = -(A * (A + eps).log()).sum(-1)
    stats["mean_branch_entropy"] = ent.mean().item()

    A_mean = A.mean(dim=1)
    A_mean = A_mean / A_mean.sum(-1, keepdim=True).clamp(min=eps)
    stats["mean_attn_entropy"] = -(A_mean * (A_mean + eps).log()).sum(-1).mean().item()

    if mask is not None:
        stats["max_possible_entropy"] = mask.sum(-1).float().clamp(min=1).log().mean().item()
    else:
        stats["max_possible_entropy"] = torch.tensor(float(N)).log().item()

    return stats


def print_branch_diversity(A_branches, mask=None, tag=""):
    s = branch_diversity_stats(A_branches, mask=mask)
    print(
        f"[branch diversity{' ' + tag if tag else ''}] "
        f"pairwise_L1={s['mean_pairwise_l1']:.4f}  "
        f"pairwise_JS={s['mean_pairwise_js']:.4f}  "
        f"argmax_agree={s['argmax_agreement']:.3f}  "
        f"branch_H={s['mean_branch_entropy']:.4f}  "
        f"mean_H={s['mean_attn_entropy']:.4f}  "
        f"(max_H={s['max_possible_entropy']:.4f})"
    )
    return s


# ══════════════════════════════════════════════════════════════════════════
#  2. ADR / AEM — Attention Entropy Maximization
# ══════════════════════════════════════════════════════════════════════════

class AttentionEntropyMaximization(nn.Module):
    """
    ADR / AEM (Attention Entropy Maximization), Zhang et al. 2024.
    https://arxiv.org/abs/2406.15303

    A single regularization term added to the training loss: the negative
    entropy of the attention distribution, encouraging HIGHER entropy
    (less concentrated / more uniform-ish, but away from degenerate
    over-concentration) over training.

    Loss term (to be ADDED, not subtracted, to the total loss — minimizing
    negative entropy is equivalent to maximizing entropy):

        L_AEM = -lambda * mean_i [ -sum_j A[i,j] * log(A[i,j] + eps) ]
              =  lambda * mean_i [ sum_j A[i,j] * log(A[i,j] + eps) ]

    This is the exact opposite sign convention from this work's own
    AttentionEntropyRegularizer (which penalizes entropy BELOW a minimum
    threshold to prevent collapse toward a single instance). AEM has no
    threshold — it always pushes entropy up, unconditionally, for as long
    as it is active. This is intentional: AEM was designed to counter
    over-concentration, not under-concentration, and is included here
    exactly as published for a faithful baseline comparison.

    Parameters
    ----------
    lambda_aem : weight on the entropy-maximization term. The original
                 paper notes performance is sensitive to this value; sweep
                 e.g. {0.01, 0.05, 0.1, 0.5} if reproducing their tuning
                 protocol is in scope.
    """

    def __init__(self, lambda_aem: float = 0.1):
        super().__init__()
        self.lambda_aem = lambda_aem

    def forward(self, data: dict) -> torch.Tensor:
        """
        Expects data["attention"]: (B, N) softmax attention weights, ON
        the computation graph (not detached), exactly as logged elsewhere
        in this work for entropy diagnostics.
        """
        A = data["attention"]
        device = A.device
        entropy = -(A * torch.log(A.clamp(min=1e-8))).sum(dim=1)  # (B,)
        # Minimizing -entropy == maximizing entropy
        loss = -entropy.mean()
        return self.lambda_aem * loss


# ══════════════════════════════════════════════════════════════════════════
#  Convenience: per-grade diagnostics, matching this work's existing
#  logging convention (entropy, max_attn) for direct comparability.
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def log_branch_attention_diagnostics(
    A_mean: torch.Tensor,        # (B, N)
    grades: torch.Tensor,        # (B,)
    grade_sequence=range(6),
    log_fn=log.info,
) -> None:
    """
    Logs per-grade attention entropy / max_attn for the ACMIL head's mean
    (branch-averaged) attention, in the same format used elsewhere in this
    work, so ACMIL's attention dynamics are directly comparable to the
    proposed method's entropy curves.
    """
    for g in grade_sequence:
        mask = (grades == g)
        if mask.sum() == 0:
            continue
        A_g = A_mean[mask]
        entropy = -(A_g * torch.log(A_g.clamp(min=1e-8))).sum(dim=1).mean()
        max_attn = A_g.max(dim=1).values.mean()
        log_fn(
            f"GG{g} attention entropy: {entropy.item():.4f} | "
            f"max_attn: {max_attn.item():.4f} | n_cores={mask.sum().item()}"
        )


