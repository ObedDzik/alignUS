"""
diagnostics.py — model-agnostic diagnostics for AlignUS / GUIDE-US / ACMIL / AEM / SupCon-Only.

Everything here works for every method, including ones with no patch_cancer_head,
so the same figure can carry all baselines on one axis.

Logged per (global_step, grade):
  - normalized attention entropy  H / ln(N_valid)          [padding-aware]
  - attention concentration       max attn, top-10% mass
  - cosine dispersion             1 - mean_{i!=j} cos(h_i, h_j)   [scale-free]
  - dispersion index              std_N(H) / (||H|| / sqrt(D))    [scale-free]
  - effective rank                exp(entropy of singular value spectrum)
  - patch score mean / within-core std   [only when a patch_cancer_head exists]

Logged per epoch (validation):
  - full probability matrix + labels + ids  -> .npz  (never rerun for a metric again)
  - per-class AUC, macro AUC, binary csPCa (GG>=2) AUC
  - QWK, MAE (argmax and expectation), adjacent accuracy, confusion matrix
  - per-sample attention entropy vs. true involvement (for offline correlation)
"""

from __future__ import annotations

import logging
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

# --------------------------------------------------------------------------- #
# Attention                                                                     #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def normalized_attention_entropy(A: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Padding-aware normalized attention entropy, per sample.

    A    : (B, N) attention weights (softmax already applied over valid positions)
    mask : (B, N) bool, True on valid needle tokens

    Returns (B,) in [0, 1]. 1.0 == uniform over that sample's valid tokens.

    This differs from the naive version: N_max varies with batch composition, so
    dividing by ln(N_max) makes entropy depend on which cores happen to be batched
    together. Divide by ln(N_valid) per sample instead.
    """
    m = mask.float()
    A = A * m
    A = A / A.sum(dim=1, keepdim=True).clamp(min=1e-8)
    H = -(A * torch.log(A.clamp(min=1e-12)) * m).sum(dim=1)
    n_valid = m.sum(dim=1).clamp(min=2.0)
    return H / torch.log(n_valid)


@torch.no_grad()
def attention_concentration(A: torch.Tensor, mask: torch.Tensor) -> dict:
    """max attention weight and fraction of mass in the top 10% of valid tokens.

    Both tensors are full length (B,) — samples with <2 valid tokens get NaN
    rather than being dropped, so the result stays indexable by grade mask.
    """
    m = mask.float()
    A = A * m
    A = A / A.sum(dim=1, keepdim=True).clamp(min=1e-8)
    max_attn = A.max(dim=1).values

    top_mass = torch.full((A.shape[0],), float("nan"), device=A.device)
    for b in range(A.shape[0]):
        n = int(m[b].sum().item())
        if n < 2:
            continue
        k = max(1, int(round(0.10 * n)))
        top_mass[b] = torch.topk(A[b][mask[b]], k).values.sum()
    return {"max_attn": max_attn, "top10pct_mass": top_mass}


@torch.no_grad()
def centered_attention_similarity(A: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Cross-bag attention agreement, with the uniform component removed.

    Raw cosine between attention vectors is ~1.0 for any two distributions on the
    simplex because the 1/N component dominates. Subtract it first, or the number
    is meaningless.
    """
    m = mask.float()
    A = A * m
    A = A / A.sum(dim=1, keepdim=True).clamp(min=1e-8)
    uniform = m / m.sum(dim=1, keepdim=True).clamp(min=1e-8)
    D = A - uniform
    D = F.normalize(D, dim=1, eps=1e-8)
    S = D @ D.t()
    n = S.shape[0]
    if n < 2:
        return float("nan")
    off = ~torch.eye(n, dtype=torch.bool, device=S.device)
    return float(S[off].mean().item())


# --------------------------------------------------------------------------- #
# Patch-feature heterogeneity  (the headline mechanism metric)                   #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def patch_heterogeneity(
    H: torch.Tensor,
    mask: torch.Tensor,
    max_patches: int = 192,
    compute_effective_rank: bool = True,
) -> dict:
    """
    Within-core spatial heterogeneity of patch features. Scale-free, so it is
    comparable across methods and backbones with different feature magnitudes.

    H    : (B, N, D) patch features (post-projection H, or raw encoder tokens)
    mask : (B, N) bool, True on valid needle tokens

    Returns dict of (B_valid,) tensors:
      cosine_dispersion : 1 - mean_{i!=j} cos(h_i, h_j).  0 == all patches identical.
      dispersion_index  : std across patches / (mean norm / sqrt(D)).
                          Report THIS, not raw std — raw std scales with ||H||
                          and will differ across methods for trivial reasons.
      effective_rank    : exp(H(singular value spectrum)) of centered patches.
                          Separates "patches vary along one direction" from
                          "patches span a subspace".
      feat_norm         : mean ||h||, diagnostic only.
    """
    out = defaultdict(list)
    B, N, D = H.shape
    Hf = H.detach().float()

    for b in range(B):
        h = Hf[b][mask[b]]
        n = h.shape[0]
        if n < 3:
            continue
        if n > max_patches:
            idx = torch.randperm(n, device=h.device)[:max_patches]
            h = h[idx]
            n = max_patches

        hn = F.normalize(h, dim=-1, eps=1e-8)
        S = hn @ hn.t()
        off = ~torch.eye(n, dtype=torch.bool, device=S.device)
        out["cosine_dispersion"].append(1.0 - S[off].mean())

        norm = h.norm(dim=-1).mean()
        std_n = h.std(dim=0, unbiased=False).mean()
        out["dispersion_index"].append(std_n / (norm / (D ** 0.5) + 1e-8))
        out["feat_norm"].append(norm)

        if compute_effective_rank:
            hc = h - h.mean(dim=0, keepdim=True)
            try:
                s = torch.linalg.svdvals(hc)
                p = s / s.sum().clamp(min=1e-12)
                p = p.clamp(min=1e-12)
                out["effective_rank"].append(torch.exp(-(p * p.log()).sum()))
            except Exception:
                pass

    return {k: torch.stack(v) for k, v in out.items() if len(v)}


@torch.no_grad()
def patch_score_stats(patch_scores: torch.Tensor, mask: torch.Tensor) -> dict:
    """
    Mean and within-core std of per-patch cancer probabilities, over VALID tokens only.

    The unmasked version averages over zero-padded positions, which after
    proj -> LayerNorm -> ReLU map to a constant non-zero vector. That both dilutes
    the proportion target and contaminates the std. Always mask.

    Report mean alongside std: std of a sigmoid is mean-dependent (max std at
    mean 0.5), so per-grade std comparisons are uninterpretable without it.
    """
    m = mask.float()
    n = m.sum(dim=1).clamp(min=1.0)
    mean = (patch_scores * m).sum(dim=1) / n
    var = (((patch_scores - mean.unsqueeze(1)) ** 2) * m).sum(dim=1) / n
    std = var.clamp(min=0).sqrt()
    # variance-normalized: removes the mean-dependence of sigmoid std
    max_std = (mean * (1 - mean)).clamp(min=1e-8).sqrt()
    return {"score_mean": mean, "score_std": std, "score_std_norm": std / max_std}


# --------------------------------------------------------------------------- #
# Step-level recorder                                                           #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Cross-modal alignment  (the histopathology half of the claim)                  #
# --------------------------------------------------------------------------- #


def _row_normalize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(n, 1e-8, None)


def alignment_metrics(
    us: np.ndarray,
    histo: np.ndarray,
    labels: np.ndarray,
    prefix: str = "align/",
) -> dict:
    """
    Does the US space actually move toward histopathology, and is it ordinal?

    Everything above this point is US-side only. Without these numbers you cannot
    detect the failure mode that matters most: the histo term contributing nothing
    while the AUC gain comes entirely from L_sg and bank rebalancing.

    us    : (N, D) US bag embeddings (image_feats_needle)
    histo : (N, D) the grade-matched histo embedding for each core, taken from the
            DATASET before the model overwrites it — so this is defined even for
            us_only runs, which is exactly the unaligned reference you want.
    labels: (N,) grade group

    Key readouts:
      us_histo_margin      > 0 means US cores sit closer to their OWN grade's histo
                             centroid than to other grades'. If this is ~0 for
                             AlignUS, the alignment term is not doing its job.
      modality_gap         1 - cos(mean US, mean histo). Shrinking = gap closing.
      ordinality_us        Spearman(|Δgrade|, US centroid distance). This is the
                             ordinal-structure claim, measured directly.
      ordinality_cross     same, for US centroid -> histo centroid distances.
    """
    from scipy.stats import spearmanr

    out = {}
    labels = np.asarray(labels).astype(int)
    if us.size == 0 or histo.size == 0 or us.shape[0] != histo.shape[0]:
        return out

    # Width mismatch means there is no shared space to measure in — e.g. pnf,
    # whose image_feats_needle is a 256-d mean over ProstNFound image_feats while
    # histo embeddings are 768-d. That is not an error: pnf performs no alignment,
    # so there is nothing to report. Flag it and return rather than crashing.
    if us.shape[1] != histo.shape[1]:
        out[f"{prefix}histo_available"] = 0.0
        out[f"{prefix}dim_mismatch"] = 1.0
        return out

    U = _row_normalize(us)
    Hh = _row_normalize(histo)
    # drop all-zero histo rows (us_only runs zero the tensor downstream of the model)
    keep = np.linalg.norm(np.asarray(histo, dtype=np.float64), axis=1) > 1e-6
    if keep.sum() < 10:
        out[f"{prefix}histo_available"] = 0.0
        return out
    out[f"{prefix}histo_available"] = 1.0

    grades = sorted(set(labels[keep].tolist()))
    if len(grades) < 2:
        return out

    us_cent = {g: _row_normalize(U[labels == g].mean(0, keepdims=True))[0] for g in grades}
    hi_cent = {
        g: _row_normalize(Hh[keep & (labels == g)].mean(0, keepdims=True))[0]
        for g in grades
        if (keep & (labels == g)).sum() > 0
    }
    grades = [g for g in grades if g in hi_cent]

    # ---- per-core: own-grade vs other-grade histo centroid ----
    same, other, per_grade_same = [], [], defaultdict(list)
    for i in range(U.shape[0]):
        g = labels[i]
        if g not in hi_cent:
            continue
        cs = float(U[i] @ hi_cent[g])
        co = [float(U[i] @ hi_cent[h]) for h in grades if h != g]
        same.append(cs)
        per_grade_same[g].append(cs)
        if co:
            other.append(float(np.mean(co)))

    if same:
        out[f"{prefix}us_histo_cos_same"] = float(np.mean(same))
    if other:
        out[f"{prefix}us_histo_cos_other"] = float(np.mean(other))
        out[f"{prefix}us_histo_margin"] = float(np.mean(same) - np.mean(other))
    for g, v in per_grade_same.items():
        out[f"{prefix}us_histo_cos_GG{g}"] = float(np.mean(v))

    # ---- modality gap ----
    mu_u = _row_normalize(U.mean(0, keepdims=True))[0]
    mu_h = _row_normalize(Hh[keep].mean(0, keepdims=True))[0]
    out[f"{prefix}modality_gap"] = float(1.0 - mu_u @ mu_h)

    # ---- retrieval: does a US core retrieve its own grade's histo centroid? ----
    correct = 0
    total = 0
    gl = list(grades)
    Hc = np.stack([hi_cent[g] for g in gl])
    for i in range(U.shape[0]):
        if labels[i] not in hi_cent:
            continue
        total += 1
        correct += int(gl[int(np.argmax(U[i] @ Hc.T))] == labels[i])
    if total:
        out[f"{prefix}centroid_retrieval_acc"] = correct / total

    # ---- ordinality: distance should grow with |Δgrade| ----
    def _ordinality(dist_fn):
        d, gap = [], []
        for a in range(len(gl)):
            for b in range(a + 1, len(gl)):
                d.append(dist_fn(gl[a], gl[b]))
                gap.append(abs(gl[a] - gl[b]))
        if len(d) < 3 or np.std(d) == 0:
            return float("nan")
        return float(spearmanr(gap, d).statistic)

    out[f"{prefix}ordinality_us"] = _ordinality(
        lambda a, b: 1.0 - float(us_cent[a] @ us_cent[b])
    )
    out[f"{prefix}ordinality_histo"] = _ordinality(
        lambda a, b: 1.0 - float(hi_cent[a] @ hi_cent[b])
    )
    # cross-modal version uses all (i, j) pairs including i == j
    d, gap = [], []
    for a in gl:
        for b in gl:
            d.append(1.0 - float(us_cent[a] @ hi_cent[b]))
            gap.append(abs(a - b))
    if len(d) >= 3 and np.std(d) > 0:
        out[f"{prefix}ordinality_cross"] = float(spearmanr(gap, d).statistic)

    # ---- scale check: how separated are the US centroids at all? ----
    seps = [
        1.0 - float(us_cent[gl[a]] @ us_cent[gl[b]])
        for a in range(len(gl))
        for b in range(a + 1, len(gl))
    ]
    if seps:
        out[f"{prefix}us_centroid_separation"] = float(np.mean(seps))
    return out


# --------------------------------------------------------------------------- #
# Step-level recorder                                                           #
# --------------------------------------------------------------------------- #


def setup_wandb_metrics(wandb_run=None):
    """
    Call once, immediately after wandb.init().

    Without this, wandb keys everything to an internal counter that increments on
    every log() call. Since the loop logs step metrics every iteration, diagnostics
    every `diag_every` iterations, and eval metrics every epoch, that counter runs
    ahead of the true training step and any log() with an explicit lower step is
    discarded ("Steps must be monotonically increasing"). Declaring global_step as
    the x-axis makes logging order irrelevant.
    """
    import wandb

    run = wandb_run or wandb.run
    if run is None:
        return

    # wandb only accepts SUFFIX globs ("prefix*"). "*_lr" raises. Name the lr keys
    # explicitly and glob the numbered fallback group_{i}_lr.
    patterns = (
        "diag/*",
        "linear_probe/*",
        "head/*",
        "val/*",
        "align/*",
        "train_loss",
        "encoder_lr",
        "main_lr",
        "cnn_lr",
        "group_*",
    )
    try:
        wandb.define_metric("global_step")
    except Exception as e:  # noqa: BLE001
        logging.warning(f"[wandb] define_metric('global_step') failed: {e}")
        return
    patterns = patterns + ("val/*", "test/*", "train/*")
    for pattern in patterns:
        # Never let a metric-declaration quirk kill a training run.
        try:
            wandb.define_metric(pattern, step_metric="global_step")
        except Exception as e:  # noqa: BLE001
            logging.warning(f"[wandb] define_metric({pattern!r}) skipped: {e}")


class DiagRecorder:
    """
    Accumulates per-(step, grade) diagnostics and writes one tidy CSV per run.

    One CSV per method -> one pandas concat -> every dynamics figure, all methods
    on a shared step axis. Log at a fixed GLOBAL step interval, not `iter % 200`,
    or runs with different epoch lengths end up on incompatible axes (which is
    why your entropy figure runs to 450 and the others to 275).
    """

    def __init__(self, out_dir: str, run_name: str, every: int = 10):
        self.rows = []
        self.every = every
        self.out_dir = out_dir
        self.run_name = run_name
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f"diag_{run_name}.csv")

    def should_log(self, global_step: int) -> bool:
        return global_step % self.every == 0

    def add(self, global_step: int, epoch: int, grade: int, metrics: dict, n_cores: int):
        row = {
            "run": self.run_name,
            "global_step": global_step,
            "epoch": epoch,
            "grade": grade,
            "n_cores": n_cores,
        }
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = float(v.float().mean().item()) if v.numel() else float("nan")
            row[k] = v
        self.rows.append(row)

    def flush(self):
        if not self.rows:
            return
        import pandas as pd

        pd.DataFrame(self.rows).to_csv(self.path, index=False)


@torch.no_grad()
def collect_step_diagnostics(
    data: dict,
    recorder: DiagRecorder,
    global_step: int,
    epoch: int,
    patch_cancer_head=None,
    wandb_run=None,
):
    """
    Single entry point called from the training loop. Works for every method.

    Requires in `data`:
      attention, needle_valid_mask, patch_feats_needle, grade_group
    Optional: involvement
    """
    A = data["attention"].detach().float()
    mask = data["needle_valid_mask"].detach().bool()
    H = data["patch_feats_needle"].detach()
    grades = data["grade_group"].detach()

    ent = normalized_attention_entropy(A, mask)
    conc = attention_concentration(A, mask)
    het = patch_heterogeneity(H, mask)

    scores = None
    if patch_cancer_head is not None:
        scores = patch_cancer_head(H).squeeze(-1).sigmoid()

    wb = {}
    for grade in range(6):
        g = grades == grade
        if g.sum() == 0:
            continue
        m = {
            "attn_entropy_norm": ent[g],
            "attn_max": conc["max_attn"][g],
            "attn_top10pct_mass": conc["top10pct_mass"][g],
        }
        # heterogeneity is computed only over samples with >=3 valid tokens, so
        # it is indexed positionally against the filtered set; recompute per grade
        het_g = patch_heterogeneity(H[g], mask[g])
        for k, v in het_g.items():
            m[k] = v
        if scores is not None:
            ps = patch_score_stats(scores[g], mask[g])
            for k, v in ps.items():
                m[k] = v

        recorder.add(global_step, epoch, grade, m, n_cores=int(g.sum().item()))
        for k, v in m.items():
            if isinstance(v, torch.Tensor) and v.numel():
                wb[f"diag/GG{grade}/{k}"] = float(v.float().mean().item())

    # batch-level (all grades pooled) — this is the single-panel figure version
    wb["diag/all/attn_entropy_norm"] = float(ent.mean().item())
    for k, v in het.items():
        wb[f"diag/all/{k}"] = float(v.float().mean().item())
    wb["diag/all/attn_centered_sim"] = centered_attention_similarity(A, mask)
    wb["diag/global_step"] = global_step

    if wandb_run is not None:
        # Never pass step=global_step here. Other wandb.log() calls in the loop
        # auto-increment wandb's internal counter, so it drifts ahead of
        # global_step and every diagnostic row gets silently dropped. The
        # define_metric() calls in setup_wandb_metrics() make global_step the
        # x-axis instead, which is order-independent.
        wandb_run.log(wb)
    return wb


# --------------------------------------------------------------------------- #
# Evaluation metrics                                                            #
# --------------------------------------------------------------------------- #


def classification_metrics(y_true: np.ndarray, y_proba: np.ndarray, classes: np.ndarray,
                           prefix: str = "") -> dict:
    """
    Everything the paper needs from one probability matrix.

    y_proba : (n_samples, n_classes) aligned with `classes`
    """
    out = {}
    y_true = np.asarray(y_true).astype(int)
    classes = np.asarray(classes).astype(int)

    # --- per-class one-vs-rest AUC + macro (what your main table reports) ---
    per_class = {}
    for i, c in enumerate(classes):
        yb = (y_true == c).astype(int)
        if yb.sum() == 0 or yb.sum() == len(yb):
            per_class[int(c)] = float("nan")
            continue
        try:
            per_class[int(c)] = float(roc_auc_score(yb, y_proba[:, i]))
        except ValueError:
            per_class[int(c)] = float("nan")
    valid = [v for v in per_class.values() if not np.isnan(v)]
    out[f"{prefix}macro_auc"] = float(np.mean(valid)) if valid else float("nan")
    for c, v in per_class.items():
        out[f"{prefix}auc_GG{c}"] = v

    # --- binary csPCa (GG >= 2): the primary clinical metric ---
    cs_cols = [i for i, c in enumerate(classes) if c >= 2]
    if cs_cols:
        score_cs = y_proba[:, cs_cols].sum(axis=1)
        y_cs = (y_true >= 2).astype(int)
        if 0 < y_cs.sum() < len(y_cs):
            out[f"{prefix}auc_csPCa"] = float(roc_auc_score(y_cs, score_cs))
        else:
            out[f"{prefix}auc_csPCa"] = float("nan")

    # --- binary any-cancer (GG >= 1), for comparability with GUIDE-US ---
    ca_cols = [i for i, c in enumerate(classes) if c >= 1]
    if ca_cols:
        score_ca = y_proba[:, ca_cols].sum(axis=1)
        y_ca = (y_true >= 1).astype(int)
        if 0 < y_ca.sum() < len(y_ca):
            out[f"{prefix}auc_cancer"] = float(roc_auc_score(y_ca, score_ca))
        else:
            out[f"{prefix}auc_cancer"] = float("nan")

    # --- ordinal metrics: the target is ordered, so measure that ---
    y_hat = classes[np.argmax(y_proba, axis=1)]
    expected = (y_proba * classes[None, :]).sum(axis=1)

    out[f"{prefix}acc"] = float(accuracy_score(y_true, y_hat))
    out[f"{prefix}balanced_acc"] = float(balanced_accuracy_score(y_true, y_hat))
    out[f"{prefix}macro_f1"] = float(f1_score(y_true, y_hat, average="macro"))
    try:
        out[f"{prefix}qwk"] = float(
            cohen_kappa_score(y_true, y_hat, weights="quadratic")
        )
    except Exception:
        out[f"{prefix}qwk"] = float("nan")
    out[f"{prefix}mae_argmax"] = float(np.abs(y_true - y_hat).mean())
    out[f"{prefix}mae_expected"] = float(np.abs(y_true - expected).mean())
    out[f"{prefix}adjacent_acc"] = float((np.abs(y_true - y_hat) <= 1).mean())
    return out


# Keys big enough to blow up disk if written every epoch. ~5.7 MB per epoch at
# N=1000, D=768, versus 36 KB for everything else.
_HEAVY_KEYS = ("X_val", "meta_histo_embed", "histo_embed", "patch_feats")


def save_predictions(
    out_dir: str,
    run_name: str,
    fold,
    epoch: int,
    payload: dict,
    save_embeddings: bool = False,
    embed_dtype=np.float16,
):
    """
    Dump everything needed to recompute ANY metric offline, forever.

    Split into two files by weight:

      preds_*.npz   labels, probabilities, ids, involvement, entropy.
                    ~36 KB. Written EVERY epoch. Every scalar metric in the
                    paper -- csPCa AUC, QWK, per-fold paired tests -- comes
                    from this file alone.

      embeds_*.npz  US and histo embedding matrices. ~2.9 MB at fp16.
                    Written only when save_embeddings=True, because 7 configs
                    x 5 folds x 50 epochs of these is ~10 GB. Needed only to
                    recompute the alignment metrics or to make t-SNE/UMAP
                    panels, neither of which you want at every epoch.

    fp16 is used for embeddings: cosine similarities and centroid distances are
    unaffected at that precision, and it halves the footprint.
    """
    os.makedirs(out_dir, exist_ok=True)
    light = {k: np.asarray(v) for k, v in payload.items() if k not in _HEAVY_KEYS}
    path = os.path.join(out_dir, f"preds_{run_name}_fold{fold}_ep{epoch:03d}.npz")
    np.savez_compressed(path, **light)

    if save_embeddings:
        heavy = {
            k: np.asarray(v).astype(embed_dtype)
            for k, v in payload.items()
            if k in _HEAVY_KEYS
        }
        if heavy:
            epath = os.path.join(
                out_dir, f"embeds_{run_name}_fold{fold}_ep{epoch:03d}.npz"
            )
            np.savez_compressed(epath, **heavy)
    return path


def confusion_to_json(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(6)))
    with open(path, "w") as f:
        json.dump({"labels": list(range(6)), "matrix": cm.tolist()}, f, indent=2)
    return cm