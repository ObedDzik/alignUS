"""
make_figures.py — builds the paper figures from diag_*.csv and preds_*.npz.

Usage:
    python make_figures.py --diag_dir runs/diagnostics --pred_dir runs/predictions

Produces:
    fig_heterogeneity.pdf   all methods, one averaged panel   [headline mechanism figure]
    fig_entropy.pdf         all methods, one averaged panel   [necessary-not-sufficient]
    fig_per_grade.pdf       6-panel versions                  [supplementary]
    results_table.csv       per-method metrics, mean +/- std over folds
    paired_stats.csv        per-fold paired deltas + Wilcoxon vs. the chosen baseline
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Order and labels for the figures. Keys must match cfg.wandb.run_name prefixes.
METHOD_ORDER = [
    ("guideus", "GUIDE-US"),
    ("supcon_only", "SupCon-Only"),
    ("acmil", "ACMIL"),
    ("aem", "AEM"),
    ("alignus_no_lsg", "AlignUS w/o $\\mathcal{L}_{sg}$"),
    ("alignus", "AlignUS (Ours)"),
]

METRICS = {
    "cosine_dispersion": "Patch cosine dispersion",
    "dispersion_index": "Dispersion index",
    "effective_rank": "Effective rank",
    "attn_entropy_norm": "Normalized attention entropy ($H/\\ln N$)",
}


def _match(run_name: str):
    rn = run_name.lower()
    for key, label in METHOD_ORDER:
        if key in rn:
            return label
    return None


def load_diagnostics(diag_dir: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(diag_dir, "diag_*.csv"))):
        df = pd.read_csv(path)
        label = _match(str(df["run"].iloc[0]))
        if label is None:
            print(f"  skipping unmatched run: {df['run'].iloc[0]}")
            continue
        df["method"] = label
        frames.append(df)
    if not frames:
        raise SystemExit(f"no matching diag_*.csv under {diag_dir}")
    return pd.concat(frames, ignore_index=True)


def smooth(y: np.ndarray, w: int = 9) -> np.ndarray:
    if len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode="same")


def plot_single_panel(df: pd.DataFrame, metric: str, out_path: str):
    """One averaged panel, all methods. This is the version that goes in the paper."""
    if metric not in df.columns:
        print(f"  {metric} not logged, skipping")
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    styles = ["-", "--", "-.", ":", "-", "--"]
    for i, (_, label) in enumerate(METHOD_ORDER):
        sub = df[df["method"] == label]
        if sub.empty:
            continue
        # average over grades and folds at each global step
        g = sub.groupby("global_step")[metric].mean().reset_index()
        ax.plot(g["global_step"], smooth(g[metric].values),
                styles[i % len(styles)], label=label, lw=1.6)
    ax.set_xlabel("Training step")
    ax.set_ylabel(METRICS.get(metric, metric))
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_per_grade(df: pd.DataFrame, metric: str, out_path: str):
    """6-panel version. Supplementary only — it costs a page and adds little."""
    if metric not in df.columns:
        return
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for grade, ax in zip(range(6), axes.ravel()):
        for _, label in METHOD_ORDER:
            sub = df[(df["method"] == label) & (df["grade"] == grade)]
            if sub.empty:
                continue
            g = sub.groupby("global_step")[metric].mean().reset_index()
            ax.plot(g["global_step"], smooth(g[metric].values), label=label, lw=1.3)
        ax.set_title(f"GG{grade}", fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0, 0].legend(fontsize=6, frameon=False)
    for ax in axes[1]:
        ax.set_xlabel("Training step")
    for ax in axes[:, 0]:
        ax.set_ylabel(METRICS.get(metric, metric), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------- #
# Tables: recomputed offline from the saved probability matrices                #
# --------------------------------------------------------------------------- #


def load_predictions(pred_dir: str, peak_metric: str = "macro_auc") -> pd.DataFrame:
    """
    One row per (method, fold) at the peak-validation epoch, matching how the
    main table is reported.
    """
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from diagnostics import classification_metrics

    rows = []
    for path in sorted(glob.glob(os.path.join(pred_dir, "preds_*.npz"))):
        base = os.path.basename(path)
        label = _match(base)
        if label is None:
            continue
        fold = int(base.split("fold")[1].split("_")[0])
        epoch = int(base.split("_ep")[1].split(".npz")[0])
        d = np.load(path, allow_pickle=True)
        m = classification_metrics(d["y_true"], d["y_proba"], d["classes"])
        m.update({"method": label, "fold": fold, "epoch": epoch, "path": path})
        rows.append(m)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no preds_*.npz under {pred_dir}")
    # peak-epoch selection per (method, fold)
    idx = df.groupby(["method", "fold"])[peak_metric].idxmax()
    return df.loc[idx].reset_index(drop=True)


def build_results_table(peak: pd.DataFrame, out_path: str):
    cols = [c for c in peak.columns if c.startswith("auc_") or c in
            ("macro_auc", "qwk", "mae_expected", "adjacent_acc", "balanced_acc")]
    agg = peak.groupby("method")[cols].agg(["mean", "std"])
    agg.to_csv(out_path)
    print(f"  wrote {out_path}")

    # LaTeX-ready mean+/-std for the main table columns
    main = ["auc_GG0", "auc_GG1", "auc_GG2", "auc_GG3", "auc_GG4", "auc_GG5",
            "macro_auc", "auc_csPCa"]
    print("\n  main table (x100):")
    for method in peak["method"].unique():
        sub = peak[peak["method"] == method]
        cells = []
        for c in main:
            if c not in sub:
                cells.append("--")
                continue
            cells.append(f"{100*sub[c].mean():.1f}$\\pm${100*sub[c].std():.1f}")
        print(f"    {method:28s} & " + " & ".join(cells) + " \\\\")


def paired_stats(peak: pd.DataFrame, baseline: str, out_path: str):
    """
    Per-fold paired deltas + Wilcoxon. 67.1+/-4.1 vs 63.6+/-3.8 does not survive an
    unpaired reading; the paired test is what makes the comparison defensible.
    """
    rows = []
    base = peak[peak["method"] == baseline].set_index("fold")
    for method in peak["method"].unique():
        if method == baseline:
            continue
        sub = peak[peak["method"] == method].set_index("fold")
        folds = sorted(set(base.index) & set(sub.index))
        for metric in ["macro_auc", "auc_csPCa", "auc_GG2", "auc_GG3", "qwk"]:
            if metric not in sub.columns:
                continue
            a = sub.loc[folds, metric].values
            b = base.loc[folds, metric].values
            d = a - b
            try:
                p = wilcoxon(a, b).pvalue if len(folds) >= 5 else float("nan")
            except ValueError:
                p = float("nan")
            rows.append({
                "method": method, "baseline": baseline, "metric": metric,
                "n_folds": len(folds),
                "mean_delta": float(np.mean(d)), "std_delta": float(np.std(d, ddof=1)),
                "min_delta": float(np.min(d)), "max_delta": float(np.max(d)),
                "wins": int((d > 0).sum()), "wilcoxon_p": float(p),
            })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag_dir", default="runs/diagnostics")
    ap.add_argument("--pred_dir", default="runs/predictions")
    ap.add_argument("--out_dir", default="figures")
    ap.add_argument("--baseline", default="GUIDE-US")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("diagnostics:")
    try:
        diag = load_diagnostics(args.diag_dir)
        plot_single_panel(diag, "cosine_dispersion",
                          os.path.join(args.out_dir, "fig_heterogeneity.pdf"))
        plot_single_panel(diag, "attn_entropy_norm",
                          os.path.join(args.out_dir, "fig_entropy.pdf"))
        for m in ("cosine_dispersion", "attn_entropy_norm", "effective_rank"):
            plot_per_grade(diag, m, os.path.join(args.out_dir, f"supp_{m}.pdf"))
    except SystemExit as e:
        print(f"  {e}")

    print("predictions:")
    peak = load_predictions(args.pred_dir)
    build_results_table(peak, os.path.join(args.out_dir, "results_table.csv"))
    paired_stats(peak, args.baseline, os.path.join(args.out_dir, "paired_stats.csv"))


if __name__ == "__main__":
    main()
