"""
plot_embeddings.py — 2-D projection of the micro-US embedding space, coloured by
grade group.

    python plot_embeddings.py --pred_dir runs/predictions --runs ours guideus --fold 0

Micro-US embeddings only. Histopathology is not plotted: a joint projection shows
how far apart the two modalities remain, which is a property of the modality gap
rather than of the arrangement of the micro-US space, and it is the micro-US
arrangement the grade claim is about.

USE THE BEST-EPOCH CHECKPOINT. In-training dumps are written on a fixed cadence
(epochs 0, 10, 20, ...) and the latest one is simply the last epoch reached, not
the epoch with the best tracked metric. Run extract_embeddings.py against
best.pth, which writes ep999, and this script prefers ep999 when present.

Falls back to PCA if umap-learn is missing, and labels the axes accordingly --
never silently, since the two are not interchangeable.
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

GRADE_COLORS = ["#4C72B0", "#55A868", "#C6DB4E", "#F5C242", "#E8743B", "#C44E52"]


def _row_normalize(X):
    X = np.asarray(X, dtype=np.float64)
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-8, None)


def load_run(pred_dir, run, fold, epoch=None):
    files = sorted(glob.glob(os.path.join(pred_dir, f"embeds_{run}_fold{fold}_ep*.npz")))
    if not files:
        raise FileNotFoundError(f"no embeds for run={run} fold={fold} under {pred_dir}")

    if epoch is not None:
        path = next(f for f in files if f.endswith(f"ep{epoch:03d}.npz"))
    else:
        # ep999 == extracted from best.pth; prefer it over any cadence dump
        best = [f for f in files if f.endswith("ep999.npz")]
        path = best[0] if best else files[-1]
        if not best:
            print(f"  [{run}] WARNING using {os.path.basename(path)} — a cadence "
                  f"dump, not the best epoch. Run extract_embeddings.py against "
                  f"best.pth for the model you actually report.")

    us = _row_normalize(np.load(path, allow_pickle=True)["X_val"])
    labels = np.load(path.replace("embeds_", "preds_"),
                     allow_pickle=True)["y_true"].astype(int)
    return us, labels, os.path.basename(path)


def embed_2d(X, seed=0):
    try:
        import umap

        return umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine",
                         random_state=seed).fit_transform(X), "UMAP"
    except ImportError:
        from sklearn.decomposition import PCA

        print("  umap-learn not installed — using PCA (axes labelled accordingly)")
        return PCA(n_components=2, random_state=seed).fit_transform(X), "PCA"


def plot_runs(runs_data, out_path, seed=0, title_map=None):
    n = len(runs_data)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.9), squeeze=False)
    axes = axes[0]

    for ax, (run, us, labels) in zip(axes, runs_data):
        Z, method = embed_2d(us, seed=seed)
        # draw rarer grades last so they are not buried under GG0/GG1
        for g in sorted(range(6), key=lambda g: -(labels == g).sum()):
            sel = labels == g
            if sel.sum() == 0:
                continue
            ax.scatter(Z[sel, 0], Z[sel, 1], s=10, alpha=0.6, linewidths=0,
                       c=GRADE_COLORS[g], label=f"GG{g}")
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_xlabel(f"{method}-1", fontsize=9)
        ax.set_ylabel(f"{method}-2", fontsize=9)
        ax.set_title((title_map or {}).get(run, run), fontsize=10)

    axes[0].legend(fontsize=7, frameon=False, markerscale=1.8, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default="runs/predictions")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epoch", type=int, default=None,
                    help="default: ep999 (best.pth) if present, else latest dump")
    ap.add_argument("--out", default="figures/embedding_space.pdf")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--separate", action="store_true",
                    help="one file per run instead of a single multi-panel figure")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    loaded = []
    for run in args.runs:
        print(f"{run}:")
        try:
            us, labels, src = load_run(args.pred_dir, run, args.fold, args.epoch)
        except (FileNotFoundError, StopIteration) as e:
            print(f"  skipped: {e}")
            continue
        print(f"  {src}  n={len(us)}  grades={np.bincount(labels, minlength=6).tolist()}")
        loaded.append((run, us, labels))

    if not loaded:
        raise SystemExit("nothing to plot")

    if args.separate:
        base, ext = os.path.splitext(args.out)
        for item in loaded:
            plot_runs([item], f"{base}_{item[0]}{ext}", seed=args.seed)
    else:
        plot_runs(loaded, args.out, seed=args.seed)


if __name__ == "__main__":
    main()