"""
plot_diagnostics.py — mechanism figure and table from diag_*.csv, single fold.

    python plot_diagnostics.py --out figures/ \
      --csv "ABMIL + ACMIL=$CHECKPOINT/alignUS/acmil_512_0/<job_id>/diagnostics/diag_acmil_512_0.csv" \
      --csv "ABMIL + AEM=$CHECKPOINT/alignUS/aem_512_0/<job_id>/diagnostics/diag_aem_512_0.csv" \
      --csv "GUIDE-US=$CHECKPOINT/alignUS/guideus_0/<job_id>/diagnostics/diag_guideus_0.csv" \
      --csv "Ours=$CHECKPOINT/alignUS/alignUS_512_0/<job_id>/diagnostics/diag_alignUS_512_0.csv"

Order on the command line is the order in the legend and the table.

Outputs:
    fig_mechanism.pdf    two panels over training steps
    table_mechanism.tex  paste-ready, final-window summary
    mechanism.csv        the same numbers

WHAT THE PANELS EVIDENCE
  (a) patch-feature cosine dispersion, computed on the shared patch features H
      that L_sg's gradient reaches. Scale-free, and available for every
      configuration including those with no patch cancer head.
  (b) normalized attention entropy H/ln(N_b); lower means more concentrated.

SINGLE FOLD: the +/- in the table is variation across steps inside the final
window of ONE run, not across folds. It describes how settled the trace is, not
how reproducible the value is. The caption written by this script says so.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRIC_LABELS = {
    "cosine_dispersion": "Patch-feature cosine dispersion",
    "attn_entropy_norm": "Attention entropy ($H/\\ln N_b$)",
    "score_std_norm":    "Within-core patch-score std",
    "dispersion_index":  "Dispersion index",
    "effective_rank":    "Effective rank",
    "attn_max":          "Max attention weight",
}
PANELS = ["cosine_dispersion", "attn_entropy_norm"]
TABLE_METRICS = ["cosine_dispersion", "effective_rank", "attn_entropy_norm", "score_std_norm"]


def load(pairs):
    frames = []
    for label, path in pairs:
        if not os.path.exists(path):
            print(f"  MISSING: {label} -> {path}")
            continue
        df = pd.read_csv(path)
        if df.empty:
            print(f"  EMPTY: {label}")
            continue
        df["method"] = label
        frames.append(df)
        cols = [c for c in TABLE_METRICS if c in df.columns and not df[c].isna().all()]
        print(f"  {label:28s} {len(df):6d} rows, steps "
              f"{df.global_step.min()}-{df.global_step.max()}, metrics: {cols}")
    if not frames:
        raise SystemExit("nothing loaded")
    return pd.concat(frames, ignore_index=True)


def smooth(y, w=9):
    return y if len(y) < w else np.convolve(y, np.ones(w) / w, mode="same")


def make_figure(df, order, out_path):
    metrics = [m for m in PANELS if m in df.columns and not df[m].isna().all()]
    if not metrics:
        print("  no panel metrics present")
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 3.4),
                             squeeze=False)
    styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    for ax, metric in zip(axes[0], metrics):
        for i, label in enumerate(order):
            sub = df[df["method"] == label]
            if sub.empty or sub[metric].isna().all():
                continue
            g = sub.groupby("global_step")[metric].mean().reset_index()
            ax.plot(g["global_step"], smooth(g[metric].values),
                    linestyle=styles[i % len(styles)], lw=1.6, label=label)
        ax.set_xlabel("Training step", fontsize=9)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def make_table(df, order, out_tex, out_csv, final_frac=0.25):
    metrics = [m for m in TABLE_METRICS
               if m in df.columns and not df[m].isna().all()]
    if not metrics:
        print("  no table metrics present")
        return None

    rows = []
    for label in order:
        sub = df[df["method"] == label]
        if sub.empty:
            continue
        cutoff = sub["global_step"].quantile(1 - final_frac)
        tail = sub[sub["global_step"] >= cutoff]
        r = {"method": label, "n_steps": tail["global_step"].nunique()}
        for m in metrics:
            per_step = tail.groupby("global_step")[m].mean()
            r[f"{m}_mean"] = per_step.mean()
            r[f"{m}_std"] = per_step.std()
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)

    header = " & ".join(
        f"\\textbf{{{METRIC_LABELS.get(m, m).split(' (')[0]}}}" for m in metrics)
    lines = [
        r"\begin{table}[t]",
        r"\caption{Embedding diagnostics on fold 0, averaged over the final "
        rf"{int(final_frac * 100)}\% of training steps; $\pm$ denotes variation "
        r"across steps within that window. Cosine dispersion is computed on the "
        r"patch features and attention entropy is normalized by each core's "
        r"valid-token count.}",
        r"\label{tab:mechanism}", r"\centering", r"\small",
        r"\begin{tabular}{l" + "c" * len(metrics) + "}", r"\toprule",
        r"\textbf{Configuration} & " + header + r"\\", r"\midrule",
    ]
    for _, r in out.iterrows():
        cells = []
        for m in metrics:
            mu, sd = r[f"{m}_mean"], r[f"{m}_std"]
            cells.append("--" if np.isnan(mu) else
                         f"{mu:.3f}$\\pm${0.0 if np.isnan(sd) else sd:.3f}")
        lines.append(f"{r['method']} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(out_tex, "w").write("\n".join(lines) + "\n")
    print(f"  wrote {out_tex} and {out_csv}\n")

    disp = out.set_index("method")
    for m in metrics:
        print(f"  {METRIC_LABELS.get(m, m)}:")
        for label in out["method"]:
            v = disp.loc[label, f"{m}_mean"]
            print(f"    {label:30s} {'--' if np.isnan(v) else f'{v:.4f}'}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True, metavar="LABEL=PATH",
                    help="repeat once per configuration; order is preserved")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--final_frac", type=float, default=0.25)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pairs = []
    for item in args.csv:
        if "=" not in item:
            raise SystemExit(f"expected LABEL=PATH, got: {item}")
        label, path = item.split("=", 1)
        pairs.append((label.strip(), path.strip()))

    df = load(pairs)
    order = [lab for lab, _ in pairs if lab in set(df["method"])]

    make_figure(df, order, os.path.join(args.out, "fig_mechanism.pdf"))
    out = make_table(df, order, os.path.join(args.out, "table_mechanism.tex"),
                     os.path.join(args.out, "mechanism.csv"),
                     final_frac=args.final_frac)

    if out is not None and "cosine_dispersion_mean" in out.columns and len(order) >= 2:
        ours = out[out["method"] == order[-1]]["cosine_dispersion_mean"].iloc[0]
        rest = out[out["method"] != order[-1]]["cosine_dispersion_mean"]
        print(f"\n  {order[-1]} cosine dispersion: {ours:.4f}")
        print(f"  next highest: {rest.max():.4f}")
        if not (ours > rest.max()):
            print("  NOT the highest. The abstract's 'induces patch-level "
                  "heterogeneity' claim is not evidenced by this fold; soften it.")


if __name__ == "__main__":
    main()