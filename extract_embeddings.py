"""
extract_embeddings.py — rebuild a trained model from best.pth and dump the
embeddings and predictions that plot_embeddings.py / make_figures.py expect.

    python extract_embeddings.py \\
        --config cfg/train/alignus.yaml \\
        --checkpoint runs/alignus_fold0/best.pth \\
        --train_module train_patched \\
        --run_name ours --fold 0 --with_train \\
        --plot figures/embedding_ours.pdf

Extracts from the checkpoint AND writes the figure in one call. --plot is
optional; without it only the .npz files are written and you can plot later with
plot_embeddings.py (useful for putting several runs in one multi-panel figure).

Writes, for each requested split:
    embeds_{run}_fold{f}_ep999.npz   X_val, meta_histo_embed        (fp16)
    preds_{run}_fold{f}_ep999.npz    y_true, y_proba, classes, meta

Epoch 999 marks "extracted from checkpoint" so these never collide with the
in-training dumps.

WHY --with_train MATTERS
    Your reported AUCs come from a logistic-regression probe fit on TRAIN
    embeddings and evaluated on VAL. Without --with_train this script can only
    dump val embeddings, which is enough for the UMAP and centroid figures but
    NOT enough to reproduce the numbers in your tables. Pass --with_train if you
    want metrics as well; it costs one extra pass over the training split.

WHAT THIS CANNOT RECOVER
    The per-step training dynamics in diag_*.csv. Those exist only if the run
    logged them. A checkpoint is a snapshot, not a history.
"""

import argparse
import importlib
import os

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from diagnostics import alignment_metrics, classification_metrics, save_predictions


def _to_list(v):
    return v.detach().cpu().numpy().tolist() if isinstance(v, torch.Tensor) else list(v)


# Set in the training sbatch, and read at import time by the backbone/dataset
# modules. Missing them produces a confusing failure deep inside a loader, so
# check up front and name what is absent.
REQUIRED_ENV = (
    "MEDSAM_CHECKPOINT_DIR",
    "MEDSAM_CHECKPOINT",
    "NCT_RAW_DATA_DIR",
    "NCT_METADATA_PATH",
    "DINOV3_LIBRARY_PATH",
    "EXACTVU_PCA_DATA_ROOT",
    "DINOV3_CHECKPOINTS_PATH",
)


def check_env(strict: bool = True):
    missing = [v for v in REQUIRED_ENV if not os.environ.get(v)]
    if not missing:
        return
    msg = (
        "missing environment variables: " + ", ".join(missing) + "\n"
        "These are exported in your training sbatch and are read at import time "
        "by the backbone and dataset modules. Export them before running this "
        "script, or use extract_embeddings.sh which sets them for you."
    )
    if strict:
        raise SystemExit(f"ERROR: {msg}")
    print(f"WARNING: {msg}")


def build_backbone(cfg, mod):
    """
    Backbone construction, in priority order:

      1. a factory the training module exposes (create_model / build_model /
         build_backbone) -- preferred, because it cannot drift from training;
      2. the cfg.model_type branch below, which mirrors the one in the training
         script for modules that build the backbone inline in main().

    Path 2 is a copy, and a copy can go stale. If you change the construction in
    the training script, change it here too -- or better, factor it into a
    function the training module exposes so path 1 picks it up and this branch
    stops being reachable.
    """
    # for fname in ("create_model", "build_model", "build_backbone"):
    #     fn = getattr(mod, fname, None)
    #     if fn is None:
    #         continue
    #     try:
    #         if fname == "create_model":
    #             return fn(cfg.model, **cfg.get("model_kw", {}))
    #         return fn(cfg)
    #     except (AttributeError, TypeError) as e:
    #         print(f"  {fname}() present but not usable here ({e}); "
    #               f"falling back to cfg.model_type")
    #         break

    mt = cfg.get("model_type")
    if mt is None:
        raise RuntimeError(
            "no backbone factory in the training module and cfg.model_type is "
            "unset — cannot construct the backbone."
        )

    if mt == "dino":
        # imported from the training module when it re-exports it, so the
        # DINOV3_LIBRARY_PATH shim the training script relies on is honoured
        dinov3_vitl16 = getattr(mod, "dinov3_vitl16", None)
        if dinov3_vitl16 is None:
            from medAI.modeling.dinov3 import dinov3_vitl16  # noqa: F401
        model = dinov3_vitl16()
    elif mt == "medsam":
        from medAI.modeling.sam import medsam_adapter

        model = medsam_adapter()
    elif mt == "microsegnet":
        from src.microseg_model import WrapperMicroSegNet

        model = WrapperMicroSegNet(
            checkpoint_path=cfg.microsegnet_checkpoint,
            img_size=224, n_skip=3, vit_name="R50-ViT-B_16", vit_patches_size=16,
            freeze=False,
        )
    else:
        raise RuntimeError(f"unknown cfg.model_type={mt!r}")

    print(f"  backbone built from cfg.model_type={mt!r}")
    return model


def build_model(cfg, train_module):
    """
    Wraps the backbone in the training module's own ProstNFoundMeta. That part is
    never reimplemented: a subtly different wrapper would load the checkpoint with
    strict=False and leave weights at random init, which looks like a bad result
    rather than a bug.
    """
    mod = importlib.import_module(train_module)

    backbone = build_backbone(cfg, mod)

    Meta = getattr(mod, "ProstNFoundMeta", None)
    if Meta is None:
        raise RuntimeError(f"{train_module} has no ProstNFoundMeta")

    # the two training scripts order the constructor differently
    try:
        model = Meta(backbone, cfg=cfg, **cfg.get("metamodel", {}))
    except TypeError:
        model = Meta(cfg, backbone, **cfg.get("metamodel", {}))
    return model, mod


def load_checkpoint(model, path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    msg = model.load_state_dict(state, strict=False)

    missing = [k for k in msg.missing_keys if "num_batches_tracked" not in k]
    if missing:
        print(f"  WARNING {len(missing)} missing keys — these stayed at random "
              f"init. First few: {missing[:5]}")
    if msg.unexpected_keys:
        print(f"  {len(msg.unexpected_keys)} unexpected keys (usually fine): "
              f"{msg.unexpected_keys[:3]}")
    if missing:
        print("  If anything above is an encoder or abmil weight, the extraction "
              "is NOT reproducing the trained model. Check that --config matches "
              "the config the checkpoint was trained with.")
    return model


@torch.no_grad()
def extract(model, loader, device, use_amp=False, desc="val"):
    model.eval()
    feats, histo_raw = [], []
    meta = {k: [] for k in ("grade_group", "involvement", "core_id",
                            "patient_id", "center", "bucket_label")}

    for data in tqdm(loader, desc=f"extract:{desc}"):
        # BEFORE the forward pass — us_only mode zeroes positive_hist in-place
        h = data.get("positive_hist")
        histo_raw.append(h.detach().float().cpu().numpy() if h is not None else None)

        with torch.cuda.amp.autocast(enabled=use_amp):
            data = model(data)

        feats.append(data["image_feats_needle"].detach().float().cpu().numpy())
        for k in meta:
            if k in data and data[k] is not None:
                meta[k] += _to_list(data[k])

    X = np.concatenate(feats, axis=0)
    H = (np.concatenate([h for h in histo_raw if h is not None], axis=0)
         if any(h is not None for h in histo_raw) else np.empty((0, 1)))
    meta = {k: np.array(v) for k, v in meta.items() if len(v)}
    return X, H, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train_module", required=True,
                    help="import path of the training module, e.g. "
                         "train_patched")
    ap.add_argument("--run_name", required=True,
                    help="must match the substring plot_embeddings/make_figures "
                         "look for, e.g. 'ours', 'guideus', 'beta0'")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", default="runs/predictions")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--with_train", action="store_true",
                    help="also extract train embeddings and refit the linear "
                         "probe, so AUC/QWK/adjacent-acc are recomputed too")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--plot", default=None, metavar="PATH",
                    help="also write the micro-US embedding figure straight to "
                         "this path, e.g. figures/embedding_ours.pdf")
    ap.add_argument("--seed", type=int, default=0, help="projection seed")
    ap.add_argument("--skip_env_check", action="store_true",
                    help="warn instead of exiting when env vars are missing")
    args = ap.parse_args()

    check_env(strict=not args.skip_env_check)

    cfg = OmegaConf.load(args.config)
    cfg.device = args.device
    os.makedirs(args.out, exist_ok=True)

    print("building model")
    model, mod = build_model(cfg, args.train_module)
    print(f"loading {args.checkpoint}")
    model = load_checkpoint(model, args.checkpoint)
    model.to(args.device)

    # dataloaders exactly as training built them
    if hasattr(mod, "get_dataloaders"):
        get_dataloaders = mod.get_dataloaders
    elif cfg.get("experiment_type") == "pnf":
        from baseline.guideus.src.nct_optimum_loader_pnf import get_dataloaders
    else:
        from src.nct_optimum_loader import get_dataloaders

    try:
        loaders = get_dataloaders(cfg.data, mode="train")
    except TypeError:
        loaders = get_dataloaders(cfg.data)

    use_amp = cfg.get("use_amp", False)
    X_val, H_val, meta_val = extract(model, loaders[args.split], args.device,
                                     use_amp, desc=args.split)
    print(f"  {args.split}: X={X_val.shape} histo={H_val.shape}")

    payload = {"X_val": X_val}
    if H_val.shape[0] == X_val.shape[0]:
        payload["meta_histo_embed"] = H_val
    for k, v in meta_val.items():
        payload[f"meta_{k}"] = v

    # ---- optional: refit the probe so the metrics come back too ----
    if args.with_train:
        from sklearn.linear_model import LogisticRegression

        X_tr, _, meta_tr = extract(model, loaders["train"], args.device,
                                   use_amp, desc="train")
        y_tr = meta_tr.get("bucket_label", meta_tr["grade_group"])
        y_va = meta_val.get("bucket_label", meta_val["grade_group"])

        clf = LogisticRegression(penalty="l2", C=1.0, solver="saga",
                                 max_iter=500, class_weight="balanced")
        clf.fit(X_tr, y_tr)
        y_proba = clf.predict_proba(X_va := X_val)
        classes = clf.classes_.astype(int)

        payload.update({"y_true": y_va, "y_proba": y_proba, "classes": classes})
        m = classification_metrics(y_va, y_proba, classes, prefix="")
        print("\n  recomputed metrics:")
        for k in ("macro_auc", "auc_csPCa", "auc_cancer", "qwk",
                  "adjacent_acc", "balanced_acc"):
            if k in m:
                print(f"    {k:16s} {m[k]:.4f}")
        for g in range(6):
            if f"auc_GG{g}" in m:
                print(f"    auc_GG{g}        {m[f'auc_GG{g}']:.4f}")
    else:
        # plot_embeddings needs y_true from the preds file even without a probe
        payload["y_true"] = meta_val.get("bucket_label", meta_val["grade_group"])
        payload["y_proba"] = np.zeros((X_val.shape[0], 6), dtype=np.float32)
        payload["classes"] = np.arange(6)
        print("\n  no --with_train: embeddings dumped, metrics NOT recomputed "
              "(y_proba is a placeholder). Figures will work; tables will not.")

    if H_val.shape[0] == X_val.shape[0] and "grade_group" in meta_val:
        a = alignment_metrics(X_val, H_val, meta_val["grade_group"], prefix="")
        print("\n  alignment:")
        for k in ("us_histo_margin", "modality_gap", "centroid_retrieval_acc",
                  "ordinality_us", "ordinality_cross"):
            if k in a:
                print(f"    {k:24s} {a[k]:+.4f}")

    path = save_predictions(args.out, args.run_name, args.fold, 999, payload,
                            save_embeddings=True)
    print(f"\nwrote {path} and the matching embeds_*.npz")

    if args.plot:
        from plot_embeddings import plot_runs

        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        labels = payload["y_true"].astype(int)
        us = X_val / np.clip(np.linalg.norm(X_val, axis=1, keepdims=True), 1e-8, None)
        plot_runs([(args.run_name, us, labels)], args.plot, seed=args.seed)
    else:
        print(f"next: python plot_embeddings.py --pred_dir {args.out} "
              f"--runs {args.run_name} --fold {args.fold}")


if __name__ == "__main__":
    main()