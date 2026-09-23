"""Core- and patient-level csPCa AUC/sens@60spe/bal_acc for a trained
guideus baseline checkpoint, in the same table format used throughout the
ProstCam comparison (2026-09-21, obed's ask) -- mirrors
`projects/prostnfound/eval_paper_metrics.py` almost line-for-line, adapted
for guideus's own `ProstNFoundMeta` (`guideus_pnf_train.py`, a different
class from prostnfound's own `ProstNFoundMeta`) and its own dataloader
(`src/nct_optimum_loader_pnf.py::get_dataloaders`).

Why this can't reuse the training loop's own `val/auc`/linear-probe metrics:
`ProstNFoundTransform`'s `out["label"]` is thresholded at
`grade_group_for_positive_label` (default 1, i.e. ANY cancer/PCa, not
clinically-significant/csPCa), and the transform's own output dict drops
both `grade_group` (raw int) and `patient_id` entirely (confirmed by
reading `medAI/transforms/prostnfound_transform.py::ProstNFoundTransform.__call__`
-- only `label`, `involvement`, `core_id`, `center` survive from the
source item). So this script independently re-derives the TRUE
grade_group per core (same `0 if Diagnosis!=Carcinoma or GG is NaN else
int(GG)` convention as `primus_vlm/dataset.py::parse_grade_group`) by
joining `core_id` (== `cine_id`) against each core's own `info.json`
under `root_dir_c3` (`/datasets/exactvu_pca/OPTIMUM/UA_OL_PU_annotated_needles_multiframe`,
the config's own OPTIMUM export root), and derives `patient_id` from
`core_id`'s own `<center>-<case>-<core>` format -- same conventions
`projects/prostnfound/eval_paper_metrics.py` uses for the same reason.

Usage (from the medAI repo root, matching this project's own `-m` launch convention):
    python -m projects.baseline_methods.guideus.eval_paper_metrics \
        --cfg projects/baseline_methods/guideus/guideus_cfg0_optimum.yaml \
        --ckpts fold0=/scratch/.../5567062/best.pth fold1=... \
        --out projects/baseline_methods/guideus/results_paper_metrics.json
"""
import argparse
import glob
import json
import logging
import os
from collections import defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf

from medAI.metrics import calculate_binary_classification_metrics as calc_metrics
from medAI.modeling.registry import create_model
from medAI.modeling import *  # noqa: F401,F403 -- registers every model factory, matches guideus_pnf_train.py's own import

from baseline.guideus.guideus_pnf_train import ProstNFoundMeta
from baseline.guideus.src.nct_optimum_loader_pnf import get_dataloaders

OPTIMUM_ROOT = "/datasets/exactvu_pca/OPTIMUM/UA_OL_PU_annotated_needles_multiframe"


def build_grade_group_lookup():
    """`core_id` (== cine_id) -> true grade_group (0 = benign, int(GG) o.w.),
    same convention as `primus_vlm/dataset.py::parse_grade_group`. Built once
    from the raw per-cine `info.json` files -- independent of whatever
    binary target `ProstNFoundTransform` produced for training."""
    lookup = {}
    for path in glob.glob(os.path.join(OPTIMUM_ROOT, "*", "*", "info.json")):
        with open(path) as f:
            info = json.load(f)
        cine_id = info.get("cine_id")
        if cine_id is None:
            continue
        if info.get("Diagnosis") != "Carcinoma":
            gg = 0.0
        else:
            gg = info.get("GG")
            gg = 0.0 if gg is None or (isinstance(gg, float) and np.isnan(gg)) else float(gg)
        lookup[cine_id] = gg
    return lookup


@torch.no_grad()
def eval_one_fold(cfg_path, fold, ckpt_path, gg_lookup, device="cuda"):
    cfg = OmegaConf.load(cfg_path)
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"fold": fold}))

    model = create_model(cfg.model, **cfg.model_kw)
    model = ProstNFoundMeta(model, cfg=cfg, **cfg.get("metamodel", {}))
    model.to(device)
    model.eval()

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)

    loaders = get_dataloaders(cfg.data, mode="train")
    val_loader = loaders["val"]

    core_ids, patient_ids, preds, gg_true = [], [], [], []
    for data in val_loader:
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=True):
            data = model(data)
        pred = data["average_needle_heatmap_value"].float().cpu().numpy()
        for i, cid in enumerate(data["core_id"]):
            # ProstNFoundTransform's own output dict never includes
            # patient_id (confirmed by reading it directly) -- derive it
            # from core_id's own `<center>-<case>-<core>` format instead,
            # same convention as projects/prostnfound's eval script.
            core_ids.append(cid)
            patient_ids.append("-".join(cid.split("-")[:2]))
            preds.append(float(pred[i]))
            gg_true.append(gg_lookup[cid])

    preds = np.array(preds)
    gg_true = np.array(gg_true)
    is_cspca = (gg_true >= 2).astype(int)

    core_metrics = calc_metrics(preds, is_cspca)

    # patient-level: max-pool prediction, label = any core in patient is csPCa
    by_patient_pred = defaultdict(list)
    by_patient_label = defaultdict(list)
    for pid, p, l in zip(patient_ids, preds, is_cspca):
        by_patient_pred[pid].append(p)
        by_patient_label[pid].append(l)
    patient_ids_sorted = sorted(by_patient_pred)
    patient_preds = np.array([max(by_patient_pred[p]) for p in patient_ids_sorted])
    patient_labels = np.array([max(by_patient_label[p]) for p in patient_ids_sorted])
    patient_metrics = calc_metrics(patient_preds, patient_labels)

    return dict(
        fold=fold, n_cores=len(core_ids), n_patients=len(patient_ids_sorted),
        core=core_metrics, patient=patient_metrics,
    )


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    gg_lookup = build_grade_group_lookup()
    logging.info(f"grade_group lookup built for {len(gg_lookup)} cores")

    ckpts = dict(item.split("=", 1) for item in args.ckpts)
    results = []
    for fold_key, ckpt_path in ckpts.items():
        fold = int(fold_key.replace("fold", ""))
        logging.info(f"=== fold {fold} ({ckpt_path}) ===")
        r = eval_one_fold(args.cfg, fold, ckpt_path, gg_lookup, device=args.device)
        results.append(r)
        c, p = r["core"], r["patient"]
        logging.info(
            f"[fold {fold}] n_cores={r['n_cores']} n_patients={r['n_patients']} | "
            f"CORE auc={c['auc']:.3f} sens@60spe={c['sens_at_60_spe']:.3f} bal_acc={c['balanced_acc_best']:.3f} | "
            f"PATIENT(max) auc={p['auc']:.3f} sens@60spe={p['sens_at_60_spe']:.3f} bal_acc={p['balanced_acc_best']:.3f}"
        )

    for level in ["core", "patient"]:
        for key in ["auc", "sens_at_60_spe", "balanced_acc_best"]:
            vals = [r[level][key] for r in results]
            logging.info(f"MEAN {level} {key}: {np.mean(vals):.3f}+-{np.std(vals):.3f}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"written to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--ckpts", nargs="+", required=True, help="foldN=/path/to/best.pth ...")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(args)
