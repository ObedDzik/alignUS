"""
train_patched.py — drop-in replacement for the training/eval loop.

Changes that matter, in order:

  [BUG]  ProstNFoundMeta.forward unpacked 7 values from NeedleABMILWrapper,
         which now returns 8 (needle_valid_mask was added). This raises ValueError at
         the first forward pass. Fixed below, and needle_valid_mask is now propagated
         as data["needle_valid_mask"] — every masked diagnostic depends on it.

  [BUG]  run_linear_probe built `log_dict` only inside `if log_wandb:` and only
         inside a try, then returned it. If wandb.run is None -> UnboundLocalError.
         It also never actually called wandb.log(). Both fixed.

  [NEW]  Every eval epoch dumps the full probability matrix + labels + core ids
         to .npz. Any metric you forget today is recomputable offline tomorrow.
         This is the change that stops you rerunning five methods x five folds.

  [NEW]  Binary csPCa (GG>=2) AUC, binary any-cancer AUC, QWK, MAE, adjacent
         accuracy, confusion matrix — computed from the same probability matrix.

  [NEW]  Model-agnostic patch heterogeneity (cosine dispersion, dispersion index,
         effective rank) logged for EVERY method, including ones with no
         patch_cancer_head. This is the all-baselines mechanism figure.

  [NEW]  Diagnostics keyed to a GLOBAL step counter on a fixed interval, written
         to one tidy CSV per run, so all methods land on a shared x-axis.

  [FIX]  Attention entropy normalized by ln(N_valid) per sample rather than
         ln(N_max), which varies with batch composition.

  [FIX]  Patch score mean/std computed over valid needle tokens only. The
         zero-padded positions pass through proj->LayerNorm->ReLU as a constant
         non-zero vector, biasing both the proportion target and the std.

  [FIX]  train_loss was divided by accumulate_grad_steps twice when logging.
"""

import logging
import os
from tempfile import mkdtemp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from diagnostics import (
    DiagRecorder,
    setup_wandb_metrics,
    alignment_metrics,
    classification_metrics,
    collect_step_diagnostics,
    confusion_to_json,
    normalized_attention_entropy,
    save_predictions,
)


import argparse
from argparse import ArgumentParser, BooleanOptionalAction
from collections import defaultdict
import json
import logging
from tempfile import mkdtemp
import typing as tp
import PIL
import copy
from PIL import Image
from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import random
from scipy.ndimage import gaussian_filter
import os
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import pandas as pd
from einops import rearrange, repeat
from medAI.modeling.prostnfound import ProstNFound
from medAI.modeling.setr import SETR
from torch.nn import functional as F
from tqdm import tqdm
from torchvision.transforms import v2 as T
from src.abmil_ce import ABMILISUP
from medAI.modeling.registry import create_model, list_models, register_model
from medAI.factories.prostnfound.models import get_model
from medAI.modeling import *
from medAI.utils.argparse import UpdateDictAction
from medAI.utils.reproducibility import (get_all_rng_states,set_all_rng_states,set_global_seed,)
from medAI.utils.accumulators import DataFrameCollector
from losses import build_alignus_loss as build_loss
from medAI.layers.masked_prediction_module import MaskedPredictionModule
from src.loaders import check_grade_distribution #get_dataloaders
from src.nct_optimum_loader import get_dataloaders
# from projects.ggnus_align.nct_trvl_op_ts import get_dataloaders
from medAI.engine.prostnfound.evaluator import (
    ProstNFoundEvaluator as Evaluator,
)
OmegaConf.register_new_resolver('getenv', os.getenv)
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from src.get_dino_model import dinov3_vitl16
from src.abmil_wrapper import NeedleABMILWrapper
from sklearn.preprocessing import StandardScaler

import numpy as np
import scipy.linalg


# =========================================================================== #
# 1. MODEL WRAPPER — unpack fix + mask propagation                             #
# =========================================================================== #


class ProstNFoundMeta(nn.Module):
    """Only forward() differs from your version; everything else is unchanged."""

    """Wraps a model to perform forward pass with ProstNFound style training

    Args:
        model: The model to wrap.
        mask_output_key: The key to use for the mask output (if the model outputs a dictionary of tensors)
    """

    def __init__(self, model, cfg=None, mask_output_key=None):
        super().__init__()
        encoder = model
        self.mask_output_key = mask_output_key
        self.train_mode = cfg.train_mode
        self.model_type = cfg.model_type

        self.register_buffer("temperature", torch.tensor([1.0]))
        self.register_buffer("bias", torch.tensor([0.0]))
 
        use_acmil = cfg.get('use_acmil', False)
        acmil_n_branches = cfg.get('acmil_n_branches', 5)
        acmil_mask_drop_prob = cfg.get('acmil_mask_drop_prob', 0.6)
        pooling = cfg.get('pooling', 'attention')
        mask_size = cfg.data.get('mask_size', 128)

        abmil = ABMILISUP(input_dim=cfg.input_dim, proj_dim=cfg.proj_dim, 
                            attn_hidden=cfg.hidden_dim, p_input_dropout=0.1, 
                            num_classes=6, use_acmil=use_acmil, acmil_n_branches=acmil_n_branches,
                            acmil_mask_drop_prob=acmil_mask_drop_prob)
        self.model = NeedleABMILWrapper(encoder=encoder, abmil=abmil, 
                    model_type=cfg.model_type, mask_size=mask_size,
                    pooling=pooling).to('cuda')

    
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, data, include_postprocessed_heatmaps=False):
        bmode = data["bmode"].to(self.device)
        needle_mask = data["needle_mask"].to(self.device)
        prostate_mask = data["prostate_mask"].to(self.device)
        neg_grade_key = None

        # -------- FIX: wrapper returns 8 values, not 7 --------
        (
            logits,
            A,
            feats,
            raw_patch_feats,
            needle_valid_mask,
            needle_indices,
            grid_H,
            grid_W,
        ) = self.model(bmode, needle_mask=needle_mask)

        if self.train_mode == "us_only":
            positive = torch.zeros_like(data["positive_hist"])
            negative = torch.zeros_like(data["negative_hist"])
        elif self.train_mode in ("us_mri", "us_mri_align_only"):
            positive = data["positive_mri"]
            negative = data["negative_mri"]
            neg_grade_key = data["negative_mri_grade"]
        elif self.train_mode == "us_histo":
            positive = data["positive_hist"]
            negative = data["negative_hist"]
            neg_grade_key = data["negative_hist_grade"]
        elif self.train_mode == "us_mri_histo":
            positive = data["positive_hist"]
            negative = data["negative_hist"]
            neg_grade_key = data["negative_hist_grade"]
            data["joint_positive_hist"] = positive
            data["joint_negative_hist"] = negative
            data["joint_positive_mri"] = data["positive_mri"]
            data["joint_negative_mri"] = data["negative_mri"]
            data["joint_neg_grade_key_hist"] = neg_grade_key
            data["joint_neg_grade_key_mri"] = data["negative_mri_grade"]
        else:
            raise ValueError("Invalid training mode specified")

        data["positive_hist"] = F.normalize(positive, p=2, dim=1)
        data["negative_hist"] = F.normalize(negative, p=2, dim=1)
        data["image_feats_needle"] = F.normalize(feats, p=2, dim=1)
        data["isup_logits"] = logits
        data["attention"] = A
        data["negative_grade"] = neg_grade_key
        data["patch_feats_needle"] = raw_patch_feats
        data["needle_valid_mask"] = needle_valid_mask          # <-- NEW, required downstream
        data["needle_indices"] = needle_indices
        data["grid_H"] = grid_H
        data["grid_W"] = grid_W
        return data

    
    def get_params_groups(self, freeze_encoder=False):
        encoder_parameters = []
        warmup_parameters = []
        cnn_parameters = []

        if self.model_type == 'microsegnet':
            if isinstance(self.model, NeedleABMILWrapper):
                warmup_parameters = []
                encoder_parameters = []
                for name, param in self.model.named_parameters():
                    if name.startswith("encoder."):
                        is_cnn_stem = (
                            ".hybrid_model." in name
                            or ".patch_embeddings." in name
                            or name.endswith(".position_embeddings")
                        )
                        if is_cnn_stem:
                            param.requires_grad = False
                        else:
                            param.requires_grad = not freeze_encoder
                            if not freeze_encoder:
                                encoder_parameters.append(param)
                    else:
                        param.requires_grad = True
                        warmup_parameters.append(param)

                frozen_cnn = sum(p.numel() for n, p in self.model.named_parameters()
                                if n.startswith("encoder.") and not p.requires_grad)
                trainable_vit = sum(p.numel() for p in encoder_parameters)
                trainable_head = sum(p.numel() for p in warmup_parameters)
                logging.info(
                    f"[NeedleABMILWrapper] freeze_encoder={freeze_encoder} | "
                    f"Frozen (CNN stem): {frozen_cnn:,} | "
                    f"Trainable (ViT): {trainable_vit:,} | "
                    f"Trainable (head): {trainable_head:,}"
                )
        else:
            if isinstance(self.model, NeedleABMILWrapper):
                for name, param in self.model.named_parameters():
                    if "encoder" in name:
                        if freeze_encoder:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                            encoder_parameters.append(param)
                    else:
                        param.requires_grad = True
                        warmup_parameters.append(param)
                encoder_params = sum(p.numel() for n, p in self.model.named_parameters() if "encoder" in n)
                trainable = sum(p.numel() for p in warmup_parameters)
                logging.info(
                    f"[NeedleABMILWrapper] freeze_encoder={freeze_encoder} | "
                    f"Encoder_parameters: {encoder_params:,} | Trainable (head): {trainable:,} | "
                    f"Encoder in optimizer: {not freeze_encoder}"
                )

            elif isinstance(self.model, SETR):
                for name, param in self.model.named_parameters():
                    if "head" in name:
                        warmup_parameters.append(param)
                    else:
                        encoder_parameters.append(param)

            elif hasattr(self.model, "image_encoder"):
                for name, param in self.model.named_parameters():
                    if "image_encoder" in name:
                        encoder_parameters.append(param)
                    else:
                        warmup_parameters.append(param)

            elif hasattr(self.model, "get_params_groups"):
                return self.model.get_params_groups()

            elif isinstance(self.model, ProstNFound):
                return self.model.get_params_groups()

            else:
                logging.warning("No matched model branch — using all parameters as warmup")
                warmup_parameters = list(self.model.parameters())

        return encoder_parameters, warmup_parameters, cnn_parameters


# =========================================================================== #
# 2. LINEAR PROBE — fixed, with binary + ordinal metrics and prediction dump   #
# =========================================================================== #


def run_linear_probe(
    cfg,
    X_tr,
    y_tr,
    X_va,
    y_va,
    meta_va: dict | None = None,
    epoch: int = 0,
    save_embeddings: bool = False,
    C=1.0,
    max_iter=500,
    class_weight="balanced",
    verbose=True,
    log_wandb=True,
):
    """
    Returns a flat metric dict AND writes the raw probability matrix to disk.

    meta_va: optional dict of per-sample arrays (core_id, patient_id, involvement,
             center, attn_entropy). Saved alongside predictions so the paired
             per-fold statistics and the entropy-vs-involvement correlation can
             be computed offline without touching the GPU again.
    """
    log_dict = {}

    clf = LogisticRegression(
        penalty="l2", C=C, solver="saga", max_iter=max_iter, class_weight=class_weight
    )
    clf.fit(X_tr, y_tr)
    y_proba = clf.predict_proba(X_va)
    classes = clf.classes_.astype(int)
    y_hat = classes[np.argmax(y_proba, axis=1)]

    metrics = classification_metrics(y_va, y_proba, classes, prefix="linear_probe/")
    log_dict.update(metrics)

    # ---- persist everything needed to recompute any metric later ----
    payload = {"y_true": y_va, "y_proba": y_proba, "classes": classes, "X_val": X_va}
    if meta_va:
        for k, v in meta_va.items():
            payload[f"meta_{k}"] = v
    pred_dir = getattr(cfg, "prediction_dir", None) or os.path.join(
        cfg.checkpoint_dir or ".", "predictions"
    )
    save_predictions(
        pred_dir,
        cfg.wandb.run_name,
        getattr(cfg, "fold", 0),
        epoch,
        payload,
        save_embeddings=save_embeddings,
    )

    if verbose:
        print(
            f"[Linear Probe] macro_auc={metrics['linear_probe/macro_auc']:.4f} | "
            f"csPCa_auc={metrics.get('linear_probe/auc_csPCa', float('nan')):.4f} | "
            f"cancer_auc={metrics.get('linear_probe/auc_cancer', float('nan')):.4f} | "
            f"qwk={metrics['linear_probe/qwk']:.4f} | "
            f"adj_acc={metrics['linear_probe/adjacent_acc']:.4f}"
        )
        print("Confusion matrix:\n", confusion_matrix(y_va, y_hat, labels=list(range(6))))
        print(classification_report(y_va, y_hat, digits=4, zero_division=0))

    confusion_to_json(
        y_va,
        y_hat,
        os.path.join(pred_dir, f"cm_{cfg.wandb.run_name}_ep{epoch:03d}.json"),
    )

    if log_wandb and wandb.run is not None:
        wandb.log(log_dict)

    return log_dict


# =========================================================================== #
# 3. TRAIN EPOCH — global-step diagnostics                                     #
# =========================================================================== #


def _get_patch_cancer_head(criterion, enabled: bool):
    """Returns the head if this run has one, else None. Baselines return None
    and still log every model-agnostic metric."""
    if not enabled:
        return None
    for loss_fn, name in zip(
        getattr(criterion, "losses", []), getattr(criterion, "names", [])
    ):
        if name == "prop_bce" and hasattr(loss_fn, "patch_cancer_head"):
            return loss_fn.patch_cancer_head
    return None


def run_train_epoch(
    args,
    model,
    loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    epoch,
    recorder: DiagRecorder,
    desc="train",
):
    model.train()
    epoch_feats, epoch_labels = [], []
    niter = len(loader)
    patch_head = _get_patch_cancer_head(criterion, getattr(args, "propbce", False))

    for train_iter, data in enumerate(tqdm(loader, desc=desc)):
        if args.debug and train_iter > 10:
            break

        global_step = epoch * niter + train_iter

        with torch.cuda.amp.autocast(enabled=args.use_amp):
            data = model(data)
            loss = criterion(data)

        epoch_feats.append(data["image_feats_needle"].detach().float().cpu().numpy())
        epoch_labels.append(data["bucket_label"].detach().long().cpu().numpy())

        # ---------------- diagnostics on a fixed global-step grid ----------------
        if recorder.should_log(global_step):
            collect_step_diagnostics(
                data,
                recorder=recorder,
                global_step=global_step,
                epoch=epoch,
                patch_cancer_head=patch_head,
                wandb_run=wandb.run,
            )

        loss = loss / args.accumulate_grad_steps
        if args.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (train_iter + 1) % args.accumulate_grad_steps == 0:
            if args.use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        step_metrics = {
            # FIX: was divided by accumulate_grad_steps twice
            "train_loss": loss.item() * args.accumulate_grad_steps,
            "global_step": global_step,
        }
        for i, pg in enumerate(optimizer.param_groups):
            names = ["encoder_lr", "main_lr", "cnn_lr"]
            step_metrics[names[i] if i < len(names) else f"group_{i}_lr"] = pg["lr"]
        wandb.log(step_metrics)

    args._global_step = global_step
    recorder.flush()
    X = np.concatenate(epoch_feats, axis=0) if epoch_feats else np.empty((0, 1))
    y = np.concatenate(epoch_labels, axis=0) if epoch_labels else np.empty((0,), dtype=np.int64)
    return X, y


# =========================================================================== #
# 4. EVAL EPOCH — collects ids, involvement, per-sample entropy                #
# =========================================================================== #


def _to_list(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy().tolist()
    return list(v)


@torch.no_grad()
def run_eval_epoch(args, model, loader, epoch, X_tr, y_tr, desc="val"):
    model.eval()
    feats, labels, histo_raw = [], [], []
    isup_logits, isup_labels = [], []
    meta = {
        "core_id": [],
        "patient_id": [],
        "center": [],
        "involvement": [],
        "grade_group": [],
        "attn_entropy": [],
    }

    for data in tqdm(loader, desc=desc):
        if "positive_hist" in data and data["positive_hist"] is not None:
            histo_raw.append(data["positive_hist"].detach().float().cpu().numpy())
        else:
            histo_raw.append(None)
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            data = model(data)

        feats.append(data["image_feats_needle"].detach().float().cpu().numpy())
        labels.append(data["bucket_label"].detach().long().cpu().numpy())

        if "isup_logits" in data:
            isup_logits.append(data["isup_logits"].detach().float().cpu())
            isup_labels.append(data["grade_group"].detach().long().cpu())

        ent = normalized_attention_entropy(
            data["attention"].detach().float(), data["needle_valid_mask"].detach().bool()
        )
        meta["attn_entropy"] += ent.cpu().numpy().tolist()
        meta["involvement"] += _to_list(data["involvement"])
        meta["grade_group"] += _to_list(data["grade_group"])
        meta["core_id"] += _to_list(data["core_id"])
        meta["patient_id"] += _to_list(data["patient_id"])
        meta["center"] += _to_list(data["center"])

    X_val = np.concatenate(feats, axis=0) if feats else np.empty((0, 1))
    y_val = np.concatenate(labels, axis=0) if labels else np.empty((0,), dtype=np.int64)
    meta = {k: np.array(v) for k, v in meta.items()}

    H_val = (
        np.concatenate([h for h in histo_raw if h is not None], axis=0)
        if any(h is not None for h in histo_raw)
        else np.empty((0, 1))
    )
 
    # Embeddings are ~80x the size of everything else, so write them on a
    # cadence rather than every epoch. Alignment metrics are computed in-run
    # regardless; these dumps exist only for offline recomputation and t-SNE.
    embed_every = int(getattr(args, "embed_dump_every", 10) or 0)
    save_emb = bool(embed_every) and (
        epoch % embed_every == 0 or epoch == args.epochs - 1
    )
 
    metrics = {"epoch": epoch}
    # ---- cross-modal alignment: is the histo term doing anything at all? ----
    if H_val.shape[0] == X_val.shape[0] and X_val.shape[0] > 0:
        metrics.update(alignment_metrics(X_val, H_val, meta["grade_group"]))

    metrics.update(
        run_linear_probe(
            args, X_tr, y_tr, X_val, y_val,
            meta_va={**meta, "histo_embed": H_val} if H_val.shape[0] == X_val.shape[0]
            else meta,
            epoch=epoch, save_embeddings=save_emb,
            C=1.0, max_iter=500, class_weight="balanced",
            verbose=True, log_wandb=False,   # logged once below
        )
    )

    # ---- the ISUP classification head, scored the same way ----
    if isup_logits:
        logits = torch.cat(isup_logits).numpy()
        y_head = torch.cat(isup_labels).numpy()
        proba = torch.softmax(torch.tensor(logits), dim=1).numpy()
        metrics.update(
            classification_metrics(y_head, proba, np.arange(proba.shape[1]), prefix="head/")
        )
        pred_dir = getattr(args, "prediction_dir", None) or os.path.join(
            args.checkpoint_dir or ".", "predictions"
        )
        save_predictions(
            pred_dir, f"{args.wandb.run_name}_head", getattr(args, "fold", 0), epoch,
            {"y_true": y_head, "y_proba": proba, "classes": np.arange(proba.shape[1]),
             **{f"meta_{k}": v for k, v in meta.items()}},
            save_embeddings=False,
        )
 
    # ---- attention entropy vs. TRUE involvement: the one measure against a real label ----
    if len(meta["involvement"]) > 3:
        inv = meta["involvement"].astype(float)
        ent = meta["attn_entropy"].astype(float)
        ok = np.isfinite(inv) & np.isfinite(ent)
        if ok.sum() > 3 and np.std(inv[ok]) > 0 and np.std(ent[ok]) > 0:
            from scipy.stats import pearsonr, spearmanr
 
            metrics["val/entropy_involvement_pearson"] = float(pearsonr(ent[ok], inv[ok])[0])
            metrics["val/entropy_involvement_spearman"] = float(spearmanr(ent[ok], inv[ok])[0])
 
    metrics["global_step"] = getattr(args, "_global_step", epoch)
    wandb.log(metrics)
    return metrics

def setup_optimizer(args, model, train_loader):
    from torch.optim import AdamW
    freeze_encoder = getattr(args, 'freeze_encoder', False)
    encoder_parameters, warmup_parameters, cnn_parameters = model.get_params_groups(freeze_encoder=freeze_encoder)
    total_epochs = args.epochs
    encoder_frozen_epochs = args.warmup_epochs
    warmup_epochs = 5
    niter_per_ep = len(train_loader)
    warmup_lr_factor = args.warmup_lr / args.lr

    def compute_lr_multiplier(iter, is_encoder_or_cnn=True):
        schedule = args.get('scheduler', 'cosine')
        if schedule == 'constant':
            return 1
        if iter < encoder_frozen_epochs * niter_per_ep:
            if is_encoder_or_cnn:
                return 0
            else:
                if iter < warmup_epochs * niter_per_ep:
                    return (iter * warmup_lr_factor) / (warmup_epochs * niter_per_ep)
                else:
                    cur_iter_in_frozen_phase = iter - warmup_epochs * niter_per_ep
                    total_iter_in_frozen_phase = (encoder_frozen_epochs - warmup_epochs) * niter_per_ep
                    return (0.5 * (1 + np.cos(np.pi * cur_iter_in_frozen_phase / total_iter_in_frozen_phase)) * warmup_lr_factor)
        else:
            iter -= encoder_frozen_epochs * niter_per_ep
            if iter < warmup_epochs * niter_per_ep:
                return iter / (warmup_epochs * niter_per_ep)
            else:
                cur_iter = iter - warmup_epochs * niter_per_ep
                total_iter = (total_epochs - warmup_epochs - encoder_frozen_epochs) * niter_per_ep
                return 0.5 * (1 + np.cos(np.pi * cur_iter / total_iter))

    # Build param groups dynamically based on what's non-empty
    params = []
    lr_lambdas = []

    if len(encoder_parameters) > 0:
        params.append({"params": encoder_parameters, "lr": args.encoder_lr})
        lr_lambdas.append(lambda iter: compute_lr_multiplier(iter, is_encoder_or_cnn=True))
        logging.info(f"Optimizer: encoder group — {sum(p.numel() for p in encoder_parameters):,} params")

    if len(warmup_parameters) > 0:
        params.append({"params": warmup_parameters, "lr": args.lr})
        lr_lambdas.append(lambda iter: compute_lr_multiplier(iter, is_encoder_or_cnn=False))
        logging.info(f"Optimizer: head group — {sum(p.numel() for p in warmup_parameters):,} params")

    if len(cnn_parameters) > 0:
        params.append({"params": cnn_parameters, "lr": args.cnn_lr})
        lr_lambdas.append(lambda iter: compute_lr_multiplier(iter, is_encoder_or_cnn=True))
        logging.info(f"Optimizer: cnn group — {sum(p.numel() for p in cnn_parameters):,} params")
    assert len(params) > 0, "No parameters to optimize — check get_params_groups"
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = LambdaLR(optimizer, lr_lambdas)

    return optimizer, lr_scheduler


# =========================================================================== #
# 5. MAIN — only the diagnostics wiring differs                                #
# =========================================================================== #


def main(cfg):
    logging.basicConfig(
        level=logging.INFO if not cfg.debug else logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )
    if cfg.debug:
        cfg.name = "debug"

    wandb.init(project=cfg.wandb.project, name=cfg.wandb.run_name,
               config=OmegaConf.to_object(cfg))
    setup_wandb_metrics()   # global_step as x-axis; must come right after init
    _tmpdir = mkdtemp()
    OmegaConf.save(cfg, os.path.join(_tmpdir, "train_config.yaml"), resolve=True)
    wandb.save(os.path.join(_tmpdir, "train_config.yaml"), base_path=_tmpdir, policy="now")
    cfg.wandb_url = wandb.run.url if wandb.run else None

    if cfg.checkpoint_dir is not None:
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # ---- one CSV per run; concat across runs to build every dynamics figure ----
    recorder = DiagRecorder(
        out_dir=getattr(cfg, "diag_dir", None) or os.path.join(cfg.checkpoint_dir or ".", "diagnostics"),
        run_name=cfg.wandb.run_name,
        every=getattr(cfg, "diag_every", 10),   # fixed global-step grid, all methods
    )

    if cfg.checkpoint_dir is not None:
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        state = None
    else:
        state = None

    set_global_seed(cfg.seed)                     # noqa: F821  (your helper)

    if cfg.model_type == 'dino':
        model = dinov3_vitl16()
    elif cfg.model_type == 'medsam':
        from medAI.modeling.sam import medsam_adapter
        model = medsam_adapter()
    elif cfg.model_type == 'microsegnet':
        from src.microseg_model import WrapperMicroSegNet
        model = WrapperMicroSegNet(
            checkpoint_path=cfg.microsegnet_checkpoint,
            img_size=224, n_skip=3, vit_name='R50-ViT-B_16', vit_patches_size=16,
            freeze=False,   # fine-tuning baseline
        )
    print(f"Model: {type(model)}")

    print(f"Model: {type(model)}")
    model = ProstNFoundMeta(model, cfg=cfg, **cfg.get('metamodel', {}))
    model.to(cfg.device)
    if cfg.torch_compile:
        torch.compile(model)
    logging.info("Model setup complete")
    logging.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    logging.info(
        f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    if cfg.model_checkpoint:
        model_state = torch.load(cfg.model_checkpoint, map_location="cpu",weights_only=False)
        if "model" in model_state:
            model_state = model_state["model"]
        msg = model.load_state_dict(model_state, strict=False)
        logging.info(f"Loaded model from {cfg.model_checkpoint} with message `{msg}`.")
    if state is not None:
        model.load_state_dict(state["model"])

    criterion = build_loss(cfg)                   # noqa: F821
    loaders = get_dataloaders(cfg.data, mode="train")   # noqa: F821
    train_loader, val_loader = loaders["train"], loaders["val"]
    optimizer, lr_scheduler = setup_optimizer(cfg, model, train_loader)   # noqa: F821

    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        lr_scheduler.load_state_dict(state["lr_scheduler"])

    scaler = torch.cuda.amp.GradScaler()
    if state is not None:
        scaler.load_state_dict(state["gradient_scaler"])

    epoch = 0 if state is None else state["epoch"]
    logging.info(f"Starting at epoch {epoch}")
    best_score = 0 if state is None else state["best_score"]
    logging.info(f"Best score so far: {best_score}")
    if state is not None:
        rng_state = state["rng"]
        set_all_rng_states(rng_state)

    model.return_dict = True
    print("!!!!!!!!! model return dict state: ", model.return_dict)

    def get_state():
        return {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "gradient_scaler": scaler.state_dict(),
            "rng": get_all_rng_states(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "args": vars(cfg),
        }

    def save_checkpoint(name):
        state = get_state()
        if cfg.checkpoint_dir is not None:
            logging.info(f"Saving experiment snapshot to {cfg.checkpoint_dir}")
            torch.save(state, os.path.join(cfg.checkpoint_dir, name))
            if cfg.save_checkpoint_wandb:
                wandb.save(
                    os.path.join(cfg.checkpoint_dir, name),
                    base_path=cfg.checkpoint_dir,
                    policy="now",
                )

    best_score = 0.0
    for epoch in range(cfg.epochs):
        if cfg.cutoff_epoch is not None and epoch > cfg.cutoff_epoch:
            break
        logging.info(f"Epoch {epoch}")
        X, y = run_train_epoch(
            cfg, model, train_loader, criterion, optimizer, lr_scheduler,
            scaler, epoch, recorder=recorder, desc="train",
        )
        if cfg.run_val:
            val_metrics = run_eval_epoch(cfg, model, val_loader, epoch, X, y, desc="val")
            tracked = val_metrics.get(cfg.tracked_metric)
            if tracked is not None and tracked > best_score:
                best_score = tracked
                logging.info(f"New best score: {best_score}")
                if cfg.save_best_weights:
                    save_checkpoint("best.pth")

    recorder.flush()
    logging.info("Finished training")


if __name__ == "__main__":
    p = ArgumentParser(description="Train ProstNFound model")
    p.add_argument('--config', '-c', help='Path to config file (located in cfg/train/...)')
    args = p.parse_args()
    cfg = OmegaConf.load(args.config)

    main(cfg)