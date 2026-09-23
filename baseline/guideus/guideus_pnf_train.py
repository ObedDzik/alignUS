"""
unified_meta.py — one ProstNFoundMeta + one train/eval loop covering all three
experiment types.

    alignus  : encoder -> needle pooling (ABMIL) -> alignment loss. No decoder.
    guideus  : ProstNFound -> decoder cancer logits (loss)  AND  image_feats ->
               the SAME needle pooling -> alignment loss.
    pnf      : ProstNFound -> decoder cancer logits only. No pooling, no
               alignment; image_feats_needle is a mean over all tokens purely so
               the linear probe still has something to read.

KEY DESIGN CHOICE
-----------------
Needle pooling is factored into needle_pool(), called by BOTH alignus and
guideus. Previously guideus had its own copy with sample-0 indexing while
alignus used per-sample padded indexing — so the two differed in pooling as well
as in objective, and your margin would not have been attributable to the loss.
One function, one behaviour, one thing varying between the rows.

For guideus this also means the image encoder runs ONCE. Wrapping
model.image_encoder in NeedleABMILWrapper while also calling the full
ProstNFound forward would run it twice per step: double compute, and two
separate graphs through shared parameters.

BUGS FIXED FROM THE VERSION YOU PASTED
  [1] `if self.experiment_type == 'pnf' or 'guideus':` is ALWAYS True — the bare
      string is truthy. alignus was entering the cancer-logits block and only
      escaping because the isinstance check failed.
  [2] guideus set self.model = NeedleABMILWrapper(encoder=model.image_encoder),
      then called self.model(bmode, rf, prostate_mask, needle_mask, ...) as if it
      were ProstNFound. Those cannot both hold.
  [3] `positive = torch.zeros_like(data["positive_hist"])` in us_only mode
      requires positive_hist to exist. For pnf runs whose loader omits it, this
      raises. Now guarded.
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
from baseline.guideus.src.abmil_ce import ABMILISUP
from medAI.modeling.registry import create_model, list_models, register_model
from medAI.factories.prostnfound.models import get_model
from medAI.modeling import *
from medAI.utils.argparse import UpdateDictAction
from medAI.utils.reproducibility import (get_all_rng_states,set_all_rng_states,set_global_seed,)
from medAI.utils.accumulators import DataFrameCollector
from losses import build_guidepnf_loss as build_loss
from medAI.layers.masked_prediction_module import MaskedPredictionModule
from src.loaders import check_grade_distribution #get_dataloaders

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
from medAI.layers.masked_prediction_module import MaskedPredictionModule

import numpy as np
import scipy.linalg
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from baseline.guideus.src.diagnostics_guide_pnf import (
    DiagRecorder,
    setup_wandb_metrics,
    alignment_metrics,
    classification_metrics,
    collect_step_diagnostics,
    confusion_to_json,
    normalized_attention_entropy,
    save_predictions,
)


# =========================================================================== #
# Shared needle pooling — used identically by alignus and guideus              #
# =========================================================================== #


def needle_token_mask(needle_mask: torch.Tensor, grid_H: int, grid_W: int) -> torch.Tensor:
    """(B, mask_size, mask_size) pixel mask -> (B, grid_H, grid_W) token mask."""
    m = needle_mask.bool()
    nH, nW = m.shape[-2], m.shape[-1]
    if nH % grid_H != 0 or nW % grid_W != 0:
        raise ValueError(
            f"needle_mask ({nH}, {nW}) does not evenly divide encoder grid "
            f"({grid_H}, {grid_W}). Check mask_size against the encoder's "
            f"actual output resolution."
        )
    return m.view(m.shape[0], grid_H, nH // grid_H, grid_W, nW // grid_W).any(dim=(2, 4))


def needle_pool(image_tokens: torch.Tensor, needle_mask: torch.Tensor, abmil: nn.Module,
                pooling: str = "attention"):
    """
    image_tokens : (B, grid_H, grid_W, C)
    needle_mask  : (B, mask_size, mask_size)

    Returns logits, A, z, H, valid_mask, indices_sample0, grid_H, grid_W.

    Per-sample needle indices padded to N_max, with the padding mask handed to
    ABMIL. Reusing one core's needle geometry across the batch means every other
    core pools over the wrong tokens whenever needle position varies.
    """
    B, grid_H, grid_W, C = image_tokens.shape
    device = image_tokens.device

    tmask = needle_token_mask(needle_mask.to(device), grid_H, grid_W)
    idxs = [torch.nonzero(tmask[b], as_tuple=False) for b in range(B)]
    Ns = [i.shape[0] for i in idxs]
    N_max = max(Ns)
    if N_max == 0:
        raise ValueError("No needle tokens in this batch — check mask alignment.")

    # dtype matched to image_tokens: under autocast these are half, and a float32
    # buffer forces a silent cast on every assignment.
    feats = torch.zeros(B, N_max, C, device=device, dtype=image_tokens.dtype)
    valid = torch.zeros(B, N_max, dtype=torch.bool, device=device)
    for b in range(B):
        n = Ns[b]
        if n == 0:
            continue
        feats[b, :n] = image_tokens[b, idxs[b][:, 0], idxs[b][:, 1]]
        valid[b, :n] = True

    # mask MUST be passed, else ABMIL softmaxes over padding
    logits, A, z, H = abmil(feats, mask=valid, pooling=pooling)
    return logits, A, z, H, valid, idxs[0], grid_H, grid_W


# =========================================================================== #
# Meta model                                                                   #
# =========================================================================== #


class ProstNFoundMeta(nn.Module):
    def __init__(self, model, cfg=None, mask_output_key=None):
        super().__init__()
        self.mask_output_key = mask_output_key
        self.train_mode = cfg.train_mode
        self.model_type = cfg.model_type
        self.experiment_type = cfg.get("experiment_type", "guideus")
        self.pooling = cfg.get("pooling", "attention")
        self.mask_size = cfg.data.get("mask_size", 128)

        self.register_buffer("temperature", torch.tensor([1.0]))
        self.register_buffer("bias", torch.tensor([0.0]))

        # Which capabilities does this experiment type have?
        self.use_needle_pool = self.experiment_type in ("alignus", "guideus")
        self.use_decoder = self.experiment_type in ("guideus", "pnf")

        abmil = None
        if self.use_needle_pool:
            abmil = ABMILISUP(                                  # noqa: F821
                input_dim=cfg.input_dim, proj_dim=cfg.proj_dim,
                attn_hidden=cfg.hidden_dim, p_input_dropout=0.1, num_classes=6,
                use_acmil=cfg.get("use_acmil", False),
                acmil_n_branches=cfg.get("acmil_n_branches", 5),
                acmil_mask_drop_prob=cfg.get("acmil_mask_drop_prob", 0.6),
            )

        if self.experiment_type == "alignus":
            # encoder owned by the wrapper; wrapper runs it and pools
            self.model = NeedleABMILWrapper(                     # noqa: F821
                encoder=model, abmil=abmil, model_type=cfg.model_type,
                mask_size=self.mask_size, pooling=self.pooling,
            )
            self.abmil = None
        else:
            # guideus + pnf: keep the FULL ProstNFound so the decoder is reachable.
            # For guideus the ABMIL lives here and consumes outputs["image_feats"],
            # so the image encoder runs once, not twice.
            self.model = model
            self.abmil = abmil

        logging.info(
            f"[ProstNFoundMeta] experiment_type={self.experiment_type} | "
            f"needle_pool={self.use_needle_pool} | decoder={self.use_decoder}"
        )

    @property
    def device(self):
        return next(self.parameters()).device

    # ------------------------------------------------------------------ #

    def _set_alignment_targets(self, data):
        """Populates positive_hist / negative_hist / negative_grade by train_mode."""
        neg_grade_key = None
        tm = self.train_mode

        if tm == "us_only":
            ref = data.get("positive_hist", data.get("positive_mri"))
            if ref is None:
                return                       # pnf loaders may omit these entirely
            positive = torch.zeros_like(ref)
            negative = torch.zeros_like(ref)
        elif tm in ("us_mri", "us_mri_align_only"):
            positive, negative = data["positive_mri"], data["negative_mri"]
            neg_grade_key = data["negative_mri_grade"]
        elif tm == "us_histo":
            positive, negative = data["positive_hist"], data["negative_hist"]
            neg_grade_key = data["negative_hist_grade"]
        elif tm == "us_mri_histo":
            positive, negative = data["positive_hist"], data["negative_hist"]
            neg_grade_key = data["negative_hist_grade"]
            data["joint_positive_hist"] = positive
            data["joint_negative_hist"] = negative
            data["joint_positive_mri"] = data["positive_mri"]
            data["joint_negative_mri"] = data["negative_mri"]
            data["joint_neg_grade_key_hist"] = neg_grade_key
            data["joint_neg_grade_key_mri"] = data["negative_mri_grade"]
        else:
            raise ValueError(f"Invalid train_mode: {tm}")

        data["positive_hist"] = F.normalize(positive, p=2, dim=1)
        data["negative_hist"] = F.normalize(negative, p=2, dim=1)
        data["negative_grade"] = neg_grade_key

    def _decoder_forward(self, data, bmode, needle_mask, prostate_mask):
        """ProstNFound decoder path. Returns outputs dict (or None)."""
        rf = data["rf"].to(self.device) if "rf" in data else None
        B = len(bmode)

        if not isinstance(self.model, ProstNFound):              # noqa: F821
            out = self.model(bmode)
            cancer_logits = out[self.mask_output_key] if isinstance(out, dict) else out
            outputs = None
        else:
            prompts = {}
            for name in self.model.prompts:
                p = data[name].to(device=self.device, dtype=bmode.dtype)
                prompts[name] = p[:, None] if p.ndim == 1 else p
            outputs = self.model(bmode, rf, prostate_mask, needle_mask,
                                 output_mode="all", **prompts)
            cancer_logits = outputs["mask_logits"]
            data["mask_logits"] = outputs["mask_logits"]
            data["image_feats"] = outputs["image_feats"]
            # data["image_feats_priorneck"] = outputs["image_feats_priorneck"]
            data["image_level_classification_outputs"] = outputs["cls_outputs"]

        cancer_logits = (
            cancer_logits / self.temperature[None, None, None, :]
            + self.bias[None, None, None, :]
        )
        data["cancer_logits"] = cancer_logits

        for key, m in (
            ("average_needle_heatmap_value", (prostate_mask > 0.5) & (needle_mask > 0.5)),
            ("average_prostate_heatmap_value", prostate_mask > 0.5),
        ):
            preds, bidx = MaskedPredictionModule()(cancer_logits, m)     # noqa: F821
            data[key] = torch.stack(
                [preds[bidx == j].sigmoid().mean() for j in range(B)]
            )
        return outputs

    # ------------------------------------------------------------------ #

    def forward(self, data, include_postprocessed_heatmaps=False):
        bmode = data["bmode"].to(self.device)
        needle_mask = data["needle_mask"].to(self.device)
        prostate_mask = data["prostate_mask"].to(self.device)

        self._set_alignment_targets(data)

        # ---------------- decoder (guideus, pnf) ----------------
        outputs = None
        if self.use_decoder:
            outputs = self._decoder_forward(data, bmode, needle_mask, prostate_mask)

        # ---------------- needle pooling ----------------
        if self.experiment_type == "alignus":
            (logits, A, feats, H, valid, indices, gH, gW) = self.model(
                bmode, needle_mask=needle_mask
            )
        elif self.experiment_type == "guideus":
            # reuse the encoder features already computed by the decoder pass
            image_tokens = outputs["image_feats"].permute(0, 2, 3, 1)
            (logits, A, feats, H, valid, indices, gH, gW) = needle_pool(
                image_tokens, needle_mask, self.abmil, pooling=self.pooling
            )
        else:  # pnf — no pooling, no ABMIL, no attention
            image_tokens = data["image_feats"].permute(0, 2, 3, 1)
            data["image_feats_needle"] = F.normalize(
                image_tokens.mean(dim=(1, 2)), p=2, dim=1
            )
            return data

        data["image_feats_needle"] = F.normalize(feats, p=2, dim=1)
        data["isup_logits"] = logits
        data["attention"] = A
        data["patch_feats_needle"] = H
        data["needle_valid_mask"] = valid
        data["needle_indices"] = indices
        data["grid_H"] = gH
        data["grid_W"] = gW
        return data

    # ------------------------------------------------------------------ #

    def get_params_groups(self, freeze_encoder=False):
        """
        guideus needs self.abmil in the optimizer — it is NOT inside self.model
        (which is the ProstNFound), so iterating self.model.named_parameters()
        alone leaves it at random init and it never trains.
        """
        encoder_parameters, warmup_parameters, cnn_parameters = [], [], []

        if self.experiment_type == "alignus":
            for name, param in self.model.named_parameters():
                if "encoder" in name:
                    param.requires_grad = not freeze_encoder
                    if not freeze_encoder:
                        encoder_parameters.append(param)
                else:
                    param.requires_grad = True
                    warmup_parameters.append(param)
        else:
            if hasattr(self.model, "get_params_groups"):
                encoder_parameters, warmup_parameters, cnn_parameters = (
                    self.model.get_params_groups()
                )
                encoder_parameters = list(encoder_parameters)
                warmup_parameters = list(warmup_parameters)
                cnn_parameters = list(cnn_parameters)
            else:
                for name, param in self.model.named_parameters():
                    (encoder_parameters if "image_encoder" in name
                     else warmup_parameters).append(param)
            if self.abmil is not None:
                warmup_parameters += list(self.abmil.parameters())

        logging.info(
            f"[params] encoder={sum(p.numel() for p in encoder_parameters):,} | "
            f"head={sum(p.numel() for p in warmup_parameters):,} | "
            f"cnn={sum(p.numel() for p in cnn_parameters):,}"
        )
        return encoder_parameters, warmup_parameters, cnn_parameters


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
# Train loop                                                                   #
# =========================================================================== #


def _has_pooling_keys(data) -> bool:
    """pnf produces no attention / patch features; skip pooling diagnostics."""
    return all(
        k in data and data[k] is not None
        for k in ("attention", "needle_valid_mask", "patch_feats_needle")
    )


def _get_patch_cancer_head(criterion, enabled: bool):
    if not enabled:
        return None
    for loss_fn, name in zip(getattr(criterion, "losses", []),
                             getattr(criterion, "names", [])):
        if name == "prop_bce" and hasattr(loss_fn, "patch_cancer_head"):
            return loss_fn.patch_cancer_head
    return None


def run_train_epoch(args, model, loader, criterion, optimizer, scheduler, scaler,
                    epoch, recorder: DiagRecorder, desc="train"):
    model.train()
    epoch_feats, epoch_labels = [], []
    niter = len(loader)
    patch_head = _get_patch_cancer_head(criterion, getattr(args, "propbce", False))
    global_step = epoch * niter

    for train_iter, data in enumerate(tqdm(loader, desc=desc)):
        if args.debug and train_iter > 10:
            break
        global_step = epoch * niter + train_iter

        with torch.cuda.amp.autocast(enabled=args.use_amp):
            data = model(data)
            if "cancer_logits" in data and torch.any(torch.isnan(data["cancer_logits"])):
                logging.warning("NaNs in decoder logits")
            loss = criterion(data)

        epoch_feats.append(data["image_feats_needle"].detach().float().cpu().numpy())
        epoch_labels.append(data["bucket_label"].detach().long().cpu().numpy())

        # pooling diagnostics only where pooling exists
        if recorder.should_log(global_step) and _has_pooling_keys(data):
            collect_step_diagnostics(
                data, recorder=recorder, global_step=global_step, epoch=epoch,
                patch_cancer_head=patch_head, wandb_run=wandb.run,
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
# Eval loop                                                                    #
# =========================================================================== #


def _to_list(v):
    return v.detach().cpu().numpy().tolist() if isinstance(v, torch.Tensor) else list(v)


@torch.no_grad()
def run_eval_epoch(args, model, loader, epoch, X_tr, y_tr, desc="val"):
    model.eval()
    feats, labels, histo_raw = [], [], []
    isup_logits, isup_labels = [], []
    meta = {k: [] for k in ("core_id", "patient_id", "center", "involvement",
                            "grade_group", "attn_entropy")}

    for data in tqdm(loader, desc=desc):
        # captured BEFORE the forward pass: us_only mode zeroes positive_hist
        h = data.get("positive_hist")
        histo_raw.append(h.detach().float().cpu().numpy() if h is not None else None)

        with torch.cuda.amp.autocast(enabled=args.use_amp):
            data = model(data)

        feats.append(data["image_feats_needle"].detach().float().cpu().numpy())
        labels.append(data["bucket_label"].detach().long().cpu().numpy())

        if data.get("isup_logits") is not None:
            isup_logits.append(data["isup_logits"].detach().float().cpu())
            isup_labels.append(data["grade_group"].detach().long().cpu())

        if _has_pooling_keys(data):
            ent = normalized_attention_entropy(
                data["attention"].detach().float(),
                data["needle_valid_mask"].detach().bool(),
            )
            meta["attn_entropy"] += ent.cpu().numpy().tolist()

        for k in ("involvement", "grade_group", "core_id", "patient_id", "center"):
            if k in data and data[k] is not None:
                meta[k] += _to_list(data[k])

    X_val = np.concatenate(feats, axis=0) if feats else np.empty((0, 1))
    y_val = np.concatenate(labels, axis=0) if labels else np.empty((0,), dtype=np.int64)
    meta = {k: np.array(v) for k, v in meta.items() if len(v)}
    H_val = (
        np.concatenate([h for h in histo_raw if h is not None], axis=0)
        if any(h is not None for h in histo_raw) else np.empty((0, 1))
    )

    embed_every = int(getattr(args, "embed_dump_every", 10) or 0)
    save_emb = bool(embed_every) and (epoch % embed_every == 0 or epoch == args.epochs - 1)

    metrics = {"epoch": epoch, "global_step": getattr(args, "_global_step", epoch)}

    # alignment metrics: returns histo_available=0 for pnf / us_only zeroed histo
    if H_val.shape[0] == X_val.shape[0] and X_val.shape[0] > 0 and "grade_group" in meta:
        metrics.update(alignment_metrics(X_val, H_val, meta["grade_group"],
                                         prefix=f"{desc}/align/"))

    metrics.update(
        run_linear_probe(                                        # noqa: F821
            args, X_tr, y_tr, X_val, y_val,
            meta_va={**meta, "histo_embed": H_val}
            if H_val.shape[0] == X_val.shape[0] else meta,
            epoch=epoch, save_embeddings=save_emb,
        )
    )

    if isup_logits:
        proba = torch.softmax(torch.cat(isup_logits), dim=1).numpy()
        y_head = torch.cat(isup_labels).numpy()
        metrics.update(classification_metrics(
            y_head, proba, np.arange(proba.shape[1]), prefix=f"{desc}/head/"))
        pred_dir = getattr(args, "prediction_dir", None) or os.path.join(
            args.checkpoint_dir or ".", "predictions")
        save_predictions(
            pred_dir, f"{args.wandb.run_name}_head", getattr(args, "fold", 0), epoch,
            {"y_true": y_head, "y_proba": proba,
             "classes": np.arange(proba.shape[1]),
             **{f"meta_{k}": v for k, v in meta.items()}},
            save_embeddings=False,
        )

    if "attn_entropy" in meta and "involvement" in meta and \
            len(meta["attn_entropy"]) == len(meta["involvement"]):
        inv = meta["involvement"].astype(float)
        ent = meta["attn_entropy"].astype(float)
        ok = np.isfinite(inv) & np.isfinite(ent)
        if ok.sum() > 3 and np.std(inv[ok]) > 0 and np.std(ent[ok]) > 0:
            from scipy.stats import pearsonr, spearmanr

            metrics[f"{desc}/entropy_involvement_pearson"] = float(pearsonr(ent[ok], inv[ok])[0])
            metrics[f"{desc}/entropy_involvement_spearman"] = float(spearmanr(ent[ok], inv[ok])[0])

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


    model = create_model(cfg.model, **cfg.model_kw)
    # if cfg.model_type == 'dino':
    #     model = dinov3_vitl16()
    # elif cfg.model_type == 'medsam':
    #     from medAI.modeling.sam import medsam_adapter
    #     model = medsam_adapter()
    # elif cfg.model_type == 'microsegnet':
    #     from projects.ggnus_align.microseg_model import WrapperMicroSegNet
    #     model = WrapperMicroSegNet(
    #         checkpoint_path=cfg.microsegnet_checkpoint,
    #         img_size=224, n_skip=3, vit_name='R50-ViT-B_16', vit_patches_size=16,
    #         freeze=False,   # fine-tuning baseline
    #     )
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
    if cfg.experiment_type == 'pnf':
        from baseline.guideus.src.nct_optimum_loader_pnf import get_dataloaders
    else:
        from src.nct_optimum_loader import get_dataloaders
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