import argparse
import json
from typing import Callable
import torch
from torch import nn
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import repeat, rearrange
import logging

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, p=2):
        super().__init__()
        self.criterion = nn.TripletMarginLoss(margin=margin, p=p)

    def forward(self, data):
        if "cancer_logits" not in data or "negative_hist" not in data or "positive_hist" not in data:
            print("Returning 0.0")
            return torch.tensor(0.0, device=data["label"].device)
        embed_us = data["image_feats_needle"]
        # print("embed_us",embed_us.shape)
        # return
        X_pos = data["positive_hist"].to('cuda')
        X_neg = data["negative_hist"].to('cuda')
        # print(embed_us.shape,X_pos.shape,X_neg.shape)
        loss = self.criterion(embed_us, X_pos, X_neg)
        return loss
    

import torch
import torch.nn as nn
import torch.nn.functional as F
class SupervisedContrastiveCrossModal(nn.Module):
    def __init__(self, temperature=0.07, normalize=True, safe_float=True):
        super().__init__()
        self.tau = temperature
        self.normalize = normalize
        self.safe_float = safe_float  # upcast logits to float32 for stability if needed

    def forward(self, data):
        z1 = data["image_feats_needle"]        # [B, D]  (anchor; usually CUDA)
        z2 = data["positive_hist"]             # [B, D]  (may be CPU)
        y  = data["grade_group"]                   # [B]

        # unify device & dtype to the anchor
        device, dtype = z1.device, z1.dtype
        z2 = z2.to(device=device, dtype=dtype, non_blocking=True)
        y  = y.to(device=device, non_blocking=True)

        if self.normalize:
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)

        Z = torch.cat([z1, z2], dim=0)         # [2B, D]

        # compute similarities; upcast for numerical stability if using mixed precision
        logits = (Z @ Z.t())
        if self.safe_float and logits.dtype != torch.float32:
            logits = logits.float()
        logits = logits / self.tau             # [2B, 2B]

        # mask self-contrast
        n = logits.size(0)
        eye = torch.eye(n, device=device, dtype=torch.bool)
        # use finfo min instead of -inf to avoid AMP edge cases
        neg_inf = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(eye, neg_inf)

        labels = torch.cat([y, y], dim=0)      # [2B]
        pos_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & (~eye)

        # log-softmax over rows
        log_prob = logits - logits.logsumexp(dim=1, keepdim=True)

        # average over positives per anchor (skip anchors with zero positives)
        denom = pos_mask.sum(dim=1).clamp_min(1)
        loss = -(pos_mask * log_prob).sum(dim=1) / denom
        return loss.mean()


class SymmetricInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, data):
        us = data["image_feats_needle"]        
        hist = data["positive_hist"]          

        # unify device & dtype to the anchor (US) tensor
        device = us.device
        dtype = us.dtype
        hist = hist.to(device=device, dtype=dtype, non_blocking=True)


        logits_us_to_hist = (us @ hist.t()) / self.tau   
        logits_hist_to_us = (hist @ us.t()) / self.tau    

        targets = torch.arange(us.size(0), device=device)
        loss_us = F.cross_entropy(logits_us_to_hist, targets)
        loss_hist = F.cross_entropy(logits_hist_to_us, targets)
        return 0.5 * (loss_us + loss_hist)





class CancerDetectionValidRegionLoss(nn.Module):
    def __init__(
        self,
        base_loss: Callable = F.binary_cross_entropy_with_logits,
        prostate_mask: bool = True,
        needle_mask: bool = True,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.prostate_mask = prostate_mask
        self.needle_mask = needle_mask

    def forward(self, data: dict):
        cancer_logits = data["cancer_logits"]
        label = data["label"].to(cancer_logits.device)
        prostate_mask = data["prostate_mask"].to(cancer_logits.device)
        needle_mask = data["needle_mask"].to(cancer_logits.device)

        masks = []
        for i in range(len(cancer_logits)):
            mask = torch.ones(
                prostate_mask[i].shape, device=prostate_mask[i].device
            ).bool()
            if self.prostate_mask:
                mask &= prostate_mask[i] > 0.5
            if self.needle_mask:
                mask &= needle_mask[i] > 0.5
            masks.append(mask)
        masks = torch.stack(masks)
        predictions, batch_idx = MaskedPredictionModule()(cancer_logits, masks)
        labels = torch.zeros(len(predictions), device=predictions.device)
        for i in range(len(predictions)):
            labels[i] = label[batch_idx[i]]
        labels = labels[..., None]  # needs to match N, C shape of preds

        return self.base_loss(predictions, labels)


class ProportionBCE(nn.Module):
    def __init__(
        self,
        l1_penalty_lambda: float | None = None,
        entropy_penalty_lambda: float | None = None,
    ):
        super().__init__()
        self.l1_penalty_lambda = l1_penalty_lambda
        self.entropy_penalty_lambda = entropy_penalty_lambda

    def forward(self, bag_of_logits, true_prop):
        probs = bag_of_logits.sigmoid()
        pred_prob = probs.mean()

        loss = -true_prop * pred_prob.log() - (1 - true_prop) * (1 - pred_prob).log()

        if self.l1_penalty_lambda:
            loss = loss + self.l1_penalty_lambda * probs.abs().sum()
        if self.entropy_penalty_lambda:
            entropy = -probs * probs.log() - (1 - probs) * (1 - probs).log()
            loss = loss + self.entropy_penalty_lambda * entropy.mean()

        return loss


class CancerDetectionMILLoss(nn.Module):
    def __init__(self, base_loss=ProportionBCE(), treat_gg1_as_benign=False):

        super().__init__()
        self.base_loss = base_loss
        self.treat_gg1_as_benign = treat_gg1_as_benign

    def forward(self, data):
        cancer_logits = data["cancer_logits"]
        batch_size = len(cancer_logits)
        prostate_mask = data["prostate_mask"].to(cancer_logits.device)
        needle_mask = data["needle_mask"].to(cancer_logits.device)
        involvement = data["involvement"].to(cancer_logits.device)
        grade_group = data["grade_group"].to(cancer_logits.device)

        if self.treat_gg1_as_benign:
            involvement[grade_group == 1] = 0.0

        masks = []
        for i in range(len(cancer_logits)):
            mask = torch.ones(
                prostate_mask[i].shape, device=prostate_mask[i].device
            ).bool()
            mask &= prostate_mask[i] > 0.5
            mask &= needle_mask[i] > 0.5
            masks.append(mask)
        masks = torch.stack(masks)
        predictions, batch_idx = MaskedPredictionModule()(cancer_logits, masks)

        loss = torch.tensor(0, device=cancer_logits.device)
        for i in range(batch_size):
            bag_i = predictions[batch_idx == i]
            involvement_i = involvement[i]

            loss = loss + self.base_loss(bag_i, involvement_i)

        return loss


class InvolvementL1Loss(nn.Module):
    def __init__(self, prostate_penalty=True, pos_weight=1):
        super().__init__()
        self.prostate_penalty = prostate_penalty
        self.pos_weight = pos_weight

    def __call__(self, data):
        avg_needle_heatmap_value = data["average_needle_heatmap_value"]
        B = len(avg_needle_heatmap_value)
        device = avg_needle_heatmap_value.device
        avg_prostate_heatmap_value = data["average_prostate_heatmap_value"]
        involvement = data["involvement"].to(device)
        cores_positive_for_patient = data["cores_positive_for_patient"]

        loss = torch.tensor(0, device=device)
        loss = loss + torch.nn.functional.l1_loss(
            avg_needle_heatmap_value, involvement, reduction="none"
        )
        for idx in range(B):
            if involvement[idx] > 0:
                loss[idx] *= self.pos_weight
        loss = loss.mean()

        if self.prostate_penalty:
            for idx in range(B):
                if cores_positive_for_patient[idx] == 0:
                    loss += avg_prostate_heatmap_value[idx]

        return loss


class InvolvementMSELoss(nn.Module):
    def __call__(self, data):
        avg_needle_heatmap_value = data["average_needle_heatmap_value"]
        B = len(avg_needle_heatmap_value)
        device = avg_needle_heatmap_value.device
        involvement = data["involvement"].to(device)

        loss = torch.nn.functional.mse_loss(avg_needle_heatmap_value, involvement)
        return loss


class MaskedPredictionModule(nn.Module):
    """
    Computes the patch and core predictions and labels within the valid loss region for a heatmap.
    """

    def __init__(self):
        super().__init__()

    def forward(self, heatmap_logits, mask):
        """Computes the patch and core predictions and labels within the valid loss region."""
        B, C, H, W = heatmap_logits.shape

        assert mask.shape == (
            B,
            1,
            H,
            W,
        ), f"Expected mask shape to be {(B, 1, H, W)}, got {mask.shape} instead."

        # mask = mask.float()
        # mask = torch.nn.functional.interpolate(mask, size=(H, W)) > 0.5

        core_idx = torch.arange(B, device=heatmap_logits.device)
        core_idx = repeat(core_idx, "b -> b h w", h=H, w=W)

        core_idx_flattened = rearrange(core_idx, "b h w -> (b h w)")
        mask_flattened = rearrange(mask, "b c h w -> (b h w) c")[..., 0]
        logits_flattened = rearrange(heatmap_logits, "b c h w -> (b h w) c", h=H, w=W)

        logits = logits_flattened[mask_flattened]
        core_idx = core_idx_flattened[mask_flattened]

        patch_logits = logits

        return patch_logits, core_idx


class ImageLevelClassificationLoss(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, data):
        """
        Computes the image-level classification loss.
        """

        if "image_level_classification_outputs" not in data:
            return torch.tensor(0.0, device=data["label"].device)

        logits = data["image_level_classification_outputs"][0]

        if self.mode == "pca":
            labels = data["label"].to(logits.device)
        else:
            labels = (data["grade_group"] > 2).long().to(logits.device)

        # Compute the binary cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss


# class SumLoss(nn.Module):
#     def __init__(self, losses: list[nn.Module], weights=None):
#         super().__init__()
#         self.losses = nn.ModuleList(losses)
#         self.weights = weights if weights is not None else [1.0] * len(losses)

#     def forward(self, data):
#         loss = self.losses[0](data) * self.weights[0]
#         for i in range(1, len(self.losses)):
#             loss += self.losses[i](data) * self.weights[i]
#         return loss


class SumLoss(nn.Module):
    def __init__(self, losses, weights=None, names=None, log_every=10, normalize=False):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights if weights is not None else [1.0] * len(losses)
        self.names = names if names is not None else [f'loss_{i}' for i in range(len(losses))]
        self.log_every = log_every
        self.normalize = normalize
        self._step = 0
        self._loss_scale = None  # set on first forward pass

    def set_epoch(self, epoch: int):
        """Propagate epoch to any child loss that needs it."""
        for loss_fn in self.losses:
            if hasattr(loss_fn, 'set_epoch'):
                loss_fn.set_epoch(epoch)

    def forward(self, data):
        total_loss = None
        individual_losses = {}

        raw_losses = {}
        for loss_fn, name in zip(self.losses, self.names):
            raw_losses[name] = loss_fn(data)

        # on first step, record initial magnitudes as scale factors
        if self.normalize and self._loss_scale is None:
            self._loss_scale = {
                name: max(raw_losses[name].item(), 1e-8) 
                for name in self.names
            }
            logging.info(f"Loss scales initialized: { {k: f'{v:.4f}' for k, v in self._loss_scale.items()} }")

        for name, raw, weight in zip(self.names, raw_losses.values(), self.weights):
            scale = self._loss_scale[name] if (self.normalize and self._loss_scale) else 1.0
            component = (raw / scale) * weight
            individual_losses[name] = raw.item()  # log raw for interpretability
            total_loss = component if total_loss is None else total_loss + component

        individual_losses['total'] = total_loss.item()
        data['_loss_components'] = individual_losses

        if self._step % self.log_every == 0:
            loss_str = " | ".join(f"{name}={val:.4f}" for name, val in individual_losses.items())
            logging.info(f"[step {self._step}] {loss_str}")

        self._step += 1
        return total_loss


class OutsideProstatePenaltyLoss(nn.Module):

    def forward(self, data: dict):
        cancer_logits = data["cancer_logits"]
        prostate_mask = data["prostate_mask"].to(cancer_logits.device)

        masks = []
        for i in range(len(cancer_logits)):
            mask = torch.ones(
                prostate_mask[i].shape, device=prostate_mask[i].device
            ).bool()
            mask &= prostate_mask[i] < 0.5
            masks.append(mask)

        masks = torch.stack(masks)
        predictions, batch_idx = MaskedPredictionModule()(cancer_logits, masks)

        loss = torch.nn.L1Loss()(predictions, torch.zeros_like(predictions))

        return loss


def build_heatmap_loss(args):
    if args.loss == "needle_region_ce":
        return CancerDetectionValidRegionLoss()
    elif args.loss == "inv_l1":
        return InvolvementL1Loss()
    elif args.loss == "inv_l1_v2":
        return InvolvementL1Loss(prostate_penalty=False)
    elif args.loss == "inv_mse":
        return InvolvementMSELoss()
    elif args.loss == "mil_prop_bce":
        return CancerDetectionMILLoss()
    elif args.loss == "mil_prop_bce_l1_reg":
        return CancerDetectionMILLoss(
            base_loss=ProportionBCE(0.001), treat_gg1_as_benign=args.treat_gg1_as_benign
        )
    elif args.loss == "mil_prop_bce_entropy_reg":
        return CancerDetectionMILLoss(
            base_loss=ProportionBCE(entropy_penalty_lambda=0.01),
            treat_gg1_as_benign=args.get('treat_gg1_as_benign', False),
        )

    elif args.loss == "none":
        return None
    else:
        raise ValueError(f"Unknown loss function: {args.loss}")


def build_loss(args):

    losses = []
    weights=[]
    names=[]
    hmap_loss = build_heatmap_loss(args)

    # losses.append(SymmetricInfoNCELoss())
    if args.experiment_type != 'pnf':
        print("Correctly triggered")
        losses.append(TripletLoss(margin=1.0, p=2))
        names.append('triplet')
        weights.append(1)

    if hmap_loss is not None:
        losses.append(hmap_loss)
        weights.append(1)
        names.append('hmap')

    if args.add_image_clf:
        print(f"Adding image-level classification loss: {args.add_image_clf}")
        losses.append(ImageLevelClassificationLoss(mode=args.image_clf_mode))

    # if args.outside_prostate_penalty:
    #     print("Adding outside prostate penalty loss.")
    #     losses.append(OutsideProstatePenaltyLoss())

    return SumLoss(losses,weights, names)


def get_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--loss", default="needle_region_ce")
    parser.add_argument(
        "--outside_prostate_penalty",
        action="store_true",
        default=False,
        help="Whether to penalize the model for making predictions outside the prostate region.",
    )
    return parser
