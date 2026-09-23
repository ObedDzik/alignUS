import argparse
from dataclasses import dataclass, field
import json
from typing import Callable
import torch
from torch import nn
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import repeat, rearrange
import logging


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
        pos_weight: float = 1.0,
    ):
        super().__init__()
        self.l1_penalty_lambda = l1_penalty_lambda
        self.entropy_penalty_lambda = entropy_penalty_lambda
        self.pos_weight = pos_weight

    def forward(self, bag_of_logits, true_prop):
        probs = bag_of_logits.sigmoid()
        pred_prob = probs.mean()
        loss = -true_prop * pred_prob.log() - (1 - true_prop) * (1 - pred_prob).log()

        if true_prop > 0:
            loss = loss * self.pos_weight
        if self.l1_penalty_lambda:
            loss = loss + self.l1_penalty_lambda * probs.abs().sum()
        if self.entropy_penalty_lambda:
            entropy = -probs * probs.log() - (1 - probs) * (1 - probs).log()
            loss = loss + self.entropy_penalty_lambda * entropy.mean()

        return loss

class ImprovedProportionBCE(nn.Module):
    def __init__(
        self,
        l1_penalty_lambda: float | None = None,
        entropy_penalty_lambda: float | None = None,
        pos_weight: float = 4.0,  # Higher default
        focal_gamma: float = 2.0,  # Add focal focusing
    ):
        super().__init__()
        self.l1_penalty_lambda = l1_penalty_lambda
        self.entropy_penalty_lambda = entropy_penalty_lambda
        self.pos_weight = pos_weight
        self.focal_gamma = focal_gamma

    def forward(self, bag_of_logits, true_prop):
        probs = bag_of_logits.sigmoid()
        pred_prob = probs.mean()
        
        # Clamp for numerical stability
        pred_prob = torch.clamp(pred_prob, min=1e-7, max=1-1e-7)
        
        # Separate positive and negative terms
        pos_loss = -true_prop * torch.log(pred_prob)
        neg_loss = -(1 - true_prop) * torch.log(1 - pred_prob)
        
        # Optional: Focal weighting (focuses on hard examples)
        if self.focal_gamma > 0:
            pt = true_prop * pred_prob + (1 - true_prop) * (1 - pred_prob)
            focal_weight = (1 - pt) ** self.focal_gamma
            pos_loss = pos_loss * focal_weight
            neg_loss = neg_loss * focal_weight
        
        # Apply pos_weight ONLY to positive term
        loss = self.pos_weight * pos_loss + neg_loss
        
        # Regularization
        if self.l1_penalty_lambda:
            loss = loss + self.l1_penalty_lambda * probs.abs().sum()
        if self.entropy_penalty_lambda:
            entropy = -probs * torch.log(probs + 1e-7) - (1 - probs) * torch.log(1 - probs + 1e-7)
            loss = loss + self.entropy_penalty_lambda * entropy.mean()
        
        return loss

#the original code for all other use cases
# class CancerDetectionMILLoss(nn.Module):
#     def __init__(self, base_loss=ProportionBCE(), treat_gg1_as_benign=False):

#         super().__init__()
#         self.base_loss = base_loss
#         self.treat_gg1_as_benign = treat_gg1_as_benign

#     def forward(self, data):
#         cancer_logits = data["cancer_logits"]
#         batch_size = len(cancer_logits)
#         prostate_mask = data["prostate_mask"].to(cancer_logits.device)
#         needle_mask = data["needle_mask"].to(cancer_logits.device)
#         involvement = data["involvement"].to(cancer_logits.device)
#         grade_group = data["grade_group"].to(cancer_logits.device)

#         if self.treat_gg1_as_benign:
#             involvement[grade_group == 1] = 0.0

#         masks = []
#         for i in range(len(cancer_logits)):
#             mask = torch.ones(
#                 prostate_mask[i].shape, device=prostate_mask[i].device
#             ).bool()
#             mask &= prostate_mask[i] > 0.5
#             mask &= needle_mask[i] > 0.5
#             masks.append(mask)
#         masks = torch.stack(masks)
#         predictions, batch_idx = MaskedPredictionModule()(cancer_logits, masks)

#         loss = torch.tensor(0, device=cancer_logits.device)
#         for i in range(batch_size):
#             bag_i = predictions[batch_idx == i]
#             involvement_i = involvement[i]

#             loss = loss + self.base_loss(bag_i, involvement_i)

#         return loss

#For the distribution based needle
# class CancerDetectionMILLoss(nn.Module):
#     def __init__(self, base_loss=ProportionBCE(), treat_gg1_as_benign=False):
#         super().__init__()
#         self.base_loss = base_loss
#         self.treat_gg1_as_benign = treat_gg1_as_benign

#     def forward(self, data):
#         cancer_logits = data["cancer_logits"]
#         batch_size = len(cancer_logits)
#         prostate_mask = data["prostate_mask"].to(cancer_logits.device)
#         involvement = data["involvement"].to(cancer_logits.device)
#         grade_group = data["grade_group"].to(cancer_logits.device)

#         target_h, target_w = cancer_logits.shape[-2], cancer_logits.shape[-1]

#         needle_weight = data.get("needle_weight", None)
#         if needle_weight is not None:
#             needle_weight = needle_weight.to(cancer_logits.device)
#             if needle_weight.dim() == 3:
#                 needle_weight = needle_weight.unsqueeze(1)  # (B, 1, gh, gw) for interpolate
#             needle_weight = F.interpolate(
#                 needle_weight, size=(target_h, target_w), mode="bilinear", align_corners=False
#             ).squeeze(1)  # (B, H, W), upsampled to cancer_logits' resolution
#         else:
#             needle_weight = (data["needle_mask"].to(cancer_logits.device) > 0.5).float()
#             if needle_weight.dim() == 4:
#                 needle_weight = needle_weight[:, 0]  # (B, H, W)

#         if self.treat_gg1_as_benign:
#             involvement[grade_group == 1] = 0.0

#         prostate_bool = prostate_mask > 0.5
#         if prostate_bool.dim() == 4:
#             prostate_bool = prostate_bool[:, 0]

#         loss = torch.tensor(0.0, device=cancer_logits.device)
#         n_valid = 0
#         for i in range(batch_size):
#             w_i = (needle_weight[i] * prostate_bool[i].float()).flatten()
#             logits_i = cancer_logits[i].flatten()
#             involvement_i = involvement[i]

#             if w_i.sum() < 1e-6:
#                 continue

#             pred_prob_i = torch.sigmoid(logits_i)
#             pred_prob = (pred_prob_i * w_i).sum() / w_i.sum()

#             eps = 1e-6
#             pred_prob = pred_prob.clamp(min=eps, max=1 - eps)
#             loss_i = -involvement_i * pred_prob.log() - (1 - involvement_i) * (1 - pred_prob).log()
#             loss = loss + loss_i
#             n_valid += 1

#         if n_valid == 0:
#             return torch.tensor(0.0, device=cancer_logits.device, requires_grad=True)
#         return loss / n_valid


#This is for the posweight

class CancerDetectionMILLoss(nn.Module):
    def __init__(
        self,
        entropy_penalty_lambda: float = 0.01,
        l1_penalty_lambda: float | None = None,
        pos_weight: float = 8.0,
        treat_gg1_as_benign: bool = False,
        base_loss=ProportionBCE()
    ):
        super().__init__()
        self.entropy_penalty_lambda = entropy_penalty_lambda
        self.l1_penalty_lambda = l1_penalty_lambda
        self.pos_weight = pos_weight
        self.treat_gg1_as_benign = treat_gg1_as_benign

    def forward(self, data):
        cancer_logits = data["cancer_logits"]
        batch_size = len(cancer_logits)
        prostate_mask = data["prostate_mask"].to(cancer_logits.device)
        involvement = data["involvement"].to(cancer_logits.device)
        grade_group = data["grade_group"].to(cancer_logits.device)

        target_h, target_w = cancer_logits.shape[-2], cancer_logits.shape[-1]

        needle_weight = data.get("needle_weight", None)
        if needle_weight is not None:
            needle_weight = needle_weight.to(cancer_logits.device)
            if needle_weight.dim() == 3:
                needle_weight = needle_weight.unsqueeze(1)
            needle_weight = F.interpolate(
                needle_weight, size=(target_h, target_w), mode="bilinear", align_corners=False
            ).squeeze(1)
        else:
            needle_weight = (data["needle_mask"].to(cancer_logits.device) > 0.5).float()
            if needle_weight.dim() == 4:
                needle_weight = needle_weight[:, 0]

        if self.treat_gg1_as_benign:
            involvement[grade_group == 1] = 0.0

        prostate_bool = prostate_mask > 0.5
        if prostate_bool.dim() == 4:
            prostate_bool = prostate_bool[:, 0]

        loss = torch.tensor(0.0, device=cancer_logits.device)
        n_valid = 0
        for i in range(batch_size):
            w_i = (needle_weight[i] * prostate_bool[i].float()).flatten()
            logits_i = cancer_logits[i].flatten()
            involvement_i = involvement[i]

            if w_i.sum() < 1e-6:
                continue

            probs_i = torch.sigmoid(logits_i)
            pred_prob = (probs_i * w_i).sum() / w_i.sum()

            eps = 1e-6
            pred_prob_clamped = pred_prob.clamp(min=eps, max=1 - eps)
            loss_i = -involvement_i * pred_prob_clamped.log() - (1 - involvement_i) * (1 - pred_prob_clamped).log()

            if involvement_i > 0:
                loss_i = loss_i * self.pos_weight

            if self.l1_penalty_lambda:
                # weighted analogue of probs.abs().sum() -- only counts mass
                # actually inside the (soft) needle/prostate region, consistent
                # with everything else in this loss being weighted rather than
                # a hard-selected subset
                loss_i = loss_i + self.l1_penalty_lambda * (probs_i * w_i).abs().sum()

            if self.entropy_penalty_lambda:
                probs_c = probs_i.clamp(min=eps, max=1 - eps)
                entropy = -probs_c * probs_c.log() - (1 - probs_c) * (1 - probs_c).log()
                # weighted mean entropy over the region, not a plain mean over
                # every patch in the image -- matches ProportionBCE's intent
                # (regularizing predictions within the relevant bag) rather
                # than penalizing entropy everywhere including background
                weighted_entropy = (entropy * w_i).sum() / w_i.sum()
                loss_i = loss_i + self.entropy_penalty_lambda * weighted_entropy

            loss = loss + loss_i
            n_valid += 1

        if n_valid == 0:
            # Every sample in this batch had zero valid needle+prostate
            # pixels (w_i.sum() < 1e-6). A bare `torch.tensor(0.0,
            # requires_grad=True)` here is a disconnected leaf -- backward()
            # succeeds but touches no real model parameter, which makes
            # AMP's GradScaler.step() raise "No inf checks were recorded
            # for this optimizer" (unscale_ finds nothing to unscale).
            # Multiply a real graph-connected tensor by 0 instead, so the
            # loss is still exactly 0 but backward() correctly produces
            # (all-zero) gradients for the actual parameters.
            return cancer_logits.sum() * 0.0
        return loss / n_valid


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


class PooledProbabilityBCELoss(nn.Module):
    """BCE loss on a precomputed *scalar* per-core probability already sitting
    in `data` (default `average_needle_heatmap_value`) against `involvement`,
    same pos_weight convention as `CancerDetectionMILLoss` (weight applied to
    the positive-involvement term only, per-core, then mean-reduced over the
    batch). Added for fusion heads that produce a fused probability directly
    (e.g. `projects.primus_feedback.model.GatedCalibrationFusionHead`:
    `p_fused = (1-g)*p_base + g*p_vlm`) rather than a per-pixel heatmap.

    Deliberately does NOT pool from `data["cancer_logits"]` the way
    `CancerDetectionMILLoss` does: broadcasting an equivalent logit shift
    across every pixel and re-pooling via sigmoid-then-mean would not recover
    the same probability (mean-of-sigmoids != sigmoid-of-shifted-logits-mean),
    so this supervises the actual quantity such a head produces -- which is
    also the exact quantity `ProstNFoundEvaluator.aggregate_metrics` scores
    (it reads `average_needle_heatmap_value` straight off `data`, never
    re-pooling `cancer_logits` for the AUC metric).

    Added alongside, not modifying, `CancerDetectionMILLoss` -- per obed's
    direct instruction (2026-09-03, primus_feedback project) to keep that
    class, used by every other project's from-scratch heatmap training,
    untouched; any project wanting this loss opts in via `cfg.loss.loss =
    "pooled_prob_bce"` (see `build_heatmap_loss` below).
    """

    def __init__(self, prob_key: str = "average_needle_heatmap_value", pos_weight: float = 1.0):
        super().__init__()
        self.prob_key = prob_key
        self.pos_weight = pos_weight

    def forward(self, data: dict):
        prob = data[self.prob_key]
        involvement = data["involvement"].to(prob.device)
        eps = 1e-6
        prob_c = prob.clamp(min=eps, max=1 - eps)
        loss = -involvement * prob_c.log() - (1 - involvement) * (1 - prob_c).log()
        pos_w = torch.full_like(involvement, self.pos_weight)
        weight = torch.where(involvement > 0, pos_w, torch.ones_like(involvement))
        return (loss * weight).mean()


class DoNoHarmLoss(nn.Module):
    """Soft "do no harm" regularizer for a fusion pipeline that produces both
    a fused prediction and the underlying base detector's own prediction on
    the same core (`projects.primus_feedback`) -- added 2026-09-08 (obed's
    direct request: "worse case, the model should at least get the same
    value as the baseline"). This is a *training incentive*, not an
    architectural guarantee like a bounded convex-combination gate
    (`projects.primus_feedback.model.GatedCalibrationFusionHead`) -- an
    unconstrained head can still ignore it if the primary loss term
    disagrees strongly, but it directly penalizes exactly the failure mode
    seen across this project's `end_to_end`-family runs (fused BCE worse
    than the base detector's own BCE on the same core).

    `hinge = relu(bce(p_fused) - bce(p_base).detach())`, mean-reduced over
    the batch -- zero for every core where the fused prediction is already
    at least as good as the base detector's on that core's real label, so
    it never penalizes genuine improvement, only regression relative to
    doing nothing. `p_base` (and its own loss) is detached: gradient only
    ever flows to pull `p_fused` toward matching-or-beating `p_base`'s
    loss, never to make the (frozen or independently-supervised) base
    prediction look artificially worse.
    """

    def __init__(
        self, prob_key: str = "average_needle_heatmap_value",
        base_prob_key: str = "average_needle_heatmap_value_base", pos_weight: float = 1.0,
    ):
        super().__init__()
        self.prob_key = prob_key
        self.base_prob_key = base_prob_key
        self.pos_weight = pos_weight

    def _bce(self, prob, involvement):
        eps = 1e-6
        p = prob.clamp(min=eps, max=1 - eps)
        loss = -involvement * p.log() - (1 - involvement) * (1 - p).log()
        pos_w = torch.full_like(involvement, self.pos_weight)
        weight = torch.where(involvement > 0, pos_w, torch.ones_like(involvement))
        return loss * weight

    def forward(self, data: dict):
        involvement = data["involvement"].to(data[self.prob_key].device)
        fused_loss = self._bce(data[self.prob_key], involvement)
        base_loss = self._bce(data[self.base_prob_key].detach(), involvement).detach()
        return F.relu(fused_loss - base_loss).mean()


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
            labels = (data["grade_group"] > 1).long().to(logits.device)

        # Compute the binary cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss


class SumLoss(nn.Module):
    def __init__(self, losses: list[nn.Module], weights=None):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights if weights is not None else [1.0] * len(losses)

    def forward(self, data):
        loss = self.losses[0](data) * self.weights[0]
        for i in range(1, len(self.losses)):
            loss += self.losses[i](data) * self.weights[i]
        return loss


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
        return InvolvementL1Loss(**args.loss_kw)
    elif args.loss == "inv_l1_v2":
        return InvolvementL1Loss(prostate_penalty=False)
    elif args.loss == "inv_mse":
        return InvolvementMSELoss(**args.loss_kw)
    elif args.loss == "mil_prop_bce":
        # `CancerDetectionMILLoss.forward` computes its own inline BCE/pos_weight/
        # entropy/l1 math and never reads its `base_loss` argument (dead param,
        # kept only for call-site compatibility) -- pos_weight/entropy_penalty_lambda/
        # l1_penalty_lambda MUST be passed directly here, not nested inside a
        # `ProportionBCE(...)`/`ImprovedProportionBCE(...)` object, or they're
        # silently discarded and every variant below falls back to the class's own
        # hardcoded defaults (`pos_weight=8.0`, `entropy_penalty_lambda=0.01`)
        # regardless of `args.pos_weight`/cfg. Caught 2026-09-03 (obed) via
        # primus_feedback's cfg `pos_weight: 1.0` silently having no effect.
        return CancerDetectionMILLoss(
            pos_weight=args.pos_weight,
            entropy_penalty_lambda=None,
            treat_gg1_as_benign=args.treat_gg1_as_benign,
        )
    elif args.loss == "mil_prop_bce_l1_reg":
        return CancerDetectionMILLoss(
            l1_penalty_lambda=0.001,
            entropy_penalty_lambda=None,
            pos_weight=args.pos_weight,
            treat_gg1_as_benign=args.treat_gg1_as_benign,
        )
    elif args.loss == "mil_prop_bce_entropy_reg":
        return CancerDetectionMILLoss(
            entropy_penalty_lambda=0.01,
            pos_weight=args.pos_weight,
            treat_gg1_as_benign=args.treat_gg1_as_benign,
        )
    elif args.loss == "mil_prop_bce_entropy_reg_imp":
        # NOTE: `ImprovedProportionBCE`'s distinguishing logic (focal weighting)
        # is unreachable here for the same dead-`base_loss` reason as above --
        # this is currently identical to "mil_prop_bce_entropy_reg", not a
        # regression introduced by this fix (base_loss was already dead in the
        # class this replaced). Left as-is rather than papering over it; revisit
        # if a project actually depends on the focal-loss behavior.
        return CancerDetectionMILLoss(
            entropy_penalty_lambda=0.01,
            pos_weight=args.pos_weight,
            treat_gg1_as_benign=args.treat_gg1_as_benign,
        )
    elif args.loss == "pooled_prob_bce":
        return PooledProbabilityBCELoss(
            prob_key=args.loss_kw.get("prob_key", "average_needle_heatmap_value"),
            pos_weight=args.pos_weight,
        )
    elif args.loss == "none":
        return None
    else:
        raise ValueError(f"Unknown loss function: {args.loss}")


def build_loss(args):

    losses = []
    weights = []

    hmap_loss = build_heatmap_loss(args)

    if hmap_loss is not None:
        losses.append(hmap_loss)
        weights.append(1.0)

    if args.add_image_clf:
        print(f"Adding image-level classification loss: {args.add_image_clf}")
        losses.append(ImageLevelClassificationLoss(mode=args.image_clf_mode))
        weights.append(1.0)

    if args.outside_prostate_penalty:
        print("Adding outside prostate penalty loss.")
        losses.append(OutsideProstatePenaltyLoss())
        weights.append(1.0)

    # `args.base_aux_weight` (default 0.0 -- byte-identical for every
    # pre-2026-09-08 cfg): opt-in second BCE term directly on the *base*
    # detector's own pooled probability (`average_needle_heatmap_value_base`,
    # written by `projects.primus_feedback.model.PrimusFeedbackPipeline.forward`
    # for any fusion head that uses a bounded convex combination -- the
    # gradient the detector receives through `p_fused = (1-g)*p_base +
    # g*p_vlm` is scaled by `(1-g)`, diluted relative to a standalone
    # detector's own undiluted BCE gradient (`projects.primus_feedback.train_base`).
    # This term gives the detector a second, direct, undiluted gradient path
    # on top of whatever it gets through the fused prediction -- added
    # alongside, not modifying, the primary `PooledProbabilityBCELoss` term
    # above (which reads `average_needle_heatmap_value`, the fused value) --
    # see `projects/primus_feedback/CLAUDE.md`, "Full pipeline review" for
    # the diagnosis this was built to test.
    # `getattr` (not `args.get`) -- `args` is sometimes a plain `LossArgs`
    # dataclass (no `.get()` method) at other call sites across this repo,
    # not always an OmegaConf config; this field is new and won't exist on
    # either kind of `args` unless a caller explicitly sets it.
    base_aux_weight = getattr(args, "base_aux_weight", 0.0)
    if base_aux_weight > 0:
        print(f"Adding base-detector auxiliary loss, weight={base_aux_weight}")
        losses.append(PooledProbabilityBCELoss(
            prob_key="average_needle_heatmap_value_base", pos_weight=args.pos_weight,
        ))
        weights.append(base_aux_weight)

    # `args.do_no_harm_weight` (default 0.0, `getattr` for the same
    # plain-dataclass-vs-OmegaConf reason as `base_aux_weight` above):
    # opt-in hinge penalty (see `DoNoHarmLoss`) so the fusion head is
    # actively pushed to never do worse than the base detector's own
    # prediction on the same core, not just left to hope zero-init +
    # the primary loss term gets there on its own.
    do_no_harm_weight = getattr(args, "do_no_harm_weight", 0.0)
    if do_no_harm_weight > 0:
        print(f"Adding do-no-harm regularization, weight={do_no_harm_weight}")
        losses.append(DoNoHarmLoss(pos_weight=args.pos_weight))
        weights.append(do_no_harm_weight)

    return SumLoss(losses, weights)


@dataclass
class LossArgs:
    loss: str = "mil_prop_bce_entropy_reg"
    loss_kw: dict = field(default_factory=dict)
    add_image_clf: bool = False
    image_clf_mode: str = "cspca"
    outside_prostate_penalty: bool = False
    treat_gg1_as_benign: bool = False
    pos_weight: float = 1.0


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