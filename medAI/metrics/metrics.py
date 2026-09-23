from typing import Literal
import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    f1_score,
)
import warnings
import logging


def dice_loss(mask_probs, target_mask):
    intersection = (mask_probs * target_mask).sum()
    union = mask_probs.sum() + target_mask.sum()
    return 1 - 2 * intersection / union


def dice_score(mask_probs, target_mask):
    mask_probs = mask_probs > 0.5
    intersection = (mask_probs * target_mask).sum()
    union = mask_probs.sum() + target_mask.sum()
    return 2 * intersection / union


# def calculate_binary_classification_metrics(
#     scores, labels, log_images=False, image_fmt: Literal["wandb", "plt"] = "plt"
# ):
#     """Calculate metrics for the cancer classification problem.

#     Args:
#         predictions (np.array or torch.Tensor) - A column vector of predicted probabilities for
#             cancer (1) or benign(0)
#         labels (np.array or torch.Tensor) - A column vector of true labels for cancer (1) or benign(0)
#         log_images (bool) - If True, log images of the histogram of predictions and the ROC curve to
#             wandb. Default is False.
#     """

#     if isinstance(scores, torch.Tensor):
#         scores = scores.cpu().numpy()
#     if isinstance(labels, torch.Tensor):
#         labels = labels.cpu().numpy()

#     if scores.ndim > 1 and scores.shape[1] == 2:
#         scores = scores[:, 1]

#     # augmentations can cause NaNs
#     nanvalues = np.isnan(scores)
#     scores = scores[~nanvalues]
#     labels = labels[~nanvalues]

#     metrics = {}

#     # core predictions
#     core_probs = scores
#     core_labels = labels
#     try:
#         metrics["auc"] = roc_auc_score(core_labels, core_probs)
#     except ValueError:
#         warnings.warn("ROC AUC score could not be calculated. Setting to 0.5")
#         metrics["auc"] = 0.5

#     try:
#         metrics["auprc"] = average_precision_score(core_labels, core_probs)
#     except:
#         warnings.warn("AUPRC score could not be calculated. Setting to 0.5")
#         metrics["auprc"] = 0.5

#     # find the sensitivity at fixed specificities
#     fpr, tpr, thresholds = roc_curve(core_labels, core_probs)
#     spe = 1 - fpr

#     for specificity in [0.20, 0.40, 0.60, 0.80]:
#         sensitivity = tpr[np.argmax(fpr > 1 - specificity)]
#         metrics[f"sens_at_{specificity*100:.0f}_spe"] = sensitivity

#     # choose the threshold that maximizes balanced accuracy
#     best_threshold = thresholds[np.argmax(tpr - fpr)]

#     try:
#         metrics["f1"] = f1_score(core_labels, core_probs > best_threshold)
#     except ValueError:
#         warnings.warn("F1 score could not be calculated. Setting to 0.5")
#         metrics["f1"] = 0.5

#     _best_threshold_idx = np.argmax(thresholds > best_threshold)
#     metrics["balanced_acc_best"] = (
#         fpr[_best_threshold_idx] + spe[_best_threshold_idx]
#     ) / 2

#     # 50pct_threshold_idx = np.argmax(thresholds > 0.5)
#     # metrics['balanced_acc'] = (fpr[_50pct_threshold_idx] + spe[_50pct_threshold_idx]) / 2
#     metrics["balanced_acc"] = balanced_accuracy_score(core_labels, (core_probs > 0.5))
#     metrics["spec"] = recall_score(1 - core_labels, (core_probs <= 0.5))
#     metrics["sens"] = recall_score(core_labels, (core_probs > 0.5))

    # print("nan:", np.isnan(core_probs).sum(), "inf:", np.isinf(core_probs).sum())
    # print("min/max:", np.nanmin(core_probs), np.nanmax(core_probs))

    # if log_images:
    #     fig = plt.figure()
    #     plt.hist(core_probs[core_labels == 0], bins=100, alpha=0.5, density=True)
    #     plt.hist(core_probs[core_labels == 1], bins=100, alpha=0.5, density=True)
    #     plt.legend(["Benign", "Cancer"])
    #     plt.xlabel(f"Probability of cancer")
    #     plt.ylabel("Density")
    #     plt.title(f"AUC: {metrics['auc']:.3f}")

    #     if image_fmt == "wandb":
    #         metrics["histogram"] = wandb.Image(
    #             plt, caption="Histogram of core predictions"
    #         )
    #         plt.close()
    #     else:
    #         metrics["histogram"] = fig

    #     fig = plt.figure()
    #     plt.plot(fpr, tpr)
    #     plt.xlabel("False positive rate")
    #     plt.ylabel("True positive rate")
    #     plt.title("ROC curve")

    #     if image_fmt == "wandb":
    #         metrics["roc_curve"] = wandb.Image(plt, caption="ROC curve")
    #         plt.close()
    #     else:
    #         metrics["roc_curve"] = fig

    # return metrics

    # if log_images:
    #     fig = plt.figure()
        
    #     neg_probs = core_probs[core_labels == 0]
    #     pos_probs = core_probs[core_labels == 1]
        
    #     # Guard against constant predictions (e.g. at start of training)
    #     bins = min(100, len(np.unique(core_probs)))
    #     if bins > 1:
    #         plt.hist(neg_probs, bins=bins, alpha=0.5, density=True)
    #         plt.hist(pos_probs, bins=bins, alpha=0.5, density=True)
    #     else:
    #         plt.text(0.5, 0.5, 'Predictions are constant', 
    #                 ha='center', va='center', transform=plt.gca().transAxes)
        
    #     plt.legend(["Benign", "Cancer"])
    #     plt.xlabel(f"Probability of cancer")
    #     plt.ylabel("Density")
    #     plt.title(f"AUC: {metrics['auc']:.3f}")
        
    #     if image_fmt == "wandb":
    #         metrics["histogram"] = wandb.Image(
    #             plt, caption="Histogram of core predictions"
    #         )
    #         plt.close()
    #     else:
    #         metrics["histogram"] = fig

    #     fig = plt.figure()
    #     plt.plot(fpr, tpr)
    #     plt.xlabel("False positive rate")
    #     plt.ylabel("True positive rate")
    #     plt.title("ROC curve")

    #     if image_fmt == "wandb":
    #         metrics["roc_curve"] = wandb.Image(plt, caption="ROC curve")
    #         plt.close()
    #     else:
    #         metrics["roc_curve"] = fig

    # return metrics

    # if log_images:
    #     fig = plt.figure()

    #     core_probs = np.asarray(core_probs)
    #     core_labels = np.asarray(core_labels)

    #     finite_mask = np.isfinite(core_probs)
    #     n_nonfinite = (~finite_mask).sum()
    #     if n_nonfinite > 0:
    #         logging.warning(f"{n_nonfinite} non-finite core_probs values found; excluding from histogram.")
    #     core_probs = core_probs[finite_mask]
    #     core_labels = core_labels[finite_mask]

    #     neg_probs = core_probs[core_labels == 0]
    #     pos_probs = core_probs[core_labels == 1]

    #     data_min = core_probs.min() if len(core_probs) > 0 else 0.0
    #     data_max = core_probs.max() if len(core_probs) > 0 else 1.0
    #     data_range = data_max - data_min
    #     n_unique = len(np.unique(core_probs))
    #     bins = min(100, n_unique)

    #     if bins > 1 and data_range > 1e-6:
    #         plt.hist(neg_probs, bins=bins, range=(data_min, data_max), alpha=0.5, density=True)
    #         plt.hist(pos_probs, bins=bins, range=(data_min, data_max), alpha=0.5, density=True)
    #     else:
    #         plt.text(0.5, 0.5, 'Predictions are constant or near-constant',
    #                 ha='center', va='center', transform=plt.gca().transAxes)

    #     plt.legend(["Benign", "Cancer"])
    #     plt.xlabel(f"Probability of cancer")
    #     plt.ylabel("Density")
    #     plt.title(f"AUC: {metrics['auc']:.3f}")

    #     if image_fmt == "wandb":
    #         metrics["histogram"] = wandb.Image(plt, caption="Histogram of core predictions")
    #         plt.close()
    #     else:
    #         metrics["histogram"] = fig

    # return metrics



def calculate_binary_classification_metrics(
    scores, labels, log_images=False, image_fmt: Literal["wandb", "plt"] = "plt"
):
    """Calculate metrics for the cancer classification problem.

    Args:
        scores (np.array or torch.Tensor) - A column vector of predicted probabilities for
            cancer (1) or benign(0)
        labels (np.array or torch.Tensor) - A column vector of true labels for cancer (1) or benign(0)
        log_images (bool) - If True, log images of the histogram of predictions and the ROC curve to
            wandb. Default is False.
    """

    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    scores = np.asarray(scores)
    labels = np.asarray(labels)

    if scores.ndim > 1 and scores.shape[1] == 2:
        scores = scores[:, 1]

    # augmentations can cause NaNs
    nanvalues = np.isnan(scores)
    scores = scores[~nanvalues]
    labels = labels[~nanvalues]

    metrics = {}

    core_probs = scores
    core_labels = labels

    # single-class labels make every threshold-based metric degenerate, and
    # roc_curve/roc_auc_score raise -- bail out explicitly rather than letting
    # each metric fail independently below
    n_pos = int((core_labels == 1).sum())
    n_neg = int((core_labels == 0).sum())
    metrics["n"] = int(len(core_labels))
    metrics["n_pos"] = n_pos
    if n_pos == 0 or n_neg == 0:
        warnings.warn(
            f"Single-class labels (n_pos={n_pos}, n_neg={n_neg}); "
            "threshold-based metrics are undefined. Returning neutral values."
        )
        for key in ["auc", "auprc", "balanced_acc", "balanced_acc_best", "f1", "spec", "sens"]:
            metrics[key] = 0.5
        for specificity in [0.20, 0.40, 0.60, 0.80]:
            metrics[f"sens_at_{specificity*100:.0f}_spe"] = 0.5
        metrics["best_threshold"] = float("nan")
        return metrics

    try:
        metrics["auc"] = roc_auc_score(core_labels, core_probs)
    except ValueError:
        warnings.warn("ROC AUC score could not be calculated. Setting to 0.5")
        metrics["auc"] = 0.5

    try:
        metrics["auprc"] = average_precision_score(core_labels, core_probs)
    except Exception:
        warnings.warn("AUPRC score could not be calculated. Setting to 0.5")
        metrics["auprc"] = 0.5

    fpr, tpr, thresholds = roc_curve(core_labels, core_probs)
    spe = 1 - fpr

    # sensitivity at fixed specificities
    for specificity in [0.20, 0.40, 0.60, 0.80]:
        sensitivity = tpr[np.argmax(fpr > 1 - specificity)]
        metrics[f"sens_at_{specificity*100:.0f}_spe"] = sensitivity

    # threshold maximizing Youden's J -- keep the INDEX, don't recover it by
    # searching `thresholds` for the value: roc_curve returns thresholds in
    # descending order, so `np.argmax(thresholds > best_threshold)` returns 0
    # (the trivial all-negative operating point), not the optimum.
    best_idx = int(np.argmax(tpr - fpr))
    best_threshold = thresholds[best_idx]
    metrics["best_threshold"] = float(best_threshold)

    try:
        metrics["f1"] = f1_score(core_labels, core_probs > best_threshold)
    except ValueError:
        warnings.warn("F1 score could not be calculated. Setting to 0.5")
        metrics["f1"] = 0.5

    # balanced accuracy at the Youden-optimal threshold.
    # (sensitivity + specificity) / 2 == (tpr + spe) / 2.
    # The previous version used (fpr + spe)/2 == (fpr + 1 - fpr)/2 == 0.5
    # identically, for every input.
    metrics["balanced_acc_best"] = (tpr[best_idx] + spe[best_idx]) / 2

    # fixed-0.5-threshold variants -- only meaningful when `scores` are
    # calibrated probabilities. For raw logits, or scores compressed into a
    # narrow band (e.g. mean-aggregated over many cores), these can collapse
    # to a single predicted class; prefer balanced_acc_best there.
    metrics["balanced_acc"] = balanced_accuracy_score(core_labels, (core_probs > 0.5))
    metrics["spec"] = recall_score(1 - core_labels, (core_probs <= 0.5))
    metrics["sens"] = recall_score(core_labels, (core_probs > 0.5))


    if log_images:
        fig = plt.figure()

        core_probs = np.asarray(core_probs)
        core_labels = np.asarray(core_labels)

        finite_mask = np.isfinite(core_probs)
        n_nonfinite = (~finite_mask).sum()
        if n_nonfinite > 0:
            logging.warning(f"{n_nonfinite} non-finite core_probs values found; excluding from histogram.")
        core_probs = core_probs[finite_mask]
        core_labels = core_labels[finite_mask]

        neg_probs = core_probs[core_labels == 0]
        pos_probs = core_probs[core_labels == 1]

        data_min = core_probs.min() if len(core_probs) > 0 else 0.0
        data_max = core_probs.max() if len(core_probs) > 0 else 1.0
        data_range = data_max - data_min
        n_unique = len(np.unique(core_probs))
        bins = min(100, n_unique)

        if bins > 1 and data_range > 1e-6:
            plt.hist(neg_probs, bins=bins, range=(data_min, data_max), alpha=0.5, density=True)
            plt.hist(pos_probs, bins=bins, range=(data_min, data_max), alpha=0.5, density=True)
        else:
            plt.text(0.5, 0.5, 'Predictions are constant or near-constant',
                    ha='center', va='center', transform=plt.gca().transAxes)

        plt.legend(["Benign", "Cancer"])
        plt.xlabel(f"Probability of cancer")
        plt.ylabel("Density")
        plt.title(f"AUC: {metrics['auc']:.3f}")

        if image_fmt == "wandb":
            metrics["histogram"] = wandb.Image(plt, caption="Histogram of core predictions")
            plt.close()
        else:
            metrics["histogram"] = fig

    return metrics



def calculate_aggregated_metrics(
    scores,
    labels,
    item_ids,
    metric_fn=calculate_binary_classification_metrics,
    **kwargs,
):
    """Calculate aggregated metrics."""

    item_ids = np.array(item_ids)
    aggregated_scores = []
    aggregated_labels = []
    for core_id in np.unique(item_ids):
        aggregated_scores.append(scores[item_ids == core_id].mean(0))
        aggregated_labels.append(labels[item_ids == core_id][0])

    aggregated_scores = np.array(aggregated_scores)
    aggregated_labels = np.array(aggregated_labels)

    return metric_fn(aggregated_scores, aggregated_labels, **kwargs)
