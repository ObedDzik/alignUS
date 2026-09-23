from collections import defaultdict
import torch
import numpy as np
from medAI.layers.masked_prediction_module import get_bags_of_predictions
from medAI.utils.accumulators import DataFrameCollector
from medAI.metrics import calculate_binary_classification_metrics as calculate_metrics
from sklearn.metrics import roc_auc_score


def _auc_roc(predictions, labels):
    """Safe ROC AUC for single-class or NaN-pruned predictions."""
    nan_mask = np.isnan(predictions)
    predictions = predictions[~nan_mask]
    labels = labels[~nan_mask]
    return roc_auc_score(labels, predictions)


class ProstNFoundEvaluator:
    """Evaluator that accumulates predictions/labels and computes metrics once per iteration."""

    def __init__(
        self,
        log_images=False,
        log_images_every=10,
        include_patient_metrics=False,
        include_heatmap_cspca_metrics=True,
    ):
        self.iter = 0
        self.log_images = log_images
        self.log_images_every = log_images_every
        self.include_patient_metrics = include_patient_metrics
        self.include_heatmap_cspca_metrics = include_heatmap_cspca_metrics
        self.accumulator = DataFrameCollector()
        self.results_table = None

    @torch.no_grad()
    def __call__(self, data):
        """
        Process a batch of data and accumulate per-sample metrics.
        Does NOT compute classification metrics - only accumulates data.
        """
        step_metrics = {}

        # Determine batch size
        first_key = next(iter(data))
        batch_size = data[first_key].shape[0]

        # Bag-level predictions
        if "cancer_logits" in data:
            bags_of_logits = get_bags_of_predictions(
                data["cancer_logits"], data["prostate_mask"], data["needle_mask"]
            )
            bags_of_probs = [bag.sigmoid() for bag in bags_of_logits]
        elif "cancer_probs" in data:
            bags_of_probs = get_bags_of_predictions(
                data["cancer_probs"], data["prostate_mask"], data["needle_mask"]
            )
        else:
            raise ValueError("Data must contain 'cancer_logits' or 'cancer_probs'.")

        # Compute bag-level metrics per sample
        bag_entropies = []
        bag_topk_scores = []
        for probs in bags_of_probs:
            probs = probs.flatten()
            probs_sum = probs.sum()
            entropy = float(-(probs / probs_sum * (probs / probs_sum).log()).sum()) if probs_sum > 0 else 0.0
            N = len(probs)
            k = max(int(N * 0.5), 1)
            topk_score = float(torch.sort(probs, descending=True).values[:k].mean())
            bag_entropies.append(entropy)
            bag_topk_scores.append(topk_score)

        # Build the batch dict (all keys should have lists of length batch_size)
        tracked_data = {}
        
        # Copy scalar fields from data
        keys = [
            "center", "core_id", "patient_id", "loc", "grade", "age",
            "family_history", "psa", "pct_cancer", "grade_group",
            "average_needle_heatmap_value", "average_prostate_heatmap_value",
            "label", "involvement", "clinically_significant"
        ]
        
        for key in keys:
            if key in data:
                val = data[key]
                if isinstance(val, torch.Tensor):
                    tracked_data[key] = val.detach().cpu().tolist()
                else:
                    tracked_data[key] = val

        # Add bag-level metrics
        tracked_data["entropy"] = bag_entropies
        tracked_data["topk_score"] = bag_topk_scores

        # Optional image-level logits
        if data.get("image_level_classification_outputs"):
            logits = data["image_level_classification_outputs"][0].detach().cpu()
            tracked_data["image_level_cancer_logits"] = logits.softmax(-1)[:, 1].tolist()

        # Accumulate the entire batch at once
        self.accumulator(tracked_data)

        # Optional heatmap logging (NO metrics computation here)
        if self.log_images and (self.iter % self.log_images_every == 0):
            step_metrics["heatmap_example"] = show_heatmap_prediction(data)

        self.iter += 1
        # Return empty dict or just the figure - NO classification metrics
        return step_metrics



    def get_full_results_table(self):
        """Return the concatenated predictions/labels across all batches."""
        return self.accumulator.compute()

    def aggregate_metrics(self, results_table=None, desc="meta_val"):
        """
        Compute final metrics over the full accumulated DataFrame.

        Args:
            results_table: Optional DataFrame to use; otherwise use accumulator.
            desc: Prefix for metric names.

        Returns:
            metrics: dict of aggregated metrics
        """
        import pandas as pd

        # Use the provided table or compute from accumulator
        results_table = results_table or self.accumulator.compute()
        self.results_table = results_table.copy()

        metrics = {}

        # Ensure all columns are 1D scalars
        for col in results_table.columns:
            # Convert any tensors to floats
            results_table[col] = results_table[col].apply(
                lambda x: float(x.item()) if isinstance(x, torch.Tensor) else x
            )

        # --- Core predictions ---
        core_probs = results_table["average_needle_heatmap_value"].values
        core_labels = results_table["label"].values
        involvement = results_table["involvement"].values

        # Basic metrics
        metrics.update(calculate_metrics(core_probs, core_labels, log_images=self.log_images))
        metrics[f"{desc}/topk_probs_auroc"] = _auc_roc(results_table["topk_score"].values, core_labels)
        metrics[f"{desc}/avg_bag_entropy"] = results_table["entropy"].mean()

        # Prop prediction errors
        prop_err = np.abs(core_probs - involvement)
        metrics[f"{desc}/prop_pred_err"] = prop_err.mean()

        # Balanced prop error
        bal_err = (
            prop_err[core_labels == 0].mean() + prop_err[core_labels == 1].mean()
        ) / 2
        metrics[f"{desc}/bal_prop_pred_err"] = bal_err

        # --- High-involvement cores ---
        if getattr(self, "include_high_involvement_metrics", True):
            high_inv = involvement > 0.4
            benign = core_labels == 0
            keep = np.logical_or(high_inv, benign)
            if keep.sum() > 0:
                hi_probs = core_probs[keep]
                hi_labels = core_labels[keep]
                metrics_hi = calculate_metrics(hi_probs, hi_labels, log_images=self.log_images)
                for k, v in metrics_hi.items():
                    metrics[f"{desc}/{k}_high_involvement"] = v
                metrics[f"{desc}/topk_probs_auroc_high_inv"] = _auc_roc(results_table["topk_score"].values[keep], hi_labels)

        # --- Patient-level metrics ---
        if getattr(self, "include_patient_metrics", False):
            patient_preds = results_table.groupby("patient_id")["average_prostate_heatmap_value"].mean().values
            patient_labels = (results_table.groupby("patient_id")["clinically_significant"].sum() > 0).values
            metrics_patient = calculate_metrics(patient_preds, patient_labels, log_images=self.log_images)
            for k, v in metrics_patient.items():
                metrics[f"{desc}/{k}_patient"] = v

        # --- Image-level metrics ---
        if "image_level_cancer_logits" in results_table.columns:
            img_preds = results_table["image_level_cancer_logits"].values
            img_labels = core_labels
            metrics_img = calculate_metrics(img_preds, img_labels, log_images=self.log_images)
            for k, v in metrics_img.items():
                metrics[f"{desc}/{k}_image_level"] = v

            # Low vs high grade comparison
            img_labels_grade = (results_table["grade_group"].values > 2).astype(int)
            metrics_img_grade = calculate_metrics(img_preds, img_labels_grade, log_images=self.log_images)
            for k, v in metrics_img_grade.items():
                metrics[f"{desc}/{k}_image_level_cspca"] = v

        # --- Heatmap CSPCA metrics ---
        if getattr(self, "include_heatmap_cspca_metrics", True):
            heatmap_preds = results_table["average_needle_heatmap_value"].values
            heatmap_labels = (results_table["grade_group"].values > 2).astype(int)
            metrics_heatmap = calculate_metrics(heatmap_preds, heatmap_labels, log_images=self.log_images)
            for k, v in metrics_heatmap.items():
                metrics[f"{desc}/{k}_heatmap_cspca"] = v

        # Convert any numpy floats to Python floats
        for k, v in metrics.items():
            if isinstance(v, (np.floating, np.int64)):
                metrics[k] = float(v)

        return metrics


class ProstNFoundMetricsCalculator:
    """Compute metrics on fully accumulated evaluator results."""

    def __init__(
        self,
        log_images=False,
        include_patient_metrics=False,
        include_heatmap_cspca_metrics=True,
        include_high_involvement_metrics=True,
    ):
        self.log_images = log_images
        self.include_patient_metrics = include_patient_metrics
        self.include_heatmap_cspca_metrics = include_heatmap_cspca_metrics
        self.include_high_involvement_metrics = include_high_involvement_metrics

    def __call__(self, results_table):
        predictions = results_table["average_needle_heatmap_value"].values
        labels = results_table["label"].values
        involvement = results_table["involvement"].values

        metrics = {}

        # core metrics
        metrics.update(calculate_metrics(predictions, labels, log_images=self.log_images))
        metrics["topk_probs_auroc"] = _auc_roc(results_table["topk_score"].values, labels)
        metrics["avg_bag_entropy"] = results_table["entropy"].mean()
        metrics["prop_pred_err"] = np.abs(predictions - involvement).mean()
        results_table["prop_pred_err"] = np.abs(predictions - involvement)
        metrics["bal_prop_pred_err"] = (
            results_table.query("label==0")["prop_pred_err"].mean() +
            results_table.query("label==1")["prop_pred_err"].mean()
        ) / 2

        # high involvement metrics
        if self.include_high_involvement_metrics:
            high_inv_mask = (involvement > 0.4) | (labels == 0)
            if high_inv_mask.sum() > 0:
                metrics_hi = calculate_metrics(
                    predictions[high_inv_mask], labels[high_inv_mask], log_images=self.log_images
                )
                metrics.update({f"{k}_high_involvement": v for k, v in metrics_hi.items()})
                metrics["topk_probs_auroc_high_inv"] = _auc_roc(
                    results_table["topk_score"].values[high_inv_mask], labels[high_inv_mask]
                )

        # patient-level metrics
        if self.include_patient_metrics:
            patient_preds = results_table.groupby("patient_id")["average_prostate_heatmap_value"].mean().values
            patient_labels = (results_table.groupby("patient_id")["clinically_significant"].sum() > 0).values
            metrics_patient = calculate_metrics(patient_preds, patient_labels, log_images=self.log_images)
            metrics.update({f"{k}_patient": v for k, v in metrics_patient.items()})

        # image-level metrics
        if "image_level_cancer_logits" in results_table.columns:
            img_preds = results_table["image_level_cancer_logits"].values
            img_labels = labels
            metrics_img = calculate_metrics(img_preds, img_labels, log_images=self.log_images)
            metrics.update({f"{k}_image_level": v for k, v in metrics_img.items()})

            # CSPCA high-grade comparison
            grade_labels = (results_table["grade_group"].values > 2).astype(int)
            metrics_img_cspca = calculate_metrics(img_preds, grade_labels, log_images=self.log_images)
            metrics.update({f"{k}_image_level_cspca": v for k, v in metrics_img_cspca.items()})

        # heatmap CSPCA
        if self.include_heatmap_cspca_metrics:
            heatmap_preds = results_table["average_needle_heatmap_value"].values
            grade_labels = (results_table["grade_group"].values > 2).astype(int)
            metrics_hm = calculate_metrics(heatmap_preds, grade_labels, log_images=self.log_images)
            metrics.update({f"{k}_heatmap_cspca": v for k, v in metrics_hm.items()})

        # ensure all metrics are floats
        metrics = {k: float(v) if isinstance(v, np.floating) else v for k, v in metrics.items()}
        return metrics
