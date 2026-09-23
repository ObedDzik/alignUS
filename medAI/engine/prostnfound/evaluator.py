from collections import defaultdict
import torch
from medAI.layers.masked_prediction_module import get_bags_of_predictions
from medAI.utils.accumulators import DataFrameCollector
import numpy as np
from sklearn.metrics import roc_auc_score
from torchvision.transforms import v2 as T
from matplotlib import pyplot as plt
from PIL import Image
from medAI.metrics import calculate_binary_classification_metrics as calculate_metrics
from sklearn.metrics import roc_auc_score
import matplotlib.ticker as mticker
from scipy.ndimage import gaussian_filter


def _auc_roc(predictions, labels):
    nanvalues = np.isnan(predictions)
    predictions = predictions[~nanvalues]
    labels = labels[~nanvalues]
    return roc_auc_score(labels, predictions)

def _topk_mean(s, k):
    """Mean of the k highest values in a group. Falls back to the mean of
    all available values when a patient has fewer than k cores. Uses numpy
    rather than Series.nlargest, which fails on object-dtype columns."""
    arr = np.asarray(s, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    k = min(k, arr.size)
    return float(np.sort(arr)[-k:].mean())



@torch.no_grad()
def show_heatmap_prediction_publication(data, dpi=100, save_path=None, save_dpi=300):
    plt.close("all")

    if "cancer_logits" in data:
        pred = torch.sigmoid(data["cancer_logits"].cpu())
    elif "cancer_probs" in data:
        pred = data["cancer_probs"].cpu()
    else:
        raise ValueError("Missing prediction key")

    image = data["bmode"].cpu()
    if image.ndim == 5:
        image = image[:, 0] 
    prostate_mask = data["prostate_mask"].cpu()
    needle_mask = data["needle_mask"].cpu()
    label = data["label"]

    B, C, H, W = image.shape

    resize_img = T.Resize((H, W), interpolation=Image.BICUBIC, antialias=True)
    resize_mask = T.Resize((H, W), interpolation=Image.NEAREST)

    image = resize_img(image)
    prostate_mask = resize_mask(prostate_mask)
    needle_mask = resize_mask(needle_mask)
    pred = resize_img(pred)

    img = image[0].permute(1, 2, 0).numpy()
    img = np.clip(img, 0.0, 1.0)
    hm = pred[0, 0].numpy()
    prostate = prostate_mask[0, 0].numpy()
    needle = needle_mask[0, 0].numpy()

    prostate = (prostate > 0.5).astype(np.float32)
    needle = (needle > 0.5).astype(np.float32)

    hm_plot = hm.copy()
    if prostate.sum() > 0:
        hm_plot[prostate == 0] = np.nan

    vmin, vmax = 0, 1

    # Render at low DPI — fast for display/iteration
    fig, ax = plt.subplots(1, 2, figsize=(9, 4), dpi=dpi)
    for a in ax:
        a.axis("off")

    ax[0].imshow(img, cmap="gray", origin="lower")
    ax[0].contour(prostate, levels=[0.5], colors="white", linewidths=0.8)
    ax[0].contour(needle, levels=[0.5], colors="white", linewidths=0.8)

    ax[1].imshow(img, cmap="gray", origin="lower")
    heat = ax[1].imshow(
        hm_plot,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        alpha=0.95,
        interpolation="bicubic",  # looks fine even at 100 DPI
        origin="lower"
    )
    ax[1].contour(prostate, levels=[0.5], colors="black", linewidths=0.6)
    ax[1].contour(needle, levels=[0.5], colors="white", linewidths=0.6)

    cbar = fig.colorbar(heat, ax=ax[1], fraction=0.046, pad=0.04)
    cbar.set_label("Model activation", fontsize=9)
    cbar.outline.set_linewidth(0.5)

    fig.suptitle(
        f"Label: {'Cancer' if label[0].item() else 'Benign'} | "
        f"Involvement: {data['involvement'][0].item():.2f} | "
        f"Grade: {data['grade_group'][0]}",
        fontsize=11
    )
    plt.tight_layout()

    # Only pay the 300 DPI cost when you actually need to save
    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight")

    return fig



@torch.no_grad()
def show_heatmap_prediction(data):

    plt.close("all")
    plt.figure()

    if "cancer_logits" in data:
        logits = data["cancer_logits"].cpu()
        pred = logits.sigmoid()
    elif "cancer_probs" in data:
        pred = data["cancer_probs"].cpu()
    else:
        raise ValueError()

    needle_mask = data["needle_mask"]
    prostate_mask = data["prostate_mask"]
    # print(torch.unique(prostate_mask))
    image = data["bmode"]
    label = data["label"]

    if image.ndim == 5:
        image = image[:, 0] 

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    [ax.set_axis_off() for ax in ax.flatten()]
    kwargs = dict(vmin=0, vmax=1)

    B, C, H, W = image.shape
    image = T.Resize(
        (H, W), interpolation=Image.Resampling.BICUBIC, antialias=True
    )(image)
    needle_mask = T.Resize((H, W), interpolation=Image.Resampling.NEAREST)(
        needle_mask
    )
    prostate_mask = T.Resize((H, W), interpolation=Image.Resampling.NEAREST)(
        prostate_mask
    )
    pred = T.Resize((H, W), interpolation=Image.Resampling.NEAREST)(pred)
    # pred = T.Resize((H, W), interpolation=Image.Resampling.BICUBIC, antialias=True)(pred)

    # image and contours
    ax[0].imshow(image[0].permute(1, 2, 0), **kwargs)

    ax[0].contour(prostate_mask[0, 0], **kwargs)
    ax[0].contour(needle_mask[0, 0], **kwargs)

    # prediction
    ax[1].imshow(pred[0, 0], **kwargs)
    ax[1].contour(needle_mask[0, 0], **kwargs)
    ax[1].contour(prostate_mask[0, 0], **kwargs)

    fig.suptitle(
        f"Ground truth label: Cancer {label[0].item() == 1}; Inv {data['involvement'][0].item():.2f}; Grade group {data['grade_group'][0]}"
    )

    return fig


class ProstNFoundEvaluator:
    def __init__(
        self,
        log_images=False,
        log_images_every=10,
        include_patient_metrics=True,
        include_heatmap_cspca_metrics=True,
    ):
        self.iter = 0
        self.log_images = log_images
        self.log_images_every = log_images_every
        self.include_patient_metrics = include_patient_metrics
        self.accumulator = DataFrameCollector()
        self._heatmap_fig = None
        self.include_heatmap_cspca_metrics = include_heatmap_cspca_metrics
        self.results_table = None

    @torch.no_grad()
    def __call__(self, data):
        step_metrics = {}

        if "cancer_logits" in data:
            bags_of_logits = get_bags_of_predictions(
                data["cancer_logits"], data["prostate_mask"], data["needle_mask"]
            )
            bags_of_probs = [bag.sigmoid() for bag in bags_of_logits]
        elif "cancer_probs" in data:
            bags_of_probs = get_bags_of_predictions(
                data["cancer_probs"], data["prostate_mask"], data["needle_mask"]
            )

        bag_level_info = defaultdict(list)

        for probs in bags_of_probs:
            # entropy
            normalized_probs = probs / probs.sum()
            entropy = -(normalized_probs * normalized_probs.log()).sum()
            bag_level_info["entropy"].append(entropy.item())

            # topk score
            N = len(probs)
            k = int(N * 0.5)

            # below two were commented out to run the inference.py
            # topk_score = torch.sort(probs, descending=True).values[:k].mean()
            # bag_level_info["topk_score"].append(topk_score.item())

        tracked_data = {}
        keys = [
            "center",
            "core_id",
            "patient_id",
            "loc",
            "grade",
            "age",
            "family_history",
            "psa",
            "pct_cancer",
            "grade_group",
            "average_needle_heatmap_value",
            "average_prostate_heatmap_value",
            "label",
            "involvement",
            "clinically_significant",
        ]
        for key in keys:
            if key in data:
                tracked_data[key] = data[key]
        tracked_data.update(bag_level_info)

        if data.get("image_level_classification_outputs"):
            tracked_data["image_level_cancer_logits"] = (
                data["image_level_classification_outputs"][0]
                .detach()
                .cpu()
                .softmax(-1)[:, 1]
            )

        self.accumulator(tracked_data)

        if self.log_images and (self.iter % self.log_images_every == 0):
            figure = show_heatmap_prediction(data)
            step_metrics["heatmap_example"] = figure

        self.iter += 1
        return step_metrics

    def get_full_results_table(self): 
        return self.accumulator.compute()

    def aggregate_metrics(self, results_table=None):
        
        results_table = results_table or self.accumulator.compute()
        self.results_table = results_table

        print("columns:", list(results_table.columns))
        print("n rows:", len(results_table))
        print("n unique patients:", results_table["patient_id"].nunique())
        print("grade_group value counts:\n", results_table["grade_group"].value_counts(dropna=False))

        grouped = results_table.groupby("patient_id")
        for task, thr in [("pca", 1), ("cspca", 2)]:
            pl = (grouped["grade_group"].max() >= thr).astype(int).values
            print(f"{task}: n_patients={len(pl)}, n_pos={pl.sum()}, prevalence={pl.mean():.2f}")

        for col in ["average_needle_heatmap_value", "image_level_cancer_logits"]:
            pmax = grouped[col].max().values
            pmean = grouped[col].mean().values
            print(f"{col}: max-agg std={pmax.std():.4f} range=[{pmax.min():.3f},{pmax.max():.3f}] | "
                f"mean-agg std={pmean.std():.4f} range=[{pmean.min():.3f},{pmean.max():.3f}]")


        return ProstNFoundMetricsCalculator(
            log_images=self.log_images,
            include_patient_metrics=self.include_patient_metrics,
            include_heatmap_cspca_metrics=self.include_heatmap_cspca_metrics,
        )(results_table)

        # core predictions
        predictions = results_table.average_needle_heatmap_value.values
        labels = results_table.label.values
        involvement = results_table.involvement.values

        core_probs = predictions
        core_labels = labels

        metrics = {}
        metrics_ = calculate_metrics(predictions, labels, log_images=self.log_images)
        metrics.update(metrics_)

        # below two were commented out to run the inference.py
        # metrics["topk_probs_auroc"] = _auc_roc(results_table.topk_score, labels)
        # metrics["avg_bag_entropy"] = results_table["entropy"].mean()

        # prop pred err
        metrics["prop_pred_err"] = np.abs(
            results_table["average_needle_heatmap_value"].values
            - results_table["involvement"]
        ).mean()

        # balanced prop pred err
        results_table["prop_pred_err"] = (
            results_table["average_needle_heatmap_value"] - results_table["involvement"]
        ).abs()
        metrics["bal_prop_pred_err"] = (
            results_table.query("label == 0")["prop_pred_err"].mean()
            + results_table.query("label == 1")["prop_pred_err"].mean()
        ) / 2

        # high involvement core predictions
        benign = core_labels == 0

        # high-involvement core predictions
        high_involvement = involvement > 0.4
        keep_high = np.logical_or(high_involvement, benign)
        if keep_high.sum() > 0:
            probs_high = core_probs[keep_high]
            labels_high = core_labels[keep_high]
            metrics_ = calculate_metrics(probs_high, labels_high, log_images=self.log_images)
            metrics.update({
                f"{metric}_high_involvement": value
                for metric, value in metrics_.items()
            })

        # low-involvement core predictions
        low_involvement = involvement <= 0.4
        keep_low = np.logical_or(low_involvement, benign)
        if keep_low.sum() > 0:
            probs_low = core_probs[keep_low]
            labels_low = core_labels[keep_low]
            metrics_ = calculate_metrics(probs_low, labels_low, log_images=self.log_images)
            metrics.update({
                f"{metric}_low_involvement": value
                for metric, value in metrics_.items()
            })

            # below were commented out to run the inference.py
            # metrics["topk_probs_auroc_high_inv"] = _auc_roc(
            #     results_table.topk_score.values[keep], core_labels
            # )

        # patient predictions
        if self.include_patient_metrics:
            predictions = (
                results_table.groupby("patient_id")
                .average_prostate_heatmap_value.mean()
                .values
            )
            labels = (
                results_table.groupby("patient_id").clinically_significant.sum() > 0
            ).values
            metrics_ = calculate_metrics(
                predictions, labels, log_images=self.log_images
            )
            metrics.update(
                {f"{metric}_patient": value for metric, value in metrics_.items()}
            )

        if "image_level_cancer_logits" in results_table.columns:
            image_level_predictions = results_table.image_level_cancer_logits.values
            image_level_labels = results_table.label.values
            metrics_ = calculate_metrics(
                image_level_predictions, image_level_labels, log_images=self.log_images
            )
            metrics.update(
                {f"{metric}_image_level": value for metric, value in metrics_.items()}
            )

            image_level_labels = (results_table.grade_group.values > 2).astype(int)
            metrics_low_vs_high = metrics_ = calculate_metrics(
                image_level_predictions, image_level_labels, log_images=self.log_images
            )
            metrics.update(
                {
                    f"{metric}_image_level_cspca": value
                    for metric, value in metrics_low_vs_high.items()
                }
            )

        if self.include_heatmap_cspca_metrics:
            heatmap_predictions = results_table["average_needle_heatmap_value"]
            image_level_labels = (results_table.grade_group.values > 2).astype(int)
            metrics_ = calculate_metrics(
                heatmap_predictions, image_level_labels, log_images=self.log_images
            )
            metrics.update(
                {
                    f"{metric}_heatmap_cspca": value
                    for metric, value in metrics_.items()
                }
            )

        return metrics

    
# class ProstNFoundMetricsCalculator:
#     def __init__(
#         self,
#         log_images=False,
#         include_patient_metrics=True,
#         include_heatmap_cspca_metrics=True,
#         include_high_involvement_metrics=True,
#         include_low_involvement_metrics=True,
#     ):
#         self.log_images = log_images
#         self.include_patient_metrics = include_patient_metrics
#         self.include_heatmap_cspca_metrics = include_heatmap_cspca_metrics
#         self.include_high_involvement_metrics = include_high_involvement_metrics
#         self.include_low_involvement_metrics = include_low_involvement_metrics

#     def __call__(self, results_table):

#         predictions = results_table.average_needle_heatmap_value.values
#         predictions = results_table.average_needle_heatmap_value.values
#         labels = results_table.label.values
#         involvement = results_table.involvement.values

#         core_probs = predictions
#         core_labels = labels

#         metrics = {}
#         metrics_ = calculate_metrics(predictions, labels, log_images=self.log_images)
#         metrics.update(metrics_)

#         # below two were commented out to run the inference.py
#         # metrics["topk_probs_auroc"] = _auc_roc(results_table.topk_score, labels)
#         # metrics["avg_bag_entropy"] = results_table["entropy"].mean()

#         # prop pred err
#         metrics["prop_pred_err"] = np.abs(
#             results_table["average_needle_heatmap_value"].values
#             - results_table["involvement"]
#         ).mean()

#         # balanced prop pred err
#         results_table["prop_pred_err"] = (
#             results_table["average_needle_heatmap_value"] - results_table["involvement"]
#         ).abs()
#         metrics["bal_prop_pred_err"] = (
#             results_table.query("label == 0")["prop_pred_err"].mean()
#             + results_table.query("label == 1")["prop_pred_err"].mean()
#         ) / 2

#         print(results_table["involvement"].describe())
#         print((results_table["involvement"] <= 0.4).sum(), "cores with involvement <= 0.4")
#         print((results_table["involvement"] > 0.4).sum(), "cores with involvement > 0.4")

#         benign = core_labels == 0
#         # high-involvement core predictions
#         if self.include_high_involvement_metrics:
#             high_involvement = involvement > 0.4
#             keep_high = np.logical_or(high_involvement, benign)
#             if keep_high.sum() > 0:
#                 probs_high = core_probs[keep_high]
#                 labels_high = core_labels[keep_high]
#                 metrics_ = calculate_metrics(probs_high, labels_high, log_images=self.log_images)
#                 metrics.update({
#                     f"{metric}_high_involvement": value
#                     for metric, value in metrics_.items()
#                 })

#         # low-involvement core predictions
#         if self.include_low_involvement_metrics:
#             low_involvement = involvement <= 0.4
#             keep_low = np.logical_or(low_involvement, benign)
#             if keep_low.sum() > 0:
#                 probs_low = core_probs[keep_low]
#                 labels_low = core_labels[keep_low]
#                 metrics_ = calculate_metrics(probs_low, labels_low, log_images=self.log_images)
#                 metrics.update({
#                     f"{metric}_low_involvement": value
#                     for metric, value in metrics_.items()
#                 })

#                 # below were commented out to run the inference.py
#                 # metrics["topk_probs_auroc_high_inv"] = _auc_roc(
#                 #     results_table.topk_score.values[keep], core_labels
#                 # )

#         # patient predictions
#         if self.include_patient_metrics:
#             predictions = (
#                 results_table.groupby("patient_id")
#                 .average_prostate_heatmap_value.mean()
#                 .values
#             )
#             labels = (
#                 results_table.groupby("patient_id").clinically_significant.sum() > 0
#             ).values
#             metrics_ = calculate_metrics(
#                 predictions, labels, log_images=self.log_images
#             )
#             metrics.update(
#                 {f"{metric}_patient": value for metric, value in metrics_.items()}
#             )

#         if "image_level_cancer_logits" in results_table.columns:
#             image_level_predictions = results_table.image_level_cancer_logits.values
#             image_level_labels = results_table.label.values
#             metrics_ = calculate_metrics(
#                 image_level_predictions, image_level_labels, log_images=self.log_images
#             )
#             metrics.update(
#                 {f"{metric}_image_level": value for metric, value in metrics_.items()}
#             )

#             image_level_labels = (results_table.grade_group.values >= 2).astype(int)
#             metrics_low_vs_high = metrics_ = calculate_metrics(
#                 image_level_predictions, image_level_labels, log_images=self.log_images
#             )
#             metrics.update(
#                 {
#                     f"{metric}_image_level_cspca": value
#                     for metric, value in metrics_low_vs_high.items()
#                 }
#             )

#         if self.include_heatmap_cspca_metrics:
#             heatmap_predictions = results_table["average_needle_heatmap_value"]
#             image_level_labels = (results_table.grade_group.values >= 2).astype(int)
#             metrics_ = calculate_metrics(
#                 heatmap_predictions, image_level_labels, log_images=self.log_images
#             )
#             metrics.update(
#                 {
#                     f"{metric}_heatmap_cspca": value
#                     for metric, value in metrics_.items()
#                 }
#             )

#         # convert to float 
#         for key in metrics:
#             if isinstance(metrics[key], np.floating):
#                 metrics[key] = float(metrics[key])

#         return metrics



class ProstNFoundMetricsCalculator:
    def __init__(
        self,
        log_images=False,
        include_patient_metrics=True,
        include_heatmap_cspca_metrics=True,
        include_high_involvement_metrics=True,
        include_low_involvement_metrics=True,
        involvement_threshold=0.4,
    ):
        self.log_images = log_images
        self.include_patient_metrics = include_patient_metrics
        self.include_heatmap_cspca_metrics = include_heatmap_cspca_metrics
        self.include_high_involvement_metrics = include_high_involvement_metrics
        self.include_low_involvement_metrics = include_low_involvement_metrics
        self.involvement_threshold = involvement_threshold

    def _pca_cspca_stratified(self, metrics, predictions, grade_group, involvement, prefix):
        """Computes PCa and csPCa AUROC/sensitivity, overall and stratified
        by involvement, for a given prediction source (heatmap or
        image-level). `prefix` names the metric family, e.g. 'heatmap' or
        'image_level'."""
        thresh = self.involvement_threshold

        for task, task_labels in [
            ("pca", (grade_group >= 1).astype(int)),
            ("cspca", (grade_group >= 2).astype(int)),
        ]:
            benign_task = task_labels == 0

            # overall
            try:
                metrics_ = calculate_metrics(predictions, task_labels, log_images=self.log_images)
                metrics.update({
                    f"{metric}_{prefix}_{task}": value
                    for metric, value in metrics_.items()
                })
            except ValueError:
                pass

            if self.include_high_involvement_metrics:
                keep_high = np.logical_or(involvement > thresh, benign_task)
                if keep_high.sum() > 0 and task_labels[keep_high].sum() > 0:
                    try:
                        metrics_ = calculate_metrics(
                            predictions[keep_high], task_labels[keep_high],
                            log_images=self.log_images,
                        )
                        metrics.update({
                            f"{metric}_{prefix}_{task}_high_involvement": value
                            for metric, value in metrics_.items()
                        })
                    except ValueError:
                        pass

            if self.include_low_involvement_metrics:
                keep_low = np.logical_or(involvement <= thresh, benign_task)
                if keep_low.sum() > 0 and task_labels[keep_low].sum() > 0:
                    try:
                        metrics_ = calculate_metrics(
                            predictions[keep_low], task_labels[keep_low],
                            log_images=self.log_images,
                        )
                        metrics.update({
                            f"{metric}_{prefix}_{task}_low_involvement": value
                            for metric, value in metrics_.items()
                        })
                    except ValueError:
                        pass

    def __call__(self, results_table):

        predictions = results_table.average_needle_heatmap_value.values
        labels = results_table.label.values
        involvement = results_table.involvement.values
        grade_group = results_table.grade_group.values

        core_probs = predictions
        core_labels = labels

        metrics = {}
        metrics_ = calculate_metrics(predictions, labels, log_images=self.log_images)
        metrics.update(metrics_)

        metrics["prop_pred_err"] = np.abs(
            results_table["average_needle_heatmap_value"].values
            - results_table["involvement"]
        ).mean()

        results_table["prop_pred_err"] = (
            results_table["average_needle_heatmap_value"] - results_table["involvement"]
        ).abs()
        metrics["bal_prop_pred_err"] = (
            results_table.query("label == 0")["prop_pred_err"].mean()
            + results_table.query("label == 1")["prop_pred_err"].mean()
        ) / 2

        for col in ["average_needle_heatmap_value", "image_level_cancer_logits"]:
            print(col, results_table[col].dtype, type(results_table[col].iloc[0]), results_table[col].iloc[0])

        benign = core_labels == 0
        # high-involvement core predictions (existing `label`-based split, unchanged)
        if self.include_high_involvement_metrics:
            high_involvement = involvement > self.involvement_threshold
            keep_high = np.logical_or(high_involvement, benign)
            if keep_high.sum() > 0:
                probs_high = core_probs[keep_high]
                labels_high = core_labels[keep_high]
                metrics_ = calculate_metrics(probs_high, labels_high, log_images=self.log_images)
                metrics.update({
                    f"{metric}_high_involvement": value
                    for metric, value in metrics_.items()
                })

        # low-involvement core predictions (existing `label`-based split, unchanged)
        if self.include_low_involvement_metrics:
            low_involvement = involvement <= self.involvement_threshold
            keep_low = np.logical_or(low_involvement, benign)
            if keep_low.sum() > 0:
                probs_low = core_probs[keep_low]
                labels_low = core_labels[keep_low]
                metrics_ = calculate_metrics(probs_low, labels_low, log_images=self.log_images)
                metrics.update({
                    f"{metric}_low_involvement": value
                    for metric, value in metrics_.items()
                })


        # patient predictions
        if self.include_patient_metrics and "patient_id" in results_table.columns:
            for source_col, prefix in [
                ("average_needle_heatmap_value", "heatmap"),
                ("image_level_cancer_logits", "image_level"),
            ]:
                if source_col not in results_table.columns:
                    continue

                # single agg call -> one index, label/prediction alignment
                # guaranteed by construction rather than by sort stability
                pat = results_table.groupby("patient_id").agg(
                    gg_max=("grade_group", "max"),
                    pred_max=(source_col, "max"),
                    pred_mean=(source_col, "mean"),
                )
                grouped_col = results_table.groupby("patient_id")[source_col]
                pat["pred_top2"] = grouped_col.apply(_topk_mean, k=2)
                pat["pred_top3"] = grouped_col.apply(_topk_mean, k=3)
                pat["pred_top5"] = grouped_col.apply(_topk_mean, k=5)

                for task, gg_thresh in [("pca", 1), ("cspca", 2)]:
                    patient_labels = (pat["gg_max"] >= gg_thresh).astype(int).values

                    if patient_labels.sum() == 0 or patient_labels.sum() == len(patient_labels):
                        continue

                    metrics[f"n_patients_{task}"] = int(len(patient_labels))
                    metrics[f"n_patients_pos_{task}"] = int(patient_labels.sum())

                    for agg_name in ["max", "mean", "top2", "top3", "top5"]:
                        patient_preds = pat[f"pred_{agg_name}"].values
                        try:
                            metrics_ = calculate_metrics(
                                patient_preds, patient_labels, log_images=self.log_images
                            )
                            metrics.update({
                                f"{metric}_patient_{prefix}_{task}_{agg_name}": value
                                for metric, value in metrics_.items()
                            })
                        except ValueError:
                            pass



        # patient predictions
        # ---- patient-level: aggregate core predictions to one score per patient ----
        # if self.include_patient_metrics and "patient_id" in results_table.columns:
        #     for source_col, prefix in [
        #         ("average_needle_heatmap_value", "heatmap"),
        #         ("image_level_cancer_logits", "image_level"),
        #     ]:
        #         if source_col not in results_table.columns:
        #             continue

        #         grouped = results_table.groupby("patient_id")

        #         for task, gg_thresh in [("pca", 1), ("cspca", 2)]:
        #             # patient is positive if ANY of its cores meets the grade threshold
        #             patient_labels = (
        #                 grouped["grade_group"].max() >= gg_thresh
        #             ).astype(int).values

        #             for agg_name, agg_fn in [("max", "max"), ("mean", "mean")]:
        #                 patient_preds = grouped[source_col].agg(agg_fn).values

        #                 if patient_labels.sum() == 0 or patient_labels.sum() == len(patient_labels):
        #                     continue  # single-class -- roc_auc_score would raise

        #                 try:
        #                     metrics_ = calculate_metrics(
        #                         patient_preds, patient_labels, log_images=self.log_images
        #                     )
        #                     metrics.update({
        #                         f"{metric}_patient_{prefix}_{task}_{agg_name}": value
        #                         for metric, value in metrics_.items()
        #                     })
        #                     metrics[f"n_patients_{task}"] = int(len(patient_labels))
        #                     metrics[f"n_patients_pos_{task}"] = int(patient_labels.sum())
        #                 except ValueError:
        #                     pass

        # ---- image-level: PCa/csPCa, overall + involvement-stratified ----
        if "image_level_cancer_logits" in results_table.columns:
            image_level_predictions = results_table.image_level_cancer_logits.values

            # existing binary label-based image-level metric, unchanged
            metrics_ = calculate_metrics(
                image_level_predictions, results_table.label.values, log_images=self.log_images
            )
            metrics.update(
                {f"{metric}_image_level": value for metric, value in metrics_.items()}
            )

            self._pca_cspca_stratified(
                metrics, image_level_predictions, grade_group, involvement,
                prefix="image_level",
            )

        # ---- heatmap: PCa/csPCa, overall + involvement-stratified ----
        if self.include_heatmap_cspca_metrics:
            heatmap_predictions = results_table["average_needle_heatmap_value"].values

            self._pca_cspca_stratified(
                metrics, heatmap_predictions, grade_group, involvement,
                prefix="heatmap",
            )

        # convert to float
        for key in metrics:
            if isinstance(metrics[key], np.floating):
                metrics[key] = float(metrics[key])

        return metrics