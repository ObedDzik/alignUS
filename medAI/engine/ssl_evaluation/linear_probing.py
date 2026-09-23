from typing import Callable
import logging
from medAI.metrics import calculate_binary_classification_metrics
from medAI.utils.distributed import concat_all_gather, is_main_process
from sklearn.linear_model import LogisticRegression


import numpy as np
import torch
import torch.distributed as dist
from torch import distributed as dist
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.decomposition import PCA


logger = logging.getLogger("Linear Probing")


def fit_and_predict_logistic_regression(X_train, y_train, X_val, version="default"):
    """
    Fit a logistic regression probe with optional normalization/scaling.

    version options:
      - 'legacy': original implementation (no normalization)
      - 'default': L2 normalize + standardize + LogisticRegression(saga)
      - 'fast': L2 normalize + SGDClassifier(logistic)
      - 'pca': L2 normalize + standardize + PCA(512) + LogisticRegression(saga)
      - 'bare': no scaling, plain LogisticRegression
    """

    # --- convert to numpy or torch ---
    if isinstance(X_train, torch.Tensor):
        X_train = X_train.detach().cpu()
    if isinstance(X_val, torch.Tensor):
        X_val = X_val.detach().cpu()
    if isinstance(y_train, torch.Tensor):
        y_train = y_train.detach().cpu().numpy()

    # --- legacy version (your original code) ---
    if version == "legacy":
        clf = LogisticRegression(max_iter=5000, class_weight="balanced")
        clf.fit(X_train, y_train)
        y_pred_train = clf.predict_proba(X_train)
        y_pred_val = clf.predict_proba(X_val)
        return y_pred_train, y_pred_val

    # --- otherwise, normalized/scaled variants ---
    steps = []

    # L2 normalization per feature vector
    if version in ["default", "fast", "pca"]:
        X_train = torch.tensor(X_train).float()
        X_val = torch.tensor(X_val).float()
        X_train = torch.nn.functional.normalize(X_train, dim=1).numpy()
        X_val = torch.nn.functional.normalize(X_val, dim=1).numpy()
    else:
        X_train = X_train.numpy()
        X_val = X_val.numpy()

    # select preset
    if version == "bare":
        clf = LogisticRegression(max_iter=5000, class_weight="balanced")

    elif version == "default":
        steps.append(("scaler", StandardScaler(with_mean=False)))
        clf = LogisticRegression(
            solver="saga",
            penalty="l2",
            class_weight="balanced",
            max_iter=2000,
            n_jobs=-1,
        )

    elif version == "fast":
        clf = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            class_weight="balanced",
            max_iter=1000,
            n_jobs=-1,
        )

    elif version == "pca":
        steps.append(("scaler", StandardScaler(with_mean=False)))
        steps.append(("pca", PCA(n_components=512, whiten=True, random_state=0)))
        clf = LogisticRegression(
            solver="saga",
            penalty="l2",
            class_weight="balanced",
            max_iter=2000,
            n_jobs=-1,
        )

    else:
        raise ValueError(f"Unknown version: {version}")

    steps.append(("clf", clf))
    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)

    y_pred_train = pipeline.predict_proba(X_train)
    y_pred_val = pipeline.predict_proba(X_val)
    return y_pred_train, y_pred_val


def fit_and_predict_kfold_logistic_regression(
    X_trainval, y_trainval, n_splits=5, version="default", random_state=0
):
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_y_pred = np.zeros((X_trainval.shape[0], len(np.unique(y_trainval))), dtype=np.float32)

    for train_idx, val_idx in skf.split(X_trainval, y_trainval):
        X_train, y_train = X_trainval[train_idx], y_trainval[train_idx]
        X_val = X_trainval[val_idx]

        _, y_pred_val = fit_and_predict_logistic_regression(
            X_train, y_train, X_val, version=version
        )
        all_y_pred[val_idx] = y_pred_val

    return all_y_pred



class LinearProbing:
    def __init__(
        self,
        model=None,
        train_loader=None,
        val_loader=None,
        device="cuda",
        metric_fn=calculate_binary_classification_metrics,
        concat_across_processes=True,
        regression_version="legacy",
        run_kfold_cv=False,
    ):
        """Initialize the linear probing evaluator.

        Dataloaders are expected to return (image, label) tuples or a dict with 'image' and 'label' keys.
        Model is expected to return the features directly or a dict with 'image_level_feature' key.

        Args:
            train_loader: DataLoader for the training set - batches should be image, label tuples.
            val_loader: DataLoader for the validation set.
            device: Device to run the evaluation on.
        """
        self.model = model
        self.metric_fn = metric_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.concat_across_processes = concat_across_processes
        self.regression_version = regression_version
        self.run_kfold_cv = run_kfold_cv

    @torch.no_grad()
    def _extract_features(self, loader, desc="Extracting Features"):

        assert self.model is not None, "Model must be provided for feature extraction"

        self.model.eval().to(self.device)

        features = []
        labels = []
        for data in tqdm(loader, desc=desc):
            if isinstance(data, dict):
                image = data["image"]
                label = data["label"]
            else:
                image, label = data

            image = image.to(self.device)
            label = label.to(self.device)
            outputs = self.model(image)

            if isinstance(outputs, dict) and "image_level_feature" in outputs:
                clstoken = outputs["image_level_feature"]
            else:
                clstoken = outputs
            clstoken = concat_all_gather(clstoken.contiguous())
            label = concat_all_gather(label.contiguous())
            features.append(clstoken)
            labels.append(label)

        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)

        return features, labels

    @torch.no_grad()
    def run(self):
        """Returns the metrics for the linear probing task."""

        assert self.train_loader is not None, "Train loader must be provided"
        assert self.model is not None, "Model must be provided for probing"

        X_train, y_train = self._extract_features(
            self.train_loader, desc="Extracting Train Features"
        )
        X_train = X_train.cpu().numpy()
        y_train = y_train.cpu().numpy()
        
        if self.run_kfold_cv:
            logging.info("Running k-fold cross-validation on training set")
            if self.val_loader is not None:
                logging.warning("Val loader is ignored when running k-fold CV")

            X_val, y_val = None, None
        else:
            X_val, y_val = self._extract_features(
                self.val_loader, desc="Extracting Val Features"
            )
            X_val = X_val.cpu().numpy()
            y_val = y_val.cpu().numpy()

        if not is_main_process():
            return {}

        # clf = LogisticRegression(max_iter=5000, class_weight="balanced")
        # clf.fit(X_train, y_train)
# 
        # y_pred_train = clf.predict_proba(X_train)
        # y_pred_val = clf.predict_proba(X_val)

        if not self.run_kfold_cv:
            y_pred_train, y_pred_val = fit_and_predict_logistic_regression(
                X_train, y_train, X_val, version=self.regression_version
            )

            train_metrics = self.metric_fn(y_pred_train, y_train)
            val_metrics = self.metric_fn(y_pred_val, y_val)

            return {
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()},
            }
        else: 
            X_trainval = X_train 
            y_trainval = y_train
            y_pred_trainval = fit_and_predict_kfold_logistic_regression(
                X_trainval, y_trainval, n_splits=5, version=self.regression_version
            )
            val_metrics = self.metric_fn(y_pred_trainval, y_trainval)
            return {f"val/{k}": v for k, v in val_metrics.items()}


class FeatureMapLinearProbing:
    def __init__(
        self,
        model=None,
        train_loader=None,
        val_loader=None,
        device="cuda",
        ignore_idx=-1,
        metric_fn=calculate_binary_classification_metrics,
        metrics_level="image",
        concat_across_processes=True,
        regression_version="legacy",
        run_kfold_cv=False,
    ):
        """
        Args:
            model: The model to evaluate.
            train_loader: DataLoader for the training set - batches should be image, mask tuples.
            val_loader: DataLoader for the validation set.
            device: Device to run the evaluation on.
            feature_map_fn: A function that takes the model and an image input and returns the feature map.
            ignore_idx: The index in the mask that should be ignored.
            metric_fn: A function that takes predictions and ground truth and returns a dictionary of metrics.
        """

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.ignore_idx = ignore_idx
        self.metric_fn = metric_fn
        self.metrics_level = metrics_level
        self.concat_across_processes = concat_across_processes
        self.regression_version = regression_version
        self.run_kfold_cv = run_kfold_cv

    @torch.no_grad()
    def run(self):
        assert self.train_loader is not None, "Train loader must be provided"
        assert self.model is not None, "Model must be provided for probing"

        self.model.eval()

        all_train_features, all_train_labels, all_train_indices = self.extract_features(
            self.train_loader
        )
        all_train_features = all_train_features.cpu().numpy()
        all_train_labels = all_train_labels.cpu().numpy()
        all_train_indices = all_train_indices.cpu().numpy()

        if self.run_kfold_cv:
            logging.info("Running k-fold cross-validation on training set")
            if self.val_loader is not None:
                logging.warning("Val loader is ignored when running k-fold CV")
            all_val_features, all_val_labels, all_val_indices = (
                None, 
                None, 
                None 
            )
        else: 
            assert self.val_loader is not None, "Val loader must be provided if not running k-fold CV"
            all_val_features, all_val_labels, all_val_indices = self.extract_features(
                self.val_loader
            )
            all_val_features = all_val_features.cpu().numpy()
            all_val_labels = all_val_labels.cpu().numpy()
            all_val_indices = all_val_indices.cpu().numpy()

        if not is_main_process():
            return {}

        logger.info(
            f"Training logistic regression on \
                {all_train_features.shape[0]} samples with {len(np.unique(all_train_labels))} classes, \
                    {len(np.unique(all_train_indices))} images, {all_train_features.shape[1]} features"
        )

        if self.run_kfold_cv:
            X_trainval = all_train_features
            y_trainval = all_train_labels
            indices_trainval = all_train_indices
            y_pred_trainval = fit_and_predict_kfold_logistic_regression(
                X_trainval, y_trainval, n_splits=5, version=self.regression_version
            )
            trainval_metrics = self.run_metrics_aggregation(
                y_pred_trainval, y_trainval, indices_trainval
            )
            return {f"val/{k}": v for k, v in trainval_metrics.items()}

        else: 
            train_preds, val_preds = fit_and_predict_logistic_regression(
                all_train_features,
                all_train_labels,
                all_val_features,
                version=self.regression_version,
            )
            train_metrics = {}
            val_metrics = {}

            train_metrics = self.run_metrics_aggregation(
                train_preds, all_train_labels, all_train_indices
            )
            val_metrics = self.run_metrics_aggregation(
                val_preds, all_val_labels, all_val_indices
            )
            return {
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()},
            }

        # if self.metrics_level == "pixel":
        #     # pixel level metrics
        #     if self.metric_fn is not None:
        #         train_metrics = self.metric_fn(train_preds, all_train_labels)
        #         val_metrics = self.metric_fn(val_preds, all_val_labels)
# 
        # elif self.metrics_level == "image":
        #     # reduce to image level
        #     train_preds = np.array(
        #         [
        #             train_preds[all_train_indices == i].mean(0)
        #             for i in np.unique(all_train_indices)
        #         ]
        #     )
        #     val_preds = np.array(
        #         [
        #             val_preds[all_val_indices == i].mean(0)
        #             for i in np.unique(all_val_indices)
        #         ]
        #     )
        #     all_train_labels = np.array(
        #         [
        #             all_train_labels[all_train_indices == i][0]
        #             for i in np.unique(all_train_indices)
        #         ]
        #     )
        #     all_val_labels = np.array(
        #         [
        #             all_val_labels[all_val_indices == i][0]
        #             for i in np.unique(all_val_indices)
        #         ]
        #     )
# 
        #     if self.metric_fn is not None:
        #         train_metrics = self.metric_fn(train_preds, all_train_labels)
        #         val_metrics = self.metric_fn(val_preds, all_val_labels)
# 
        # metrics = {
        #     **{f"train/{k}": v for k, v in train_metrics.items()},
        #     **{f"val/{k}": v for k, v in val_metrics.items()},
        # }
        # return metrics

    def run_metrics_aggregation(self, preds, labels, indices): 

        if self.metric_fn is None:
            return {}

        if self.metrics_level == "pixel":
            metrics = self.metric_fn(preds, labels)
        elif self.metrics_level == "image":
            # reduce to image level
            preds = np.array(
                [
                    preds[indices == i].mean(0)
                    for i in np.unique(indices)
                ]
            )
            labels = np.array(
                [
                    labels[indices == i][0]
                    for i in np.unique(indices)
                ]
            )
            metrics = self.metric_fn(preds, labels)
        else: 
            raise ValueError(f"Unknown metrics level: {self.metrics_level}")

        return metrics

    @torch.no_grad()
    def extract_features(self, loader):
        assert self.model is not None, "Model must be provided for feature extraction"
        self.model.eval().to(self.device)

        all_features = []
        all_labels = []
        all_indices = []

        for idx, batch in enumerate(tqdm(loader)):
            if dist.is_initialized():
                idx = idx + dist.get_rank() * len(
                    loader
                )  # to keep unique indices across processes

            if isinstance(batch, dict):
                image, mask = batch["image"], batch["target_mask"]
            else:
                image, mask = batch

            image = image.to(self.device)
            mask = mask.to(self.device)
            model_outputs = self.model(image)
            if isinstance(model_outputs, dict) and "feature_map" in model_outputs:
                feature_map = model_outputs["feature_map"]
            else:
                feature_map = model_outputs
            feature_map = feature_map.permute(0, 2, 3, 1)  # B, H, W, C
            B, H, W, C = feature_map.shape

            if mask.ndim != 4:
                mask = mask.unsqueeze(1)
            mask = mask.float()
            mask = torch.nn.functional.interpolate(mask, (H, W), mode="nearest")
            mask = mask.long()
            mask = mask.squeeze(1)

            ignore_mask = mask == self.ignore_idx
            image_indices = (
                torch.arange(B, device=self.device)[..., None, None].repeat(1, H, W)
                + idx * B
            )
            feature_map = feature_map[~ignore_mask]  # B, C
            mask = mask[~ignore_mask]  # B
            image_indices = image_indices[~ignore_mask]  # B

            if dist.is_initialized() and self.concat_across_processes:
                feature_map = _all_gather_variable(feature_map.contiguous())
                mask = _all_gather_variable(mask.contiguous())
                image_indices = _all_gather_variable(image_indices.contiguous())

            all_features.append(feature_map)
            all_labels.append(mask)
            all_indices.append(image_indices)

        all_features = torch.cat(all_features)
        all_labels = torch.cat(all_labels)
        all_indices = torch.cat(all_indices)

        return all_features, all_labels, all_indices


def _all_gather_variable(tensor):
    import torch.distributed as dist

    if not dist.is_initialized():
        return tensor
    local_n = torch.tensor([tensor.shape[0]], device=tensor.device)
    size_list = [torch.zeros_like(local_n) for _ in range(dist.get_world_size())]
    dist.all_gather(size_list, local_n)
    sizes = [int(x.item()) for x in size_list]
    max_size = max(sizes)
    if tensor.shape[0] < max_size:
        pad_shape = (max_size - tensor.shape[0],) + tensor.shape[1:]
        pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        tensor_padded = torch.cat([tensor, pad], dim=0)
    else:
        tensor_padded = tensor
    gather_list = [
        torch.empty_like(tensor_padded) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(gather_list, tensor_padded)
    out = [gather_list[i][: sizes[i]] for i in range(len(sizes))]
    return torch.cat(out, dim=0)
