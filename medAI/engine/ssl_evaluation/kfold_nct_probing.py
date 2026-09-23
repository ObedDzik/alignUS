from medAI.datasets.nct2013.nct2013_legacy import DataKeys, NCT2013Dataset
from medAI.engine.ssl_evaluation.probing_nct import TransformNCT, _extract_all_data
from medAI.metrics import calculate_binary_classification_metrics
from medAI.utils.distributed import is_main_process
import logging

import numpy as np
import pandas as pd
import torch
import torch.distributed


_logger = logging.getLogger("NCT Probing")


class KFoldNCTProbing:
    def __init__(
        self,
        model=None,
        batch_size=8,
        holdout_center="UVA",
        image_size=512,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        prostate_masks_dir=None,
        metadata_path=None,
        raw_data_dir=None,
        dataset_kwargs={},
        device="cuda",
        output_adapter=None,
    ):
        self.model = model
        self.device = device
        self.output_adapter = output_adapter

        print(f"Selecting cores for probing (leaving out {holdout_center})...")

        assert prostate_masks_dir is not None, "prostate_masks_dir must be provided"
        assert metadata_path is not None, "metadata_path must be provided"
        assert raw_data_dir is not None, "raw_data_dir must be provided"

        _logger.info(f"Raw data dir: {raw_data_dir}")
        _logger.info(f"Prostate mask dir: {prostate_masks_dir}")
        _logger.info(f"Metadata path: {metadata_path}")

        metadata_table = pd.read_csv(metadata_path)

        _logger.info("Getting core ids...")
        self.train_ds = NCT2013Dataset(
            data_path=raw_data_dir,
            prostate_mask_dir=prostate_masks_dir,
            core_ids=metadata_table.core_id.to_list(),
            items=[
                DataKeys.BMODE,
                DataKeys.NEEDLE_MASK,
                DataKeys.PROSTATE_MASK,
                DataKeys.GRADE_GROUP,
                DataKeys.PCT_CANCER,
                DataKeys.PATIENT_ID,
                DataKeys.CORE_ID,
            ],
            transform=TransformNCT(
                size=image_size,
                mean=mean,
                std=std,
            ),
            **dataset_kwargs,
        )
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.DistributedSampler(self.train_ds)
        else:
            sampler = None
        self.train_loader = torch.utils.data.DataLoader(
            self.train_ds, batch_size=batch_size, num_workers=4, sampler=sampler
        )

    def run(
        self,
    ):
        assert self.model is not None, "Model must be provided for probing"

        self.model.eval().to(self.device)

        train_outputs = _extract_all_data(
            self.model, self.train_loader, self.device, self.output_adapter
        )

        if not is_main_process():
            return {}

        patient_ids = np.unique(train_outputs["patient_id"])

        from sklearn.model_selection import KFold
        from sklearn.linear_model import LogisticRegression

        # ======== Linear probing at patch level ===================
        train_metrics_by_fold = []
        val_metrics_by_fold = []

        for train_index, val_index in KFold(
            n_splits=5, shuffle=True, random_state=0
        ).split(patient_ids):

            train_patient_ids = patient_ids[train_index]
            val_patient_ids = patient_ids[val_index]

            train_index = np.where(
                np.isin(train_outputs["patient_id"], train_patient_ids)
            )[0]
            val_index = np.where(np.isin(train_outputs["patient_id"], val_patient_ids))[
                0
            ]

            X_train = train_outputs["features"][train_index]
            y_train = train_outputs["labels"][train_index]
            inv_train = train_outputs["involvement"][train_index]

            # for training, exclude low involvement cancer cores
            mask = (y_train == 1) & (inv_train < 40)
            X_train = X_train[~mask]
            y_train = y_train[~mask]

            clf = LogisticRegression(
                random_state=0, max_iter=10000, class_weight="balanced"
            ).fit(X_train, y_train)
            y_hat_train = clf.predict_proba(X_train)[:, 1]

            X_val = train_outputs["features"][val_index]
            y_val = train_outputs["labels"][val_index]
            y_hat_val = clf.predict_proba(X_val)[:, 1]

            train_metrics_i = calculate_binary_classification_metrics(
                y_hat_train, y_train
            )
            val_metrics_i = calculate_binary_classification_metrics(y_hat_val, y_val)
            train_metrics_by_fold.append(train_metrics_i)
            val_metrics_by_fold.append(val_metrics_i)

        def _agg_metrics(metrics_by_fold):
            metrics = {}
            for key in metrics_by_fold[0].keys():
                metric_by_folds = [metrics_i[key] for metrics_i in metrics_by_fold]
                metrics[key] = sum(metric_by_folds) / len(metric_by_folds)
            return metrics

        train_metrics = _agg_metrics(train_metrics_by_fold)
        val_metrics = _agg_metrics(val_metrics_by_fold)

        return {
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"val/{k}": v for k, v in val_metrics.items()},
        }