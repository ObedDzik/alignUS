import os
from typing import Protocol
from sklearn.model_selection import KFold
import torch.utils
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch
from torchvision import transforms as T
import torch.distributed as dist
from tqdm import tqdm
import torch
from torch import Tensor, nn
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    roc_curve,
    f1_score,
    recall_score,
    balanced_accuracy_score,
    average_precision_score,
)
import matplotlib.pyplot as plt
import warnings
import wandb

from .kfold_nct_probing import KFoldNCTProbing


from .probing_nct import NCTProbing
from medAI.utils.distributed import concat_all_gather, is_main_process
from medAI.transforms.ibot import NormalizeToTensor
from sklearn.linear_model import LogisticRegression
from torch import distributed as dist
from medAI.utils.cosine_scheduler import cosine_scheduler
from medAI.engine.linear_probing import SKLearnLinearProbing
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
import logging
from torch import distributed as dist


logger = logging.getLogger(__name__)


def compute_binary_classification_metrics(y_score, y_true, log_images=False):
    """Calculate metrics for the cancer classification problem.

    Args:
        y_score (np.array or torch.Tensor) - A column vector of predicted probabilities for
            cancer (1) or benign(0)
        y_true (np.array or torch.Tensor) - A column vector of true labels for cancer (1) or benign(0)
        log_images (bool) - If True, log images of the histogram of predictions and the ROC curve to
            wandb. Default is False.
    """

    if isinstance(y_score, torch.Tensor):
        y_score = y_score.cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    if isinstance(y_score, list):
        y_score = np.array(y_score)

    if y_score.ndim > 1:
        y_score = y_score[:, -1]

    # augmentations can cause NaNs
    nanvalues = np.isnan(y_score)
    y_score = y_score[~nanvalues]
    y_true = y_true[~nanvalues]

    metrics = {}

    try:
        metrics["auc"] = roc_auc_score(y_true, y_score)
    except ValueError:
        warnings.warn("ROC AUC score could not be calculated. Setting to 0.5")
        metrics["auc"] = 0.5

    # find the sensitivity at fixed specificities
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    for specificity in [0.20, 0.40, 0.60, 0.80]:
        sensitivity = tpr[np.argmax(fpr > 1 - specificity)]
        metrics[f"sens_at_{specificity*100:.0f}_spe"] = sensitivity

    # choose the threshold that maximizes balanced accuracy
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    metrics["f1"] = f1_score(y_true, y_score > best_threshold)

    if log_images:
        plt.hist(y_score[y_true == 0], bins=100, alpha=0.5, density=True)
        plt.hist(y_score[y_true == 1], bins=100, alpha=0.5, density=True)
        plt.legend(["Benign", "Cancer"])
        plt.xlabel(f"Probability of cancer")
        plt.ylabel("Density")
        plt.title(f"AUC: {metrics['auc']:.3f}")
        metrics["histogram"] = wandb.Image(plt, caption="Histogram of core predictions")
        plt.close()

        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC curve")
        metrics["roc_curve"] = wandb.Image(plt, caption="ROC curve")
        plt.close()

    metrics["balanced_accuracy"] = balanced_accuracy_score(
        y_true, y_score > best_threshold
    )
    metrics["auprc"] = average_precision_score(y_true, y_score)

    return metrics


class GetClassTokenFn(Protocol):
    def __call__(self, model: nn.Module, x: Tensor) -> Tensor:
        """Takes a network and image input and returns the `class token` - which
        could be any D-dimensional representation for the image.

        Returns:
            Tensor - shape `batch_size, n_features`
        """
        raise NotImplementedError()


class LinearProbing:
    def __init__(self, train_loader, val_loader, device):
        """Initialize the linear probing evaluator.

        Args:
            train_loader: DataLoader for the training set - batches should be image, label tuples.
            val_loader: DataLoader for the validation set.
            device: Device to run the evaluation on.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    @torch.no_grad()
    def _extract_features(
        self, loader, model, get_cls_token_fn: GetClassTokenFn, desc: str | None = None
    ):
        model.eval().to(self.device)

        features = []
        labels = []
        for image, label in tqdm(loader, desc=desc):
            image = image.to(self.device)
            label = label.to(self.device)
            cls = get_cls_token_fn(model, image)
            cls = concat_all_gather(cls.contiguous())
            label = concat_all_gather(label.contiguous())
            features.append(cls)
            labels.append(label)

        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)

        return features, labels

    def run_probing(
        self, model, get_cls_token_fn: GetClassTokenFn, is_main_process=True
    ):
        """Returns the metrics for the linear probing task."""

        X_train, y_train = self._extract_features(
            self.train_loader, model, get_cls_token_fn
        )
        X_train = X_train.cpu().numpy()
        y_train = y_train.cpu().numpy()
        X_val, y_val = self._extract_features(self.val_loader, model, get_cls_token_fn)
        X_val = X_val.cpu().numpy()
        y_val = y_val.cpu().numpy()

        if is_main_process:

            clf = LogisticRegression(max_iter=5000, class_weight="balanced")
            clf.fit(X_train, y_train)

            y_pred_train = clf.predict_proba(X_train)[:, -1]
            y_pred_val = clf.predict_proba(X_val)[:, -1]

            train_metrics = compute_binary_classification_metrics(y_pred_train, y_train)
            val_metrics = compute_binary_classification_metrics(y_pred_val, y_val)

            return train_metrics, val_metrics

        else:
            return None


class KFoldLinearProbing:
    def __init__(self, full_loader, device, n_splits=5):
        """Sets up the KFoldLinearProbing evaluator.

        Args:
            full_loader: DataLoader for the full dataset. A batch should be a tuple of the form (group_id, image, label).
                here, group_id should be an identifier for the group the image belongs to. A group is a set of images that
                should not be split across folds. For example, in the NCT dataset, a group is a patient - all images from the
                same patient should be in the same fold.
            device: Device to run the evaluation on.
            n_splits: Number of splits for the KFold cross validation.
        """

        self.full_loader = full_loader
        self.device = device
        self.n_splits = n_splits

    def run_probing(
        self, model, get_cls_token_fn: GetClassTokenFn, is_main_process=True
    ):
        """Returns the metrics for the linear probing task.

        Args:
            model: The model to evaluate.
            get_cls_token_fn: A function that takes the model and an image input and returns the class token.
            is_main_process: If True, the metrics are computed and returned. If False, an empty dictionary is returned.
        """

        group_ids, X_full, y_full = self._extract_features(
            self.full_loader, model, get_cls_token_fn
        )
        X_full = X_full.cpu().numpy()
        y_full = y_full.cpu().numpy()
        group_ids = group_ids.cpu().numpy()

        if is_main_process:

            train_metrics = {}
            val_metrics = {}

            group_ids_unique = np.unique(group_ids)
            for train_index, val_index in KFold(
                n_splits=5, shuffle=True, random_state=0
            ).split(group_ids_unique):
                group_ids_train = group_ids_unique[train_index]
                group_ids_val = group_ids_unique[val_index]

                # select the group ids that are in the train and validation sets
                train_mask = np.where(np.isin(group_ids, group_ids_train))[0]
                val_mask = np.where(np.isin(group_ids, group_ids_val))[0]

                X_train, y_train = X_full[train_mask], y_full[train_mask]
                X_val, y_val = X_full[val_mask], y_full[val_mask]

                clf = LogisticRegression(max_iter=5000, class_weight="balanced")
                clf.fit(X_train, y_train)

                y_pred_train = clf.predict_proba(X_train)[:, -1]
                y_pred_val = clf.predict_proba(X_val)[:, -1]

                train_metrics_ = compute_binary_classification_metrics(
                    y_pred_train, y_train
                )
                val_metrics_ = compute_binary_classification_metrics(y_pred_val, y_val)

                for k, v in train_metrics_.items():
                    train_metrics[k] = train_metrics.get(k, 0) + v
                for k, v in val_metrics_.items():
                    val_metrics[k] = val_metrics.get(k, 0) + v

            for k in train_metrics.keys():
                train_metrics[k] /= self.n_splits
            for k in val_metrics.keys():
                val_metrics[k] /= self.n_splits

            return train_metrics, val_metrics

        else:
            return None

    def _extract_features(
        self, loader, model, get_cls_token_fn: GetClassTokenFn, desc: str | None = None
    ):
        model.eval().to(self.device)

        features = []
        labels = []
        group_ids = []
        for group_id, image, label in tqdm(loader, desc=desc):
            image = image.to(self.device)
            label = label.to(self.device)
            cls = get_cls_token_fn(model, image)
            cls = concat_all_gather(cls.contiguous())
            label = concat_all_gather(label.contiguous())
            features.append(cls)
            labels.append(label)
            group_ids.append(group_id)

        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
        group_ids = torch.cat(group_ids, dim=0)

        return group_ids, features, labels


class FineTuning:

    LOG_NAME = "finetune"

    def __init__(
        self,
        backbone: nn.Module,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        optimizer="adam",
        lr=1e-4,
        backbone_lr=1e-5,
        epochs=20,
        in_features=768,
        n_classes=2,
        log_fn=lambda metrics_dict: print(metrics_dict),
        metrics_fn=lambda y_score, y_true: compute_binary_classification_metrics(
            y_score, y_true
        ),
        monitored_metric="auc",
        warmup_epochs=5,
        head_hidden_dims=[],
    ):
        self.backbone = backbone
        self.optimizer = optimizer
        self.lr = lr
        self.backbone_lr = backbone_lr
        self.epochs = epochs
        self.in_features = in_features
        self.n_classes = n_classes
        self.log_fn = log_fn
        self.metrics_fn = metrics_fn
        self.monitored_metric = monitored_metric
        self._best_model_state = None
        self.best_val_metric = -1000000
        self.warmup_epochs = warmup_epochs
        self.criterion = criterion
        self.head_hidden_dims = head_hidden_dims

    @property
    def best_model(self):
        model = nn.Sequential(self.backbone, self.head)
        model.load_state_dict(self._best_model_state)
        return model

    def run(self, train_loader, val_loader, test_loader=None):
        self.head = self._make_head().to(next(self.backbone.parameters()).device)
        opt, sched = self._setup_optimizer(self.backbone, self.head, len(train_loader))
        model = nn.Sequential(self.backbone, self.head)
        if dist.is_initialized():
            model = torch.nn.parallel.DistributedDataParallel(model)

        for epoch in range(self.epochs):
            train_metrics = self._epoch(
                model, train_loader, self.criterion, opt, sched, f"Train epoch {epoch}"
            )
            val_metrics = self._epoch(
                model, val_loader, self.criterion, desc=f"Val epoch {epoch}"
            )
            if val_metrics[self.monitored_metric] > self.best_val_metric:
                self.best_val_metric = val_metrics[self.monitored_metric]
                self._best_model_state = model.state_dict()

            metrics = {}
            metrics.update(
                {f"{self.LOG_NAME}_val_{k}": v for k, v in val_metrics.items()}
            )
            metrics.update(
                {f"{self.LOG_NAME}_train_{k}": v for k, v in train_metrics.items()}
            )
            self.log_fn(metrics)

        if test_loader is not None:
            test_metrics, (class_scores, gt_labels) = self._epoch(
                self.best_model, test_loader, self.criterion, desc=f"Test epoch {epoch}"
            )
            metrics = {f"{self.LOG_NAME}_test_{k}": v for k, v in test_metrics.items()}
            self.log_fn(metrics)
            return (class_scores, gt_labels)

    def _epoch(
        self,
        model,
        loader,
        criterion,
        opt=None,
        sched: torch.optim.lr_scheduler.LRScheduler | None = None,
        desc="Running",
        return_outputs=False,
    ):
        training = opt is not None
        device = next(model.parameters()).device
        model.train(training)
        class_scores = []
        gt_labels = []

        with torch.set_grad_enabled(training):
            for image, label in tqdm(loader, desc=desc):
                image = image.to(device)
                label = label.to(device)
                step_logs = {}

                score = model(image)
                loss = criterion(score, label)

                if training:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    if sched is not None:
                        sched.step()
                        step_logs["lr"] = sched.get_last_lr()[-1]

                step_logs["loss"] = loss.item()
                class_scores.append(score.detach().cpu())
                gt_labels.append(label.detach().cpu())

                self.log_fn(step_logs)

        class_scores = torch.cat(class_scores)
        gt_labels = torch.cat(gt_labels)

        metrics = self.metrics_fn(class_scores, gt_labels)
        if return_outputs:
            return (metrics, (class_scores, gt_labels))
        else:
            return metrics

    def _setup_optimizer(self, backbone, linear_layer, n_step_per_ep):
        params_groups = [
            {"params": backbone.parameters(), "lr": self.backbone_lr},
            {"params": linear_layer.parameters(), "lr": self.lr, "wd": 1e-5},
        ]
        match self.optimizer:
            case "adam":
                opt = torch.optim.Adam(params_groups)
            case _:
                raise NotImplementedError(self.optimizer)

        sch1 = cosine_scheduler(
            1, 0, self.epochs, n_step_per_ep, warmup_epochs=self.warmup_epochs
        )
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda epoch: sch1[epoch] if epoch < len(sch1) else 0
        )
        return opt, sched

    def _make_head(self):
        if len(self.head_hidden_dims) == 0:
            return nn.Linear(self.in_features, self.n_classes)
        else:
            layers = []
            in_dim = self.in_features
            for out_dim in self.head_hidden_dims:
                layers.append(nn.Linear(in_dim, out_dim))
                layers.append(nn.ReLU())
                in_dim = out_dim
            layers.append(nn.Linear(in_dim, self.n_classes))
            return nn.Sequential(*layers)


def get_nct_patches_loaders(
    data_path: str = '',  # os.environ["NCT_PATCHES"],
    input_size: int = 512,
    batch_size: int = 8,
    shuffle_train=True,
):
    transform = T.Compose(
        [
            T.Resize((input_size, input_size)),
            NormalizeToTensor(),
        ]
    )

    train_ds = ImageFolder(
        os.path.join(data_path, "train"),
        transform=transform,
        target_transform=lambda l: torch.tensor(l).long(),
    )
    val_ds = ImageFolder(
        os.path.join(data_path, "val"),
        transform=transform,
        target_transform=lambda l: torch.tensor(l).long(),
    )

    if dist.is_initialized():
        make_loader = lambda ds, shuffle: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=torch.utils.data.distributed.DistributedSampler(ds),
        )
    else:
        make_loader = lambda ds, shuffle: DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle
        )
    train_loader = make_loader(train_ds, shuffle_train)
    val_loader = make_loader(val_ds, False)
    return train_loader, val_loader


def build_linear_probe_for_nct_patches(
    data_path: str = os.environ.get("NCT_PATCHES", ""),
    input_size: int = 512,
    batch_size: int = 8,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    train_loader, val_loader = get_nct_patches_loaders(
        data_path=data_path, input_size=input_size, batch_size=batch_size
    )

    return LinearProbing(train_loader, val_loader, device)


def build_kfold_linear_probe_for_nct_patches(
    data_path: str,
    input_size: int = 512,
    batch_size: int = 8,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):

    from src.datasets import NCT2013PatchesDataset

    ds = NCT2013PatchesDataset(data_path)
    patient_ids = ds.metadata_table["patient_id"].unique().tolist()

    def transform(image, metadata):
        image = image.convert("RGB")
        image = T.Resize((input_size, input_size))(image)
        image = T.ToTensor()(image)
        image = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(
            image
        )

        cancer = metadata["grade"] != "Benign"
        cancer = torch.tensor(cancer).long()
        patient_id = metadata["patient_id"]
        patient_id_idx = patient_ids.index(patient_id)
        group_id = torch.tensor(patient_id_idx).long()

        return group_id, image, cancer

    ds.transform = transform

    if dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(ds)
        full_loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False, sampler=sampler
        )
    else:
        full_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    return KFoldLinearProbing(full_loader, device, n_splits=5)


class SSLEvaluatorNCT:
    def __init__(self, conf):
        if conf.evaluation.do_nct_probing:
            logging.info(f"Setting up NCT probing...")
            if conf.evaluation.get("nct_probing_train_patients"):
                self.nct_probe = NCTProbing(
                    conf.evaluation.nct_probing_train_patients,
                    conf.evaluation.nct_probing_val_patients,
                    image_size=conf.transform.global_crops_size,
                    mean=conf.transform.mean,
                    std=conf.transform.std,
                    **conf.evaluation.get("nct_probing_kwargs", {}),
                )
            else:
                self.nct_probe = KFoldNCTProbing(
                    batch_size=conf.batch_size_per_gpu,
                    image_size=conf.transform.global_crops_size,
                    mean=conf.transform.mean,
                    std=conf.transform.std,
                    **conf.evaluation.get("nct_probing_kwargs", {}),
                )
        else:
            self.nct_probe = None

        build_linear_probe = lambda: build_kfold_linear_probe_for_nct_patches(
            data_path=conf.nct_patches_data_path,
            input_size=conf.transform.global_crops_size,
            batch_size=conf.batch_size_per_gpu,
            device=torch.device("cuda"),
        )
        if conf.evaluation.do_clstoken_nct_probing:
            logging.info(f"Setting up patch NCT probing...")
            self.nct_clstoken_probe = build_linear_probe()
        else:
            self.nct_clstoken_probe = None

        if conf.evaluation.do_register_probing:
            logging.info(f"Setting up probing for register token")
            self.register_token_probe = build_linear_probe()
        else:
            self.register_token_probe = None

    def __call__(
        self,
        model,
        epoch,
    ):
        logging.info(f"Running NCT Probing")
        probing_results = {}

        vit = model["vit"].backbone
        cnn = model["cnn"].backbone

        class FeatureMapVitWrapper(nn.Module): 
            def __init__(self, vit_model):
                super().__init__()
                self.vit = vit_model

            def forward(self, x):
                return self.vit.get_feature_map(x)

        class ClassTokenVitWrapper(nn.Module): 
            def __init__(self, vit_model):
                super().__init__()
                self.vit = vit_model

            def forward(self, x, return_all_tokens=False):
                return self.vit(x, return_all_tokens=return_all_tokens)[:, 0, :]

        vit_for_feature_maps = FeatureMapVitWrapper(vit)    
        vit_for_clstoken = ClassTokenVitWrapper(vit)

        if self.nct_probe is not None:
            logging.info(f"Running NCT Probing")
            self.nct_probe.model = vit_for_feature_maps            
            outputs = self.nct_probe.run()
            return outputs
                
                
            train_metrics, val_metrics = outputs
            self._add_metrics_to_dict(
                probing_results, train_metrics, "train", "probing"
            )
            self._add_metrics_to_dict(
                probing_results, val_metrics, "val", "probing"
            )

        def get_class_token(model, im):
            tokens = model(im, return_all_tokens=True)
            return tokens[:, 0, :]

        if self.nct_clstoken_probe is not None:
            outputs = self.nct_clstoken_probe.run_probing(
                model, get_class_token, is_main_process=is_main_process()  # type: ignore
            )
            if outputs is not None:
                train_metrics, val_metrics = outputs
                self._add_metrics_to_dict(
                    probing_results, train_metrics, "train", "clstoken_probing_kfold"
                )
                self._add_metrics_to_dict(
                    probing_results, val_metrics, "val", "clstoken_probing_kfold"
                )

        def get_register_token(model, im):
            tokens = model(im, return_all_tokens=True)
            return tokens[:, 1, :]

        if self.register_token_probe is not None:
            outputs = self.register_token_probe.run_probing(
                model, get_register_token, is_main_process=utils.is_main_process()  # type: ignore
            )
            if outputs is not None:
                train_metrics, val_metrics = outputs
                self._add_metrics_to_dict(
                    probing_results, train_metrics, "train", "regtoken_probing_kfold"
                )
                self._add_metrics_to_dict(
                    probing_results, val_metrics, "val", "regtoken_probing_kfold"
                )

        return probing_results

    def _add_metrics_to_dict(self, d, metrics, split, name):
        d.update({f"{split}_{k}_{name}": v for k, v in metrics.items()})


class CIFAR10SSLEvaluator:
    def __init__(self, conf):

        from src.datasets.cifar10 import CIFAR10Subsample

        transform = T.Compose(
            [
                T.Resize(
                    (conf.transform.global_crops_size, conf.transform.global_crops_size)
                ),
                T.ToTensor(),
                T.Normalize(mean=conf.transform.mean, std=conf.transform.std),
            ]
        )
        train_ds = CIFAR10Subsample(
            root="data",
            train=True,
            transform=transform,
            download=True,
            n_samples_per_class=1000,
        )
        val_ds = CIFAR10Subsample(
            root="data", train=False, transform=transform, download=True
        )
        if dist.is_initialized():
            train_sampler = DistributedSampler(train_ds)
            val_sampler = DistributedSampler(val_ds)
        else:
            train_sampler = None
            val_sampler = None
        self.train_loader = DataLoader(
            train_ds,
            batch_size=conf.batch_size_per_gpu,
            shuffle=True if train_sampler is None else False,
            sampler=train_sampler,
            num_workers=conf.num_workers,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=conf.batch_size_per_gpu,
            shuffle=False if val_sampler is None else False,
            sampler=val_sampler,
            num_workers=conf.num_workers,
        )
        self.device = torch.device("cuda")

    def __call__(
        self,
        model,
        epoch,
    ):
        rank = dist.get_rank() if dist.is_initialized() else 0
        tqdm_kwargs = dict(desc=f"Probing CIFAR10 epoch {epoch}", disable=rank != 0)

        def forward(model, data, device):
            image, label = data
            image = image.to(device)

            vit = model["vit"].backbone
            cnn = model["cnn"].backbone

            embeddings_dict = {}

            vit_output = vit(image.to(device), return_all_tokens=True)
            cls_tokens = vit_output[:, : vit.n_cls_tokens, :]
            patch_tokens = vit_output[:, vit.n_cls_tokens :, :]

            for i, cls_token in enumerate(cls_tokens.unbind(1)):
                if i == 0:
                    embeddings_dict["cls"] = cls_token
                else:
                    embeddings_dict[f"cls_{i}"] = cls_token
            pooled_patch_tokens = patch_tokens.mean(dim=1)
            embeddings_dict["patch"] = pooled_patch_tokens

            cnn_output = cnn(image.to(device))
            embeddings_dict["cnn"] = cnn_output

            return {"embedding": embeddings_dict, "label": label}

        probe = SKLearnLinearProbing(
            model,
            self.train_loader,
            self.val_loader,
            self.device,
            tqdm_kwargs=tqdm_kwargs,
            forward_fn=forward,
        )
        metrics = probe.run()
        return metrics


class SKLearnProbingSSLEvaluator:
    def __init__(self, conf):
        from .build_dataloader import get_supervised_dataloaders_from_config

        loaders = get_supervised_dataloaders_from_config(conf)
        self.conf = conf
        self.train_loader = loaders["train"]
        self.val_loader = loaders["val"]
        self.device = torch.device("cuda")

    def __call__(
        self,
        model,
        epoch,
    ):
        rank = dist.get_rank() if dist.is_initialized() else 0
        tqdm_kwargs = dict(
            desc=f"Probing {type(self.train_loader.dataset).__name__} epoch {epoch}",
            disable=rank != 0,
        )

        def forward(model, data, device):
            image, label = data
            image = image.to(device)

            vit = model["vit"].backbone
            cnn = model["cnn"].backbone

            embeddings_dict = {}

            vit_output = vit(image.to(device), return_all_tokens=True)
            cls_tokens = vit_output[:, : vit.n_cls_tokens, :]
            patch_tokens = vit_output[:, vit.n_cls_tokens :, :]

            for i, cls_token in enumerate(cls_tokens.unbind(1)):
                if i == 0:
                    embeddings_dict["cls"] = cls_token
                else:
                    embeddings_dict[f"cls_{i}"] = cls_token
            pooled_patch_tokens = patch_tokens.mean(dim=1)
            embeddings_dict["patch"] = pooled_patch_tokens

            cnn_output = cnn(image.to(device))
            embeddings_dict["cnn"] = cnn_output

            return {"embedding": embeddings_dict, "label": label}

        regression_kwargs = dict(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=self.conf.data.num_workers,
        )
        regression_kwargs.update(self.conf.evaluation.get("regression_kwargs", {}))

        metrics_fn = None
        if self.conf.evaluation.get("add_binary_clf_metrics"): 
            metrics_fn = lambda output, target: compute_binary_classification_metrics(
                output[:, 1], target                
            )

        probe = SKLearnLinearProbing(
            model,
            self.train_loader,
            self.val_loader,
            self.device,
            tqdm_kwargs=tqdm_kwargs,
            forward_fn=forward,
            max_samples_for_fit=self.conf.evaluation.get("max_samples_for_fit", None),
            regression_kwargs=regression_kwargs,
            metrics_fn=metrics_fn
        )
        metrics = probe.run()
        return metrics


class DisabledSSLEvaluator:
    def __init__(self, conf):
        pass

    def __call__(self, model, epoch):
        return {}


class ProbingSSLEvaluatorForiBOT:
    def __init__(self, probes): 
        self.probes = probes

    def __call__(self, model, epoch):
        vit = model['vit'].backbone

        original_vit_output_format = vit.output_format
        vit.output_format = 'dict'

        metrics = {}
        for probe_name, probe in self.probes.items():
            probe.model = vit
            probe_metrics = probe.run()
            metrics.update({f"{probe_name}/{k}": v for k, v in probe_metrics.items()})

        vit.output_format = original_vit_output_format
        return metrics



