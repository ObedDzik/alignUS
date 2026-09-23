from collections import defaultdict
from typing import Any, Callable
import numpy as np
import torch
from torch import distributed as dist
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


def concat_all_gather(tensors, dim=0):
    """
    Performs allgather operation on the provided tensors.

    """
    if not dist.is_initialized():
        return tensors

    # Get the world size and rank of the current process
    out_device = tensors.device
    tensors = tensors.to("cuda")
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Allgather the tensors from all processes
    output_tensors = [torch.empty_like(tensors) for _ in range(world_size)]
    dist.all_gather(output_tensors, tensors, async_op=False)
    output_tensors = torch.cat(output_tensors, dim=dim)
    output_tensors = output_tensors.to(out_device)

    return output_tensors


class SKLearnLinearProbing:
    """
    Linear probing using sklearn.

    Args:
        model: model to probe (nn.Module)
        train_loader: train loader (torch.utils.data.DataLoader)
        val_loader: validation loader (torch.utils.data.DataLoader)
        device: device to run the model on (torch.device)
        tqdm_kwargs: kwargs for tqdm
        forward_fn: function which takes model, data (batch of data from the train/val loader) and device and returns the output of the model as a dictionary.
            This function returns a dictionary with keys "embedding" and "label". The label is the ground truth label of the data.
            The "embedding" can be a tensor with emedding values or a dictionary with keys as the layer names and values as the embedding tensors.
            If None, the model's forward_data method will be used.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        tqdm_kwargs=dict(desc="Probing"),
        forward_fn: Callable[[nn.Module, Any, str], dict] | None = None,
        max_samples_for_fit=None,
        regression_kwargs=dict(),
        sep='_',
        metrics_fn=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.tqdm_kwargs = tqdm_kwargs
        self.forward_fn = forward_fn
        self.max_samples_for_fit = max_samples_for_fit
        self.regression_kwargs = regression_kwargs
        self.sep = sep
        self.metrics_fn = metrics_fn

    def run(self):
        self.model.eval()

        X_train, y_train = self.extract_embeddings(self.train_loader)
        X_val, y_val = self.extract_embeddings(self.val_loader)
        X_train, y_train = self.subsample_data(X_train, y_train)

        metrics = {}

        for key in X_train.keys():
            logger.info(f"Training model for {key}")

            kw = dict(max_iter=10000, class_weight="balanced")
            kw.update(self.regression_kwargs)
            self.clf = LogisticRegression(**kw)
            logger.info(
                f"Training model for {key} with {kw}, num_samples: {X_train[key].shape[0]}"
            )
            self.clf.fit(X_train[key], y_train)

            y_hat_train = self.clf.predict_proba(X_train[key])
            y_hat_val = self.clf.predict_proba(X_val[key])

            train_acc = self.clf.score(X_train[key], y_train)
            val_acc = self.clf.score(X_val[key], y_val)
            
            train_metrics = {'acc': train_acc}
            val_metrics = {'acc': val_acc}

            if self.metrics_fn: 
                train_metrics.update(self.metrics_fn(y_hat_train, y_train))
                val_metrics.update(self.metrics_fn(y_hat_val, y_val))
        
            metrics.update(
                {f"train{self.sep}{m_key}{self.sep}{key}": m_value for m_key, m_value in train_metrics.items()}
            )
            metrics.update(
                {f"val{self.sep}{m_key}{self.sep}{key}": m_value for m_key, m_value in val_metrics.items()}
            )

        return metrics

    def subsample_data(self, X, y):
        total_samples = len(y)

        if self.max_samples_for_fit is None:
            return X, y
        elif total_samples <= self.max_samples_for_fit:
            return X, y

        rng = np.random.RandomState(42)
        indices_keep = rng.choice(
            total_samples, self.max_samples_for_fit, replace=False
        )

        X = {k: v[indices_keep] for k, v in X.items()}
        y = y[indices_keep]

        return X, y

    @torch.no_grad()
    def extract_embeddings(self, loader):
        embeddings = defaultdict(list)
        labels = []

        for data in tqdm(loader, **self.tqdm_kwargs):
            if self.forward_fn is not None:
                data = self.forward_fn(self.model, data, self.device)
            else:
                data = self.model.forward_data(data, device=self.device)

            if isinstance(data["embedding"], dict):
                for k, v in data["embedding"].items():
                    embeddings[k].append(v.cpu())
            else:
                embeddings["cls"].append(data["embedding"].cpu())

            labels.append(data["label"].cpu())

        for k, v in embeddings.items():
            embeddings[k] = torch.cat(v, dim=0)
            embeddings[k] = concat_all_gather(embeddings[k])
            embeddings[k] = embeddings[k].cpu().numpy()

        labels = torch.cat(labels, dim=0)
        labels = concat_all_gather(labels)
        labels = labels.cpu().numpy()

        return embeddings, labels
