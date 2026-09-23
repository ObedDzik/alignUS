from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import torch
from medAI.metrics import calculate_binary_classification_metrics
import torch
from medAI.factories.optimizer import build_optimizer_v0
from ema_pytorch import EMA
import logging 


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PatchClassificationTrainer:
    """Trainer class for patch-based classification models.

    Args:
        model: The PyTorch model to be trained. It should accept B x C x H x W input (Batch of patches) and return B x num_classes.
        train_loader: DataLoader for training data. Each batch should be a dict with keys 'patches' (B x N x C x H x W tensor) and "label" (B tensor of core-level labels).
        loss_fn: Loss function to use. Default is CrossEntropyLoss.
        optimizer: Optimizer for training. If None, it will be created based on optimizer_cfg.
        lr_scheduler: Learning rate scheduler. Default is None.
        val_loader: DataLoader for validation data. Same format as train_loader. Default is None.
        device: Device to run training on. Default is 'cuda'.
        logger: Optional logger function to log metrics. Default is None.
        run_dir: Directory to save checkpoints and logs. Default is None.
        optimizer_cfg: Configuration dict for building the optimizer if optimizer is None. Default is None.
        epochs: Number of training epochs. Default is 10.
        use_ema_model_for_validation: Whether to use an EMA model for validation. Default is False.
        ema_cfg: Configuration dict for EMA model if use_ema_model_for_validation is True. Default is None.
    """

    _label_key = "label"

    def __init__(
        self,
        model,
        train_loader,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=None,
        lr_scheduler=None,
        val_loader=None,
        device="cuda",
        logger=None,
        run_dir=None,
        optimizer_cfg={},
        epochs=10,
        use_ema_model_for_validation=False,
        ema_cfg=None, 
        patches_key="patches",
        label_key="label",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        if hasattr(self.loss_fn, 'to'):
            self.loss_fn = self.loss_fn.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logger = logger
        self.lr_scheduler = lr_scheduler
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.optimizer_cfg = optimizer_cfg
        self.epochs = epochs
        self.use_ema_model_for_validation = use_ema_model_for_validation
        self.ema_cfg = ema_cfg
        self.patches_key = patches_key
        self.label_key = label_key

        self.setup()

    def setup(self):
        logger.info("Setting up PatchClassificationTrainer...")
        if self.run_dir is not None:
            (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        if self.optimizer is None:
            self.optimizer, self.lr_scheduler = build_optimizer_v0(
                self.model, len(self.train_loader), **self.optimizer_cfg, batch_size_per_gpu=self.train_loader.batch_size
            )
        if self.use_ema_model_for_validation:
            self.ema_model = EMA(self.model, **(self.ema_cfg or {}))
        else: 
            self.ema_model = None

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(self.train_loader, desc="Training"):
            B, N, C, H, W = batch[self.patches_key].shape
            patches = batch[self.patches_key].reshape(B * N, C, H, W).to(self.device)
            targets = batch[self.label_key].repeat_interleave(N, dim=0).to(self.device)

            self.optimizer.zero_grad()
            patch_predictions = self.model(patches)
            loss = self.loss_fn(
                patch_predictions.reshape(B * N, -1), targets.reshape(B * N)
            )
            loss.backward()
            self.optimizer.step()
            if self.lr_scheduler:
                self.lr_scheduler.step()
            if self.ema_model is not None:
                self.ema_model.update()

            total_loss += loss.item()
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate_one_epoch(self, loader=None):
        loader = loader or self.val_loader
        model = self.ema_model.ema_model if self.use_ema_model_for_validation else self.model
        model.eval()
        total_loss = 0.0
        validation_results = defaultdict(list)

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                B, N, C, H, W = batch[self.patches_key].shape
                patches = batch[self.patches_key].reshape(B * N, C, H, W).to(self.device)
                targets = batch[self.label_key].repeat_interleave(N, dim=0).to(self.device)

                patch_predictions = model(patches)
                loss = self.loss_fn(
                    patch_predictions.reshape(B * N, -1), targets.reshape(B * N)
                )
                total_loss += loss.item()

                aggregated_predictions = patch_predictions.softmax(-1).reshape(B, N, -1).mean(dim=1)
                aggregated_targets = batch[self.label_key].to(self.device)
                validation_results["core_level_preds"].append(
                    aggregated_predictions.cpu()
                )
                validation_results["core_level_targets"].append(
                    aggregated_targets.cpu()
                )

        for key in validation_results:
            validation_results[key] = (
                torch.cat(validation_results[key], dim=0).numpy()
                if isinstance(validation_results[key][0], torch.Tensor)
                else validation_results[key]
            )

        metrics = calculate_binary_classification_metrics(
            validation_results["core_level_preds"][:, 1],
            validation_results["core_level_targets"],
        )
        metrics["val_loss"] = total_loss / len(self.val_loader)

        return metrics, validation_results

    def train(self):
        self.setup()
        for epoch in range(self.epochs):
            train_loss = self.train_one_epoch()
            self.log_metrics({"train/loss": train_loss, "epoch": epoch + 1})
            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")

            if self.val_loader is not None:
                metrics, _ = self.validate_one_epoch()
                print(
                    f"Epoch {epoch + 1}, Validation Metrics: auc={metrics['auc']:.4f}, val_loss={metrics['val_loss']:.4f}"
                )
                self.log_metrics(
                    {
                        **{f"val/{k}": v for k, v in metrics.items()},
                        "epoch": epoch + 1,
                    }
                )

    def log_metrics(self, metrics: dict):
        if self.logger is None:
            return
        self.logger(metrics)
