import logging
from pathlib import Path
from collections import defaultdict

import torch
from tqdm import tqdm

from medAI.engine.prostnfound.prostnfound_model_wrapper import ProstNFoundModelWrapper
from medAI.engine.prostnfound.trainer import (
    setup_optimizers_and_schedulers,
)
from medAI.engine.prostnfound.evaluator import (
    ProstNFoundEvaluator as Evaluator,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ProstNFoundTrainer:
    """Trainer for ProstNFound-style models with an interface similar to PatchClassificationTrainer.

    Args:
        model: PyTorch model. If not already a ProstNFoundModelWrapper, it will be wrapped.
        train_loader: DataLoader yielding dict batches (expects keys like 'bmode', 'prostate_mask', 'needle_mask', optional 'rf', prompts).
        loss_fn: Callable taking the model output batch dict and returning a scalar loss tensor.
        optimizer: Optional torch.optim. If None, will be built via setup_optimizers_and_schedulers using optimizer_cfg.
        lr_scheduler: Optional scheduler. If None and optimizer is None, built alongside optimizer.
        val_loader: Optional DataLoader for validation.
        device: Device string.
        logger: Optional callable taking a dict of metrics.
        run_dir: Optional directory to create/checkpoints.
        optimizer_cfg: Dict forwarded to setup_optimizers_and_schedulers (e.g., lr, encoder_lr, cnn_lr, wd, schedule, warmup_epochs).
        epochs: Number of epochs.
        use_ema_model_for_validation: Placeholder for API parity (not used).
        ema_cfg: Placeholder for API parity (not used).
        use_amp: Enable mixed precision.
        accumulate_grad_steps: Gradient accumulation steps.
    """

    def __init__(
        self,
        model,
        train_loader,
        loss_fn,
        optimizer=None,
        lr_scheduler=None,
        val_loader=None,
        device="cuda",
        logger=None,
        run_dir=None,
        optimizer_cfg=None,
        epochs=10,
        use_ema_model_for_validation=False,
        ema_cfg=None,
        use_amp=True,
        accumulate_grad_steps=1,
    ):
        self.device = device
        # Wrap model if needed
        if isinstance(model, ProstNFoundModelWrapper):
            self.model = model.to(device)
        else:
            self.model = ProstNFoundModelWrapper(model).to(device)

        self.loss_fn = loss_fn.to(device) if hasattr(loss_fn, "to") else loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.logger = logger
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.optimizer_cfg = optimizer_cfg or {}
        self.epochs = epochs
        self.use_ema_model_for_validation = use_ema_model_for_validation
        self.ema_cfg = ema_cfg
        self.use_amp = use_amp
        self.accumulate_grad_steps = max(1, int(accumulate_grad_steps))
        self.scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and torch.cuda.is_available()))

        self.setup()

    def setup(self):
        logger.info("Setting up ProstNFoundTrainer...")
        if self.run_dir is not None:
            (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

        # Optimizer/scheduler
        if self.optimizer is None:
            self.optimizer, self.lr_scheduler = setup_optimizers_and_schedulers(
                epochs=self.epochs,
                wrapped_model=self.model,
                train_loader=self.train_loader,
                **self.optimizer_cfg,
            )

    def _move_batch_to_device(self, batch):
        # Move tensor entries onto device, keep non-tensors as-is
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)
        return batch

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        evaluator = Evaluator(log_images=False)

        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(self.train_loader, desc="Training")):
            batch = self._move_batch_to_device(batch)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                out = self.model(batch)  # returns augmented dict with 'cancer_logits' etc.
                loss = self.loss_fn(out)

            loss = loss / self.accumulate_grad_steps
            total_loss += loss.item()

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % self.accumulate_grad_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            if self.lr_scheduler is not None:
                # ProstNFound schedulers are typically stepped per-iteration
                self.lr_scheduler.step()

            # accumulate step metrics (no logging here to keep parity with PatchClassificationTrainer)
            evaluator(out)

        avg_loss = total_loss / max(1, len(self.train_loader))
        # Optionally compute aggregate train metrics (AUC, etc.)
        train_metrics = {f"train/{k}": v for k, v in evaluator.aggregate_metrics().items()}
        train_metrics["train/loss"] = avg_loss
        self.log_metrics(train_metrics)

        return avg_loss

    @torch.no_grad()
    def validate_one_epoch(self, loader=None):
        loader = loader or self.val_loader
        if loader is None:
            return {}, {}

        self.model.eval()
        total_loss = 0.0
        evaluator = Evaluator(log_images=False)

        for batch in tqdm(loader, desc="Validation"):
            batch = self._move_batch_to_device(batch)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                out = self.model(batch)
                val_loss = self.loss_fn(out)

            total_loss += float(val_loss.item())
            evaluator(out)

        metrics = evaluator.aggregate_metrics()
        metrics["val_loss"] = total_loss / max(1, len(loader))

        # For API parity: return metrics and raw evaluation buffers if needed
        validation_results = {}  # can be extended to return raw preds/targets if desired

        return metrics, validation_results

    def train(self):
        self.setup()
        for epoch in range(self.epochs):
            train_loss = self.train_one_epoch()
            self.log_metrics({"train/loss": train_loss, "epoch": epoch + 1})
            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")

            if self.val_loader is not None:
                metrics, _ = self.validate_one_epoch()
                auc = metrics.get("auc", metrics.get("val/auc", float("nan")))
                val_loss = metrics.get("val_loss", float("nan"))
                print(f"Epoch {epoch + 1}, Validation Metrics: auc={auc:.4f}, val_loss={val_loss:.4f}")
                self.log_metrics({**{f"val/{k}": v for k, v in metrics.items()}, "epoch": epoch + 1})

    def log_metrics(self, metrics: dict):
        if self.logger is None:
            return
        self.logger(metrics)