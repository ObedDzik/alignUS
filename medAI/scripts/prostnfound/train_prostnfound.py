"""
Original monolithic training script for ProstNFound model.

#TODO refactor to use medAI engine
"""

import argparse
import json
import logging
import os
from tempfile import mkdtemp
import typing as tp
from argparse import ArgumentParser
from warnings import warn
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import wandb
from matplotlib import pyplot as plt
from medAI.modeling.prostnfound import ProstNFound
from medAI.modeling.setr import SETR
from torch.nn import functional as F
from tqdm import tqdm
from torchvision.transforms import v2 as T

from medAI.modeling import *
from medAI.utils.distributed import is_main_process
from medAI.utils.reproducibility import (
    get_all_rng_states,
    set_all_rng_states,
    set_global_seed,
)
from medAI.losses.prostnfound import (
    build_loss,
)
from medAI.layers.masked_prediction_module import (
    MaskedPredictionModule,
)
from medAI.factories.prostnfound.dataloaders_v0 import get_dataloaders_from_args
from medAI.factories.prostnfound.models import get_model
from medAI.engine.prostnfound.evaluator import (
    ProstNFoundEvaluator as Evaluator,
)


# @hydra.main(
#     config_path="pkg://medAI/configs/prostnfound",
#     config_name="default",
#     version_base=None,
# )
def main(cfg):

    if cfg.get("output_dir") is not None:
        os.makedirs(cfg.output_dir, exist_ok=True)
        OmegaConf.save(
            cfg, os.path.join(cfg.output_dir, "train_config.yaml"), resolve=True
        )

    # setup
    handlers = [logging.StreamHandler()]
    if cfg.output_dir is not None and is_main_process():
        file_handler = logging.FileHandler(os.path.join(cfg.output_dir, "training.log"))
        handlers.append(file_handler)

    logging.basicConfig(
        level=logging.INFO if not cfg.debug else logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )

    logging.info("Setting up experiment")
    wandb.init(
        config=OmegaConf.to_object(cfg),
        **cfg.get("wandb", {}),
    )
    _tmpdir = mkdtemp()
    OmegaConf.save(cfg, os.path.join(_tmpdir, "train_config.yaml"), resolve=True)
    wandb.save(
        os.path.join(_tmpdir, "train_config.yaml"), base_path=_tmpdir, policy="now"
    )
    # cfg.wandb_url = wandb.run.url if wandb.run else None

    def log_fn(data: dict):

        if wandb.run is not None:
            data_wandb = {}
            for k, v in data.items():
                if isinstance(v, plt.Figure):
                    data_wandb[k] = wandb.Image(v)
                else:
                    data_wandb[k] = v
            wandb.log(data_wandb)
        if cfg.get("output_dir") is not None:
            metrics_path = os.path.join(cfg.output_dir, "metrics.jsonl")

            figures = {k: v for k, v in data.items() if isinstance(v, plt.Figure)}
            scalars = {k: v for k, v in data.items() if not k in figures}

            with open(metrics_path, "a") as f:
                f.write(json.dumps(scalars) + "\n")

            for k, fig in figures.items():
                fig_dir = os.path.join(cfg.output_dir, "figures", k)
                os.makedirs(fig_dir, exist_ok=True)
                index = len(os.listdir(fig_dir))
                fig_path = os.path.join(fig_dir, f"{index:05d}.png")
                fig.savefig(fig_path)
                plt.close(fig)

    if cfg.checkpoint_dir is not None:
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        exp_state_path = os.path.join(cfg.checkpoint_dir, "experiment_state.pth")
        if os.path.exists(exp_state_path):
            logging.info("Loading experiment state from experiment_state.pth")
            state = torch.load(exp_state_path)
        else:
            logging.info("No experiment state found - starting from scratch")
            state = None
    else:
        state = None

    set_global_seed(cfg.seed)

    logging.info("Setting up model")

    model = get_model(OmegaConf.to_object(cfg))
    model = ProstNFoundMeta(model, **cfg.get("metamodel", {}))

    model.to(cfg.device)
    if cfg.torch_compile:
        torch.compile(model)
    logging.info("Model setup complete")
    logging.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    logging.info(
        f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    if cfg.model_checkpoint:
        model_state = torch.load(cfg.model_checkpoint, map_location="cpu")
        if "model" in model_state:
            model_state = model_state["model"]
        msg = model.load_state_dict(model_state, strict=False)
        logging.info(f"Loaded model from {cfg.model_checkpoint} with message `{msg}`.")
    if state is not None:
        model.load_state_dict(state["model"])

    # setup criterion
    if "pos_weight" not in cfg:
        cfg.pos_weight = 1.0
    criterion = build_loss(cfg)

    loaders = get_dataloaders_from_args(cfg.data)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    optimizer, lr_scheduler = setup_optimizer(cfg, model, train_loader)
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        lr_scheduler.load_state_dict(state["lr_scheduler"])

    scaler = torch.cuda.amp.GradScaler()
    if state is not None:
        scaler.load_state_dict(state["gradient_scaler"])

    epoch = 0 if state is None else state["epoch"]
    logging.info(f"Starting at epoch {epoch}")
    best_score = 0 if state is None else state["best_score"]
    logging.info(f"Best score so far: {best_score}")
    if state is not None:
        rng_state = state["rng"]
        set_all_rng_states(rng_state)

    def get_state():
        return {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "gradient_scaler": scaler.state_dict(),
            "rng": get_all_rng_states(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "args": vars(cfg),
        }

    def save_checkpoint(name):
        state = get_state()
        if cfg.checkpoint_dir is not None:
            logging.info(f"Saving experiment snapshot to {cfg.checkpoint_dir}")
            torch.save(state, os.path.join(cfg.checkpoint_dir, name))
            if cfg.save_checkpoint_wandb:
                wandb.save(
                    os.path.join(cfg.checkpoint_dir, name),
                    base_path=cfg.checkpoint_dir,
                    policy="now",
                )

    for epoch in range(epoch, cfg.epochs):
        if cfg.cutoff_epoch is not None and epoch > cfg.cutoff_epoch:
            break
        logging.info(f"Epoch {epoch}")

        save_checkpoint("experiment_state.pth")

        run_train_epoch(
            cfg,
            model,
            train_loader,
            criterion,
            optimizer,
            lr_scheduler,
            scaler,
            epoch,
            desc="train",
            log_fn=log_fn,
        )

        if cfg.run_val:
            val_metrics, results_table = run_eval_epoch(
                cfg, model, val_loader, epoch, desc="val", log_fn=log_fn
            )

            if is_main_process() and cfg.output_dir is not None:
                table_path = os.path.join(
                    cfg.output_dir, f"val_results_epoch_{epoch:04d}.csv"
                )
                results_table.to_csv(table_path)

            if val_metrics is not None:
                tracked_metric = val_metrics[cfg.tracked_metric]
                new_record = tracked_metric > best_score
            else:
                new_record = None

            if new_record:
                best_score = tracked_metric
                logging.info(f"New best score: {best_score}")

            if cfg.run_test and new_record or cfg.test_every_epoch:
                logging.info("Running test set")
                metrics = run_eval_epoch(
                    cfg, model, test_loader, epoch, desc="test", log_fn=log_fn
                )

            if new_record and cfg.save_best_weights:
                save_checkpoint("best.pth")

    logging.info("Finished training")


def run_train_epoch(
    args,
    model,
    loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    epoch,
    desc="Train",
    log_fn=None,
):
    # setup epoch
    model.train()
    evaluator = Evaluator(**args.evaluator)

    for train_iter, data in enumerate(tqdm(loader, desc=desc)):

        if args.debug and train_iter > 10:
            break

        # run the model
        with torch.cuda.amp.autocast(enabled=args.use_amp):

            data = model(data)  # heatmap

            if torch.any(torch.isnan(data["cancer_logits"])):
                logging.warning("NaNs in heatmap logits")

            # loss calculation
            loss = criterion(data)

        loss = loss / args.accumulate_grad_steps
        # backward pass
        if args.use_amp:
            logging.debug("Backward pass")
            scaler.scale(loss).backward()
        else:
            logging.debug("Backward pass")
            loss.backward()

        # gradient accumulation and optimizer step
        if args.debug:
            for param in optimizer.param_groups[1]["params"]:
                break
            logging.debug(param.data.view(-1)[0])

        if (train_iter + 1) % args.accumulate_grad_steps == 0:
            logging.debug("Optimizer step")
            if args.use_amp:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()

            if args.debug:
                for param in optimizer.param_groups[1]["params"]:
                    break
                logging.debug(param.data.view(-1)[0])

        scheduler.step()

        # accumulate outputs
        step_metrics = {f"train/{k}": v for k, v in evaluator(data).items()}

        # log metrics
        step_metrics.update({"train_loss": loss.item() / args.accumulate_grad_steps})
        encoder_lr = optimizer.param_groups[0]["lr"]
        main_lr = optimizer.param_groups[1]["lr"]
        cnn_lr = optimizer.param_groups[2]["lr"]
        step_metrics["encoder_lr"] = encoder_lr
        step_metrics["main_lr"] = main_lr
        step_metrics["cnn_lr"] = cnn_lr

        if log_fn is not None:
            log_fn(step_metrics)

    # compute and log metrics
    metrics = evaluator.aggregate_metrics()
    desc = "train"
    metrics = {f"{desc}/{k}": v for k, v in metrics.items()}
    metrics["epoch"] = epoch
    if log_fn is not None:
        log_fn(metrics)


@torch.no_grad()
def run_eval_epoch(args, model, loader, epoch, desc="eval", log_fn=None):
    model.eval()

    evaluator = Evaluator(**args.evaluator)

    for train_iter, data in enumerate(tqdm(loader, desc=desc)):

        with torch.cuda.amp.autocast(enabled=args.use_amp):
            data = model(data)

        # accumulate outputs
        step_metrics = {f"{desc}/{k}": v for k, v in evaluator(data).items()}
        if step_metrics and log_fn is not None:
            log_fn(step_metrics)

    metrics = evaluator.aggregate_metrics()
    results_table = evaluator.results_table
    metrics = {f"{desc}/{k}": v for k, v in metrics.items()}
    metrics["epoch"] = epoch
    if log_fn is not None:
        log_fn(metrics)

    return metrics, results_table


def setup_optimizer(args, model, train_loader):
    from torch.optim import AdamW

    (
        encoder_parameters,
        warmup_parameters,
        cnn_parameters,
    ) = model.get_params_groups()

    total_epochs = args.epochs
    encoder_frozen_epochs = args.warmup_epochs
    warmup_epochs = 5
    niter_per_ep = len(train_loader)
    warmup_lr_factor = args.warmup_lr / args.lr
    params = [
        {"params": encoder_parameters, "lr": args.encoder_lr},
        {"params": warmup_parameters, "lr": args.lr},
        {"params": cnn_parameters, "lr": args.cnn_lr},
    ]

    def compute_lr_multiplier(iter, is_encoder_or_cnn=True):
        schedule = args.get("scheduler", "cosine")
        if schedule == "constant":
            return 1

        if iter < encoder_frozen_epochs * niter_per_ep:
            if is_encoder_or_cnn:
                return 0
            else:
                if iter < warmup_epochs * niter_per_ep:
                    return (iter * warmup_lr_factor) / (warmup_epochs * niter_per_ep)
                else:
                    cur_iter_in_frozen_phase = iter - warmup_epochs * niter_per_ep
                    total_iter_in_frozen_phase = (
                        encoder_frozen_epochs - warmup_epochs
                    ) * niter_per_ep
                    return (
                        0.5
                        * (
                            1
                            + np.cos(
                                np.pi
                                * cur_iter_in_frozen_phase
                                / (total_iter_in_frozen_phase)
                            )
                        )
                        * warmup_lr_factor
                    )
        else:
            iter -= encoder_frozen_epochs * niter_per_ep
            if iter < warmup_epochs * niter_per_ep:
                return iter / (warmup_epochs * niter_per_ep)
            else:
                cur_iter = iter - warmup_epochs * niter_per_ep
                total_iter = (
                    total_epochs - warmup_epochs - encoder_frozen_epochs
                ) * niter_per_ep
                return 0.5 * (1 + np.cos(np.pi * cur_iter / total_iter))

    optimizer = AdamW(params, lr=args.lr, weight_decay=args.wd)
    from torch.optim.lr_scheduler import LambdaLR

    lr_scheduler = LambdaLR(
        optimizer,
        [
            lambda iter: compute_lr_multiplier(iter, is_encoder_or_cnn=True),
            lambda iter: compute_lr_multiplier(iter, is_encoder_or_cnn=False),
            lambda iter: compute_lr_multiplier(iter, is_encoder_or_cnn=True),
        ],
    )

    return optimizer, lr_scheduler


class ProstNFoundMeta(nn.Module):
    """Wraps a model to perform forward pass with ProstNFound style training

    Args:
        model: The model to wrap.
        mask_output_key: The key to use for the mask output (if the model outputs a dictionary of tensors)
    """

    def __init__(self, model: nn.Module, mask_output_key=None):
        super().__init__()
        self.model = model
        self.mask_output_key = mask_output_key

        if isinstance(self.model, ProstNFound):
            logging.info(f"Model ProstNFound with prompts {self.model.prompts}")

        self.register_buffer("temperature", torch.tensor([1.0]))
        self.register_buffer("bias", torch.tensor([0.0]))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, data, include_postprocessed_heatmaps=False):
        # extracting relevant data from the batch
        bmode = data["bmode"].to(self.device)
        needle_mask = data["needle_mask"].to(self.device)
        prostate_mask = data["prostate_mask"].to(self.device)
        if "rf" in data:
            rf = data["rf"].to(self.device)
        else:
            rf = None

        B = len(bmode)

        # Wrapped forward pass
        if isinstance(self.model, ProstNFound):
            prompts = {}
            for prompt_name in self.model.prompts:
                prompts[prompt_name] = data[prompt_name].to(
                    device=self.device, dtype=bmode.dtype
                )
                if prompts[prompt_name].ndim == 1:
                    prompts[prompt_name] = prompts[prompt_name][:, None]

            outputs = self.model(
                bmode, rf, prostate_mask, needle_mask, output_mode="all", **prompts
            )
            cancer_logits = outputs["mask_logits"]
            image_level_classification_outputs = outputs["cls_outputs"]
            data["image_level_classification_outputs"] = (
                image_level_classification_outputs
            )
        else:
            model_outputs = self.model(bmode)
            if isinstance(model_outputs, dict):
                cancer_logits = model_outputs[self.mask_output_key]
            else:
                cancer_logits = self.model(bmode)

        cancer_logits = (
            cancer_logits / self.temperature[None, None, None, :]
            + self.bias[None, None, None, :]
        )
        data["cancer_logits"] = cancer_logits

        # compute predictions
        masks = (prostate_mask > 0.5) & (needle_mask > 0.5)
        predictions, batch_idx = MaskedPredictionModule()(cancer_logits, masks)
        mean_predictions_in_needle = []
        for j in range(B):
            mean_predictions_in_needle.append(
                predictions[batch_idx == j].sigmoid().mean()
            )
        mean_predictions_in_needle = torch.stack(mean_predictions_in_needle)
        data["average_needle_heatmap_value"] = mean_predictions_in_needle

        prostate_masks = prostate_mask > 0.5
        predictions, batch_idx = MaskedPredictionModule()(cancer_logits, prostate_masks)
        mean_predictions_in_prostate = []
        for j in range(B):
            mean_predictions_in_prostate.append(
                predictions[batch_idx == j].sigmoid().mean()
            )
        mean_predictions_in_prostate = torch.stack(mean_predictions_in_prostate)
        data["average_prostate_heatmap_value"] = mean_predictions_in_prostate

        if include_postprocessed_heatmaps:
            cancer_logits = data["cancer_logits"]
            heatmap = cancer_logits[0, 0].detach().sigmoid().cpu().numpy()
            heatmap = (heatmap * 255).astype(np.uint8)
            # blur and upsample
            import cv2

            blurred = cv2.GaussianBlur(heatmap, (5, 5), sigmaX=1.5)
            upsampled = cv2.resize(blurred, (256, 256), interpolation=cv2.INTER_LINEAR)
            heatmap = upsampled
            data["cancer_probs"] = (torch.tensor(heatmap) / 255.0)[None, None, ...]

        return data

    def get_params_groups(self):
        if isinstance(self.model, SETR):
            encoder_parameters = []
            warmup_parameters = []
            cnn_parameters = []
            for name, param in self.model.named_parameters():
                if "head" in name:
                    warmup_parameters.append(param)
                else:
                    encoder_parameters.append(param)
            return encoder_parameters, warmup_parameters, cnn_parameters

        elif isinstance(self.model, ProstNFound):
            return self.model.get_params_groups()

        elif hasattr(self.model, "image_encoder"):
            encoder_parameters = []
            warmup_parameters = []
            cnn_parameters = []
            for name, param in self.model.named_parameters():
                if "image_encoder" in name:
                    encoder_parameters.append(param)
                else:
                    warmup_parameters.append(param)
            return encoder_parameters, warmup_parameters, cnn_parameters

        elif hasattr(self.model, "get_params_groups"):
            return self.model.get_params_groups()

        else:
            from itertools import chain

            encoder_parameters = []
            warmup_parameters = self.model.parameters()
            cnn_parameters = []

            return encoder_parameters, warmup_parameters, cnn_parameters


def load_config(config_path, options):
    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(options))

    # deal with issues from old schemas
    if cfg.tracked_metric == "val/core_auc_high_involvement":
        warn("`val/core_auc_high_involvement` is deprecated - use `val/auc` instead")
        cfg.tracked_metric = "val/auc"

    return cfg


if __name__ == "__main__":
    # main()
    p = ArgumentParser(description="Train ProstNFound model")
    p.add_argument(
        "--config", "-c", help="Path to config file (located in cfg/train/...)"
    )
    p.add_argument("options", nargs=argparse.REMAINDER)
    args = p.parse_args()
    cfg = load_config(args.config, args.options)

    main(cfg)