from collections import defaultdict
from dataclasses import asdict, dataclass
from doctest import debug
import json
import logging
import os
from tempfile import mkdtemp

import PIL
import hydra
import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
import torch
import wandb
from einops import rearrange, repeat
from matplotlib import pyplot as plt
from torch.nn import functional as F
from tqdm import tqdm
from torchvision.transforms import v2 as T

from medAI.modeling.registry import create_model, list_models, register_model
from medAI.modeling import *
from medAI.utils.reproducibility import (
    get_all_rng_states,
    set_all_rng_states,
    set_global_seed,
)
from medAI.losses.prostnfound import (
    build_loss,
)
from medAI.factories.prostnfound.dataloaders_v0 import get_dataloaders_from_args
from medAI.factories.prostnfound.models import get_model
from medAI.engine.prostnfound.evaluator import (
    ProstNFoundEvaluator as Evaluator,
)
from medAI.engine.prostnfound.prostnfound_model_wrapper import ProstNFoundModelInterface
from PIL import Image


@dataclass
class ProstNFoundTrainingArgs:
    log_images: bool = False
    use_amp: bool = True
    debug: bool = False
    accumulate_grad_steps: int = 1
    checkpoint_dir: str = None
    save_checkpoint_wandb: bool = False
    save_best_weights: bool = True
    run_val: bool = True
    run_test: bool = False
    test_every_epoch: bool = False
    tracked_metric: str = "val/auc"
    epochs: int = 100
    cutoff_epoch: int = None  # if set, stops training after this epoch
    wandb: dict = None
    seed: int = 42
    wandb_url: str = None  # filled in during setup


def basic_setup(cfg: ProstNFoundTrainingArgs, extra_cfg_to_log={}):
    # setup
    logging.basicConfig(
        level=logging.INFO if not cfg.debug else logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.info("Setting up experiment")

    cfg_to_log = asdict(cfg)
    cfg_to_log.update(extra_cfg_to_log)
    wandb.init(config=cfg_to_log, **cfg.wandb or {})
    _tmpdir = mkdtemp()
    # OmegaConf.save(cfg, os.path.join(_tmpdir, "train_config.yaml"), resolve=True)
    wandb.save(
        os.path.join(_tmpdir, "train_config.yaml"), base_path=_tmpdir, policy="now"
    )
    cfg.wandb_url = wandb.run.url if wandb.run else None

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

    return state


def run_training(
    wrapped_model: ProstNFoundModelInterface,
    optimizer,
    lr_scheduler,
    train_loader,
    val_loader,
    scaler,
    criterion,
    cfg: ProstNFoundTrainingArgs,
    test_loader=None,
    start_epoch=0,
    best_score=float("-inf"),
    log_fn=lambda x: None,
):

    def get_state():
        return {
            "model": wrapped_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "gradient_scaler": scaler.state_dict(),
            "rng": get_all_rng_states(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "args": asdict(cfg),
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

    for epoch in range(start_epoch, cfg.epochs):
        if cfg.cutoff_epoch is not None and epoch > cfg.cutoff_epoch:
            break
        logging.info(f"Epoch {epoch}")

        save_checkpoint("experiment_state.pth")

        run_train_epoch(
            cfg,
            wrapped_model,
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
                cfg, wrapped_model, val_loader, epoch, desc="val", log_fn=log_fn
            )
            if results_table is not None and cfg.checkpoint_dir is not None:
                results_table.to_csv(
                    os.path.join(cfg.checkpoint_dir, f"val_results_epoch_{epoch}.csv")
                )
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
                    cfg, wrapped_model, test_loader, epoch, desc="test"
                )

            if new_record and cfg.save_best_weights:
                save_checkpoint("best.pth")

    logging.info("Finished training")


def run_train_epoch(
    args: ProstNFoundTrainingArgs,
    wrapped_model: ProstNFoundModelInterface,
    loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    epoch,
    desc="Train",
    log_fn=lambda x: None,
):
    # setup epoch
    wrapped_model.train()
    evaluator = Evaluator(log_images=args.log_images)

    for train_iter, data in enumerate(tqdm(loader, desc=desc)):

        if args.debug and train_iter > 10:
            break

        # run the model
        with torch.cuda.amp.autocast(enabled=args.use_amp):

            data = wrapped_model(data)  # forward pass, populates data dict

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

        log_fn(step_metrics)

    # compute and log metrics
    metrics = evaluator.aggregate_metrics()
    desc = "train"
    metrics = {f"{desc}/{k}": v for k, v in metrics.items()}
    metrics["epoch"] = epoch
    log_fn(metrics)


@torch.no_grad()
def run_eval_epoch(
    args: ProstNFoundTrainingArgs,
    wrapped_model: ProstNFoundModelInterface,
    loader,
    epoch,
    desc="eval",
    log_fn=lambda x: None,
):
    wrapped_model.eval()

    evaluator = Evaluator(log_images=args.log_images)

    for train_iter, data in enumerate(tqdm(loader, desc=desc)):

        with torch.cuda.amp.autocast(enabled=args.use_amp):
            data = wrapped_model(data)

        # accumulate outputs
        step_metrics = {f"{desc}/{k}": v for k, v in evaluator(data).items()}
        log_fn(step_metrics)

    metrics = evaluator.aggregate_metrics()
    metrics = {f"{desc}/{k}": v for k, v in metrics.items()}
    metrics["epoch"] = epoch
    log_fn(metrics)

    return metrics, evaluator.get_full_results_table()


def setup_optimizers_and_schedulers(
    epochs,
    wrapped_model: ProstNFoundModelInterface,
    train_loader,
    warmup_epochs=0,
    warmup_lr=1e-4,
    lr=1e-4,
    cnn_lr=1e-6,
    encoder_lr=1e-5,
    wd=0.0,
    schedule="cosine",
):
    from torch.optim import AdamW

    (
        encoder_parameters,
        warmup_parameters,
        cnn_parameters,
    ) = wrapped_model.get_params_groups()

    total_epochs = epochs
    encoder_frozen_epochs = warmup_epochs
    warmup_epochs = 5
    niter_per_ep = len(train_loader)
    warmup_lr_factor = warmup_lr / lr
    params = [
        {"params": encoder_parameters, "lr": encoder_lr},
        {"params": warmup_parameters, "lr": lr},
        {"params": cnn_parameters, "lr": cnn_lr},
    ]

    def compute_lr_multiplier(iter, is_encoder_or_cnn=True):
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

    optimizer = AdamW(params, lr=lr, weight_decay=wd)
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
