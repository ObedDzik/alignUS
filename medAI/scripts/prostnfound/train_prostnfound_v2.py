import argparse
from collections import defaultdict
import json
import logging
import os
from tempfile import mkdtemp
import typing as tp
import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import wandb
from einops import rearrange, repeat
from matplotlib import pyplot as plt
from medAI.modeling.prostnfound import ProstNFound
from medAI.modeling.setr import SETR
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
from medAI.utils.accumulators import DataFrameCollector
from medAI.losses.prostnfound import (
    LossArgs,
    build_loss,
)
from medAI.layers.masked_prediction_module import (
    MaskedPredictionModule,
)
from medAI.factories.prostnfound.dataloaders_v0 import get_dataloaders_from_args
from medAI.modeling.registry import create_model
from medAI.engine.prostnfound import trainer as pnf


def main(cfg):
    # setup
    logging.basicConfig(
        level=logging.INFO if not cfg.debug else logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
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

    logging.info("Setting up model")

    model = create_model(**cfg.model)
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
    criterion = build_loss(LossArgs(**cfg.loss))

    loaders = get_dataloaders_from_args(cfg.data)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    train_cfg = pnf.ProstNFoundTrainingArgs(**cfg.train)

    optimizer, lr_scheduler = pnf.setup_optimizers_and_schedulers(
        cfg.epochs, model, train_loader, **cfg.optim
    )
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

    pnf.run_training(
        model,
        optimizer,
        lr_scheduler,
        train_loader,
        val_loader,
        scaler=scaler,
        criterion=criterion,
        cfg=train_cfg,
        start_epoch=epoch,
        best_score=best_score,
        log_fn=wandb.log,
    )
