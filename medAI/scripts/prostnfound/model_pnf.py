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

def load_model(cfg):
    
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
    return model

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