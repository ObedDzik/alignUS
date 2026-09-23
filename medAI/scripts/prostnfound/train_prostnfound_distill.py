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


@hydra.main(
    config_path="pkg://medAI/configs/prostnfound",
    config_name="default",
    version_base=None,
)
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

    #Use teacher
    if cfg.teacher.use_teacher:
        skip_keys=['dino_head','ibot_head']
        teacher_model=get_model(OmegaConf.to_object(cfg.teacher))
        # teacher_model = teacher_model.image_encoder
        teacher_model.to(cfg.device)
        teacher_model.eval()
            # if cfg.torch_compile:
            #     torch.compile(teacher_model)
            # state_dict = torch.load(cfg.teacher.chkpt, map_location='cpu')['teacher']
            # state_dict = {k:v 
            #                 for k,v in state_dict.items()
            #                 if not any(skip_key in k for skip_key in skip_keys)
            #             }
            # if 'model' in state_dict:
            #     msg = teacher_model.image_encoder.load_state_dict(state_dict['model'])
            # else:
            #     msg = teacher_model.load_state_dict(state_dict)
            # logging.info(f"loaded teacher model from {cfg.teacher.chkpt} with msg {msg}")

    # setup criterion
    if "pos_weight" not in cfg:
        cfg.pos_weight = 1.0
    criterion = build_loss(cfg)

    #distill_loss
    distill_module = None
    if cfg.teacher.use_teacher:
        distill_module = PatchLevelDistillation(
            teacher_model,
            model,
            cfg).to(cfg.device)
        assert distill_module is not None, "Distillation module failed!"
        print(f"Distillation enabled: loss_type={cfg.teacher.loss_type}, weight={cfg.teacher.distill_weight}")
    
    recon_module = None
    if cfg.aux_task.use_aux_task:
        recon_module = MaskedAutoEncode(model,cfg).to(cfg.device)
        assert recon_module is not None, "Reconstruction module failed!"
        print(f"Reconstruction enabled: loss_type={cfg.aux_task.loss_type}, weight={cfg.aux_task.aux_weight}")

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
            distill_module=distill_module,
            recon_module=recon_module
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
    cfg,
    model,
    loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    epoch,
    desc="Train",
    log_fn=None,
    distill_module=None,
    recon_module=None,
):
    # setup epoch
    model.train()
    evaluator = Evaluator(**cfg.evaluator)

    for train_iter, data in enumerate(tqdm(loader, desc=desc)):

        if cfg.debug and train_iter > 10:
            break

        # run the model
        with torch.cuda.amp.autocast(enabled=cfg.use_amp):

            data = model(data)  # heatmap

            if torch.any(torch.isnan(data["cancer_logits"])):
                logging.warning("NaNs in heatmap logits")

            # loss calculation
            supervised_loss = criterion(data)
            
            if cfg.teacher.use_teacher:
                distill_loss = distill_module(data)
                total_loss = (
                    supervised_loss/cfg.accumulate_grad_steps +
                    cfg.teacher.distill_weight * (distill_loss/cfg.accumulate_grad_steps)
                )
            elif cfg.aux_task.use_aux_task:
                aux_weight = float(cfg.aux_task.aux_weight)  # Ensure float
                recon_loss = recon_module(data)
                
                # Combine losses
                combined_loss = supervised_loss + aux_weight * recon_loss
                total_loss = combined_loss / cfg.accumulate_grad_steps

            else:
                total_loss = supervised_loss / cfg.accumulate_grad_steps
            
        # backward pass
        if cfg.use_amp:
            logging.debug("Backward pass")
            scaler.scale(total_loss).backward()
        else:
            logging.debug("Backward pass")
            total_loss.backward()

        # gradient accumulation and optimizer step
        if cfg.debug:
            for param in optimizer.param_groups[1]["params"]:
                break
            logging.debug(param.data.view(-1)[0])

        if (train_iter + 1) % cfg.accumulate_grad_steps == 0:
            logging.debug("Optimizer step")
            if cfg.use_amp:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()

            if cfg.debug:
                for param in optimizer.param_groups[1]["params"]:
                    break
                logging.debug(param.data.view(-1)[0])

        scheduler.step()

        # accumulate outputs
        step_metrics = {f"train/{k}": v for k, v in evaluator(data).items()}

        # log metrics
        step_metrics.update({"train_supervised_loss": supervised_loss.item()})
        step_metrics.update({"train_total_loss": total_loss.item()})
        step_metrics.update({"train_reccon_loss": recon_loss.item()})
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
        

class PatchLevelDistillation(nn.Module):
    """Compare all three loss types"""
    
    def __init__(self, teacher, student, cfg, loss_type='kl', temperature=4.0):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.device = cfg.device
        self.loss_type = loss_type  # 'kl', 'mse', 'cosine'
        self.temperature = temperature
        
        self.layer_weights = {4: 0.5, 11: 0.75, 17: 1.0, 23: 1.5}
        # self.layer_weights = {0: 0.5, 7: 1.0, 15: 0.5}
        
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def compute_loss(self, s_valid, t_valid):
        """Compute loss based on loss_type"""
        
        if self.loss_type == 'kl':
            # KL Divergence
            with torch.no_grad():
                t_dist = F.softmax(t_valid / self.temperature, dim=-1)
            s_log_dist = F.log_softmax(s_valid / self.temperature, dim=-1)
            loss = F.kl_div(s_log_dist, t_dist, reduction='batchmean')
            loss = loss * (self.temperature ** 2)
            return loss
        
        elif self.loss_type == 'mse':
            # MSE (with normalization)
            s_norm = F.normalize(s_valid, dim=-1)
            t_norm = F.normalize(t_valid, dim=-1)
            return F.mse_loss(s_norm, t_norm)
        
        elif self.loss_type == 'cosine':
            # Cosine similarity (converted to loss)
            s_norm = F.normalize(s_valid, dim=-1)
            t_norm = F.normalize(t_valid, dim=-1)
            similarity = F.cosine_similarity(s_norm, t_norm, dim=-1).mean()
            return 1 - similarity
        
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
    
    def forward(self, data):
        images = data['bmode'].to(self.device)
        prostate_mask = data['prostate_mask'].to(self.device)
        B = len(images)
        blocks_to_align = [4, 11, 17, 23]
        # blocks_to_align = [0, 7, 15]
        
        with torch.no_grad():
            teacher_out = self.teacher.backbone(images)
        student_out = self.student.model.backbone(images)
        
        total_loss = 0.0
        total_weight = 0.0
        
        for block in blocks_to_align:
            t_feat = teacher_out[block].flatten(2).transpose(1, 2)
            s_feat = student_out[block].flatten(2).transpose(1, 2)
            
            N_patches = t_feat.shape[1]
            H_patches = int(np.sqrt(N_patches))
            
            mask_resized = F.interpolate(
                prostate_mask.float(), 
                size=(H_patches, H_patches), 
                mode='nearest'
            )
            mask_flat = (mask_resized.flatten(2).transpose(1, 2) > 0.5).squeeze(-1)
            
            valid_mask = mask_flat.flatten()
            t_valid = t_feat.flatten(0, 1)[valid_mask]
            s_valid = s_feat.flatten(0, 1)[valid_mask]
            
            if t_valid.shape[0] < 2:
                continue
            
            # Compute loss with selected method
            block_loss = self.compute_loss(s_valid, t_valid)
            
            weight = self.layer_weights[block]
            total_loss += weight * block_loss
            total_weight += weight
        
        return total_loss / total_weight if total_weight > 0 else torch.tensor(0.0, device=images.device)

        
class MaskedAutoEncode(nn.Module):
    """Simple masked reconstruction as auxiliary task"""
    def __init__(self, model, cfg, embed_dim=1024, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.model = model
        self.device = cfg.device
        
        # Reconstruction head (predicts pixel values)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),  # Better than Dropout for this
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            
            nn.Linear(512, patch_size * patch_size * 3)
        ).to(self.device)
    
    def forward(self, data, mask_ratio=0.3):
        """
        Args:
            patch_features: [B, N_patches, embed_dim] from backbone
            images: [B, 3, H, W] original images
            mask_ratio: fraction of patches to mask
        """
        # print(data['bmode'].shape)
        images = data['bmode'].to(self.device)
        tokens = self.model.model.image_encoder.backbone.get_intermediate_layers(images, n=1, reshape=False)
        B, N, D = tokens[0].shape
        
        # Random masking
        num_masked = int(N * mask_ratio)
        noise = torch.rand(B, N, device=tokens[0].device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask = torch.zeros(B, N, device=tokens[0].device)
        mask[:, :num_masked] = 1
        mask = torch.gather(mask, dim=1, index=ids_shuffle)
        
        # Get masked patch features
        masked_features = tokens[0][mask.bool()]  # [B*num_masked, D]
        
        # Predict original patches
        reconstructed = self.decoder(masked_features)  # [B*num_masked, patch_size^2*3]
        reconstructed = reconstructed.reshape(-1, 3, self.patch_size, self.patch_size)
        
        # Extract ground truth patches
        patches = F.unfold(
            images,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )  # [B, 3*patch_size^2, N]
        patches = patches.transpose(1, 2)  # [B, N, 3*patch_size^2]
        gt_patches = patches[mask.bool()]  # [B*num_masked, 3*patch_size^2]
        gt_patches = gt_patches.reshape(-1, 3, self.patch_size, self.patch_size)
        
        # MSE loss
        loss = F.mse_loss(reconstructed, gt_patches)
        return loss




def load_config(config_path, options):
    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(options))

    # deal with issues from old schemas
    if cfg.tracked_metric == "val/core_auc_high_involvement":
        warn("`val/core_auc_high_involvement` is deprecated - use `val/auc` instead")
        cfg.tracked_metric = "val/auc"

    return cfg


if __name__ == "__main__":
    main()
    # p = ArgumentParser(description="Train ProstNFound model")
    # p.add_argument(
    #     "--config", "-c", help="Path to config file (located in cfg/train/...)"
    # )
    # p.add_argument("options", nargs=argparse.REMAINDER)
    # args = p.parse_args()
    # cfg = load_config(args.config, args.options)
#
# main(cfg)