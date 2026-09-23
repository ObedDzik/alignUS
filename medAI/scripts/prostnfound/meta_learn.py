
import argparse
import json
import typing as tp
from warnings import warn
from medAI.modeling.prostnfound import ProstNFound
from medAI.modeling.setr import SETR
from torchvision.transforms import v2 as T
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tempfile import mkdtemp
import copy
import os
import wandb
from typing import Dict, List, Tuple
import numpy as np
from omegaconf import OmegaConf
from argparse import ArgumentParser
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from torch.utils.data import ConcatDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from medAI.losses.prostnfound import (
    build_loss,
)
from .mae_encode import MaskedAutoEncode
import logging
from medAI.utils.distributed import is_main_process
from medAI.modeling import *
from medAI.utils.reproducibility import (
    get_all_rng_states,
    set_all_rng_states,
    set_global_seed,
)
from .center_dataloader import get_dataloaders_from_args
from .model_pnf import load_model

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class MetaLearningTTT:
    """
    Meta-Learning framework for Test-Time Training on multi-center prostate US data.
    Learns model initialization that adapts quickly to new centers via TTT.
    """
    
    def __init__(self, model, mae_module, cfg, criterion, log_fn=None):
        """
        Args:
            model: Your prostate cancer detection model
            mae_module: MaskedAutoEncode instance for TTT
            cfg: Configuration object with hyperparameters
            criterion: Loss function for task evaluation
            log_fn: Optional logging function for metrics
        """
        self.model = model
        self.mae_module = mae_module
        self.cfg = cfg
        self.device = 'cuda'
        self.log_fn = log_fn
        self.criterion = criterion
        self.scaler = GradScaler()
        self.ema_model_state = None
        
        # Meta-learning hyperparameters
        self.meta_lr = cfg.get('meta_lr', 1e-3)
        self.inner_lr = cfg.get('inner_lr', 1e-4)
        self.inner_steps = cfg.get('inner_steps', 5)
        self.mask_ratio = cfg.get('mask_ratio', 0.6)
        
        # Which encoder layers to update during TTT
        self.update_layers = cfg.get('update_layers', 'last_n')
        self.n_blocks = cfg.get('n_blocks', 4)
        
        # Meta-optimizer (updates original model)
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.meta_lr,
            weight_decay=1e-4
        )

        self.scheduler = CosineAnnealingLR(
            self.meta_optimizer, 
            T_max=cfg.epochs,
            eta_min=5e-4
        )
        
        # Model checkpointing
        self.best_score = float('-inf')  # For metrics like AUC (higher is better)
        self.best_model_state = None
    
    def get_trainable_params(self, model):
        """Get parameters to update during inner loop TTT."""
        if self.update_layers == 'all':
            return model.model.image_encoder.parameters()
        elif self.update_layers == 'last_n':
            backbone = model.model.image_encoder.backbone
            if hasattr(backbone, 'blocks'):
                return backbone.blocks[-self.n_blocks:].parameters()
            else:
                return model.model.image_encoder.parameters()
        elif self.update_layers == 'last_only':
            backbone = model.model.image_encoder.backbone
            if hasattr(backbone, 'blocks'):
                return backbone.blocks[-1].parameters()
            else:
                return model.model.image_encoder.parameters()
        else:
            return model.model.image_encoder.parameters()
    
    def clone_model(self):
        """Create a deep copy of the model for inner loop adaptation."""
        cloned = copy.deepcopy(self.model)
        cloned.train()
        return cloned
    
    def inner_loop_ttt(self, adapted_model, support_batch):
        """
        Perform TTT adaptation on support set (inner loop).
        
        Args:
            adapted_model: Cloned model to adapt
            support_batch: Data batch from current center for adaptation
            
        Returns:
            adapted_model: Model after TTT adaptation
        """
        # Get parameters to update
        params = list(self.get_trainable_params(adapted_model))
        
        # Inner loop optimizer
        inner_optimizer = torch.optim.SGD(params, lr=self.inner_lr)
        
        # Perform TTT steps
        for step in range(self.inner_steps):
            inner_optimizer.zero_grad()
            
            # Compute reconstruction loss using MAE
            tokens = adapted_model.model.image_encoder.backbone.get_intermediate_layers(
                support_batch['bmode'].to(self.device), n=1, reshape=False
            )
            patch_features = tokens[0]
            B, N, D = patch_features.shape
            
            # Random masking
            num_masked = int(N * self.mask_ratio)
            noise = torch.rand(B, N, device=patch_features.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            
            mask = torch.zeros(B, N, device=patch_features.device)
            mask[:, :num_masked] = 1
            mask = torch.gather(mask, dim=1, index=ids_restore)
            
            # Get masked tokens and reconstruct
            mask_indices = mask.bool()
            masked_tokens = patch_features[mask_indices]
            
            pred_patches = self.mae_module.decoder(masked_tokens)
            
            # Ground truth patches
            gt_patches = self.mae_module.patchify(support_batch['bmode'].to(self.device))
            gt_masked = gt_patches[mask_indices]
            
            # Reconstruction loss
            recon_loss = F.mse_loss(pred_patches, gt_masked)
            
            # Backward and update
            recon_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            inner_optimizer.step()
        
        return adapted_model
    
    def compute_task_loss(self, model, query_batch):
        """
        Compute task loss (lesion detection) on query set.
        
        Args:
            model: Model to evaluate
            query_batch: Validation batch from current center
            
        Returns:
            loss: Detection loss
        """
        model.eval()
        with torch.enable_grad():  # Need gradients for meta-update
                data = model(query_batch)
                loss = self.criterion(data)
        model.train()
        return loss
            
    def meta_train_step(self, train_dataloaders, evaluator, num_tasks_per_step=2, iteration=0):
        available_centers = list(train_dataloaders.keys())
        sampled_centers = np.random.choice(
            available_centers,
            size=min(num_tasks_per_step, len(available_centers)),
            replace=False
        )
        self.meta_optimizer.zero_grad()
        meta_loss_accum = 0.0

        for center_id in sampled_centers:
            support_loader, query_loader = train_dataloaders[center_id]
            try:
                support_batch = next(iter(support_loader))
                query_batch = next(iter(query_loader))
            except StopIteration:
                print("!!!!!!!!!Iteration Stopped here")
                continue

            adapted_model = self.clone_model()
            # Inner-loop adaptation
            adapted_model = self.inner_loop_ttt(adapted_model, support_batch)
            # Compute task loss for logging only
            with torch.no_grad():
                task_loss = self.compute_task_loss(adapted_model, query_batch)
                meta_loss_accum += task_loss.item()
                # Evaluate for metrics
                data = adapted_model(query_batch)
                evaluator(data)
            # Reptile meta-gradient: move toward adapted params
            for p_meta, p_adapted in zip(
                self.model.parameters(),
                adapted_model.parameters()
            ):
                if p_meta.grad is None:
                    p_meta.grad = torch.zeros_like(p_meta)
                p_meta.grad += (p_meta - p_adapted).detach()
            del adapted_model
            torch.cuda.empty_cache()

        if iteration%5==0:
            total_grad_norm = sum(p.grad.norm().item() ** 2 for p in self.model.parameters() if p.grad is not None) ** 0.5
            print(f"Reptile grad norm (before clip): {total_grad_norm:.6f}")

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.meta_optimizer.step()

        if self.ema_model_state is None:
            self.ema_model_state = copy.deepcopy(self.model.state_dict())
        else:
            # EMA update
            for key in self.ema_model_state:
                self.ema_model_state[key] = (
                    0.9 * self.ema_model_state[key] + 
                    0.1 * self.model.state_dict()[key]
                )

        return meta_loss_accum / len(sampled_centers)

    
    def meta_train(self, train_dataloaders, num_iterations=1000, epoch=0, log_fn=None):
        """
        Full meta-training loop with batch-accumulated metrics.
        Metrics are computed only on the aggregated results to avoid sklearn warnings.
        """
        from medAI.engine.prostnfound.evaluator import ProstNFoundEvaluator as Evaluator

        self.model.train()
        training_losses = []
        print(f"Meta-training on centers: {list(train_dataloaders.keys())}")
        train_evaluator = Evaluator(**self.cfg.evaluator)

        for iteration in range(num_iterations):
            meta_loss = self.meta_train_step(train_dataloaders, train_evaluator, num_tasks_per_step=2, iteration=iteration)
            training_losses.append(meta_loss)

            step_metrics = {
                "meta_train/loss": meta_loss,
                "meta_train/iteration": iteration,
                "meta_train/epoch": epoch
            }
            if wandb.run:
                wandb.log(step_metrics)
            if log_fn is not None:
                log_fn(step_metrics)

        # Aggregate metrics ONCE at end of all iterations (per epoch)
        train_metrics = train_evaluator.aggregate_metrics()
        train_log_metrics = {f"meta_train/{k}": v for k, v in train_metrics.items()}
        train_log_metrics["epoch"] = epoch

        if wandb.run:
            wandb.log(train_log_metrics)

        print(f"Training metrics (Epoch {epoch}):")
        for k, v in train_metrics.items():
            print(f"  {k}: {v:.4f}")

        print(f"Training loss: {sum(training_losses) / len(training_losses)}")

        logging.info("Finished meta-training")
        return training_losses

        
    def meta_validate(self, val_loader, reset_after_each=True, epoch=0):
        from medAI.engine.prostnfound.evaluator import ProstNFoundEvaluator as Evaluator
        
        self.model.eval()
        evaluator = Evaluator(**self.cfg.evaluator)
        
        # Save meta-learned weights
        meta_learned_state = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.ema_model_state)

        for val_iter, batch in enumerate(tqdm(val_loader, desc='val')):
            # Clone + adapt
            adapted_model = self.clone_model()
            adapted_model = self.inner_loop_ttt(adapted_model, batch)

            adapted_model.eval()
            with torch.no_grad():
                data = adapted_model(batch)
            evaluator(data)

            del adapted_model
            torch.cuda.empty_cache()
            
        if reset_after_each:
            self.model.load_state_dict(meta_learned_state)

        metrics = evaluator.aggregate_metrics()
        results_table = evaluator.results_table
        metrics["epoch"] = epoch

        self.model.train()
        return metrics, results_table


def main(cfg_path):
    """
    Main execution function for meta-learning TTT.
    
    Args:
        cfg: Configuration object
        model: Prostate cancer detection model
        mae_module: MaskedAutoEncode instance
        center_dataloaders: Dict of {center_id: (support_loader, query_loader)}
        criterion: Loss function
        log_fn: Logging function compatible with your metrics system
        
    Returns:
        meta_ttt: Trained MetaLearningTTT instance
    """
    cfg = OmegaConf.load(cfg_path)
    best_score = float('-inf')
    best_model_state = None

    if cfg.get("output_dir") is not None:
        os.makedirs(cfg.output_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(cfg.output_dir, "train_config.yaml"), resolve=True)
        
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
    wandb.init(config=OmegaConf.to_object(cfg),**cfg.get("wandb", {}),)
    _tmpdir = mkdtemp()
    OmegaConf.save(cfg, os.path.join(_tmpdir, "train_config.yaml"), resolve=True)
    wandb.save(
        os.path.join(_tmpdir, "train_config.yaml"), base_path=_tmpdir, policy="now"
    )
    set_global_seed(cfg.seed)
    logging.info("Setting up model")

    model = load_model(cfg)
    mae_module = MaskedAutoEncode(model, cfg, embed_dim=1024, patch_size=16)

    center_dataloaders = get_dataloaders_from_args(cfg.data)
    
    if "pos_weight" not in cfg:
        cfg.pos_weight = 1.0
    criterion = build_loss(cfg)

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

    meta_ttt = MetaLearningTTT(model, mae_module, cfg, criterion, log_fn=log_fn)
    
    val_center = cfg.get('val_center', None)
    if val_center is None:
        logging.warning("No validation center specified in cfg.val_center")
        val_center = list(center_dataloaders.keys())[0]
        logging.info(f"Using {val_center} as validation center")
    
    print(f"\n{'='*60}")
    print(f"Meta-Learning TTT - Validation Center: {val_center}")
    print(f"{'='*60}\n")

    val_loader=center_dataloaders[val_center]
    support_loader, query_loader = val_loader
    combined_dataset = ConcatDataset([support_loader.dataset, query_loader.dataset])
    val_loader_full = DataLoader(
        combined_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers
    )
    
    # Meta-train on other centers, validate on val_center
    train_dataloaders = {k: v for k, v in center_dataloaders.items() if k != val_center}
    for epoch in range(cfg.epochs):
        losses = meta_ttt.meta_train(
            train_dataloaders,
            num_iterations=cfg.get('meta_iterations', 1000),
            epoch=epoch,
            log_fn=log_fn
        )
        epoch_loss = sum(losses) / len(losses)
        train_epoch_metric = {
            "meta_train/epoch": epoch, 
            "meta_train/epoch_loss": epoch_loss
        }
        wandb.log(train_epoch_metric)

        metrics, results_table = meta_ttt.meta_validate(
            val_loader = val_loader_full, 
            reset_after_each=True,
            epoch=epoch, 
            )
        wandb.log({f"meta_val/{k}": v for k, v in metrics.items()})
        log_fn(metrics)

        if cfg.output_dir:
            table_path = os.path.join(
                cfg.output_dir, f"val_results_iter_{epoch:05d}.csv"
            )
            results_table.to_csv(table_path)

        tracked_metric = metrics.get(cfg.tracked_metric)
        if tracked_metric is not None and tracked_metric > best_score:
            best_score = tracked_metric
            logging.info(f"New best score: {best_score:.4f}")
            print(f"  *** New best model! (tracked_metric: {tracked_metric:.4f}) ***")

            if wandb.run:
                wandb.log({
                    "meta_val/best_score": best_score,
                    "meta_val/best_epoch": epoch
                })

            if cfg.get('save_best_weights', True) and cfg.chkpt_dir:
                checkpoint_path = os.path.join(cfg.chkpt_dir, "best_meta.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': meta_ttt.model.state_dict(),
                    'optimizer_state_dict': meta_ttt.meta_optimizer.state_dict(),
                    'best_score': best_score,
                    'val_metrics': metrics
                }, checkpoint_path)

        meta_ttt.scheduler.step()

    print(f"\n{'='*60}")
    print("Meta-Learning TTT Complete")
    print(f"Best validation score: {best_score:.4f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    p = ArgumentParser(description="Train MetaLearn")
    p.add_argument(
        "--config", "-c", help="Path to config file (located in cfg/train/...)"
    )
    args = p.parse_args()
    main(args.config)