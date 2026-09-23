from argparse import ArgumentParser
import json
import sys
import os
import wandb
import logging
import torch

from medAI.factories.prostnfound.dataloaders_v1_simple import (
    get_dataloaders_optimum_kfold,
    get_dataloaders_nct_kfold,
)

sys.path.append(os.getcwd())

from medAI.datasets.nct2013.bmode_dataset import BModeDatasetV1
from medAI.engine.prostnfound import trainer as pnf
from medAI.modeling.prostnfound import (
    prostnfound_adapter_medsam_legacy,
    prostnfound_plus_final,
)
from medAI.factories.prostnfound.dataloaders_v0 import get_dataloaders_from_args
from medAI.datasets.nct2013.cohort_selection import select_cohort
from medAI.losses.prostnfound import build_loss, LossArgs
from medAI.utils.reproducibility import set_global_seed
from medAI.modeling.dinov3 import dinov3_backbone_from_checkpoint, dinov3_backbone
from medAI.engine.prostnfound.prostnfound_model_wrapper import (
    ProstNFoundModelInterface,
    ProstNFoundWrapperForHeatmapModel,
)
from medAI.factories.prostnfound.dataloaders_v1_simple import (
    get_dataloaders_optimum_kfold,
    get_nct2013_dataset,
)
import medAI.engine.prostnfound.trainer as pnf
from medAI.modeling.unetr import UNETR


logging.basicConfig(level=logging.INFO)


def train(
    name,
    seed=42,
    splits_seed=42,
    fold=0,
    n_folds=5,
    epochs=35,
    output_dir=None,
    model="dinov3_unetr_custom",
    image_size=224,
    save_checkpoints=False,
):
    wandb.init(
        project="prostnfound_Nov2025",
        group=name,
        name=name + f"_fold{fold}",
    )

    def log_metrics(data: dict):
        wandb.log(data)

        if output_dir is not None:
            with open(os.path.join(output_dir, "metrics.jsonl"), "a") as f:
                f.write(json.dumps(data) + "\n")

    set_global_seed(seed)

    cfg = pnf.ProstNFoundTrainingArgs(
        epochs=epochs, run_test=False, checkpoint_dir=output_dir if save_checkpoints else None
    )
    os.makedirs(output_dir, exist_ok=True)

    train_loader, val_loader = get_dataloaders_nct_kfold(
        n_folds=n_folds,
        fold=fold,
        splits_seed=splits_seed,
        image_size=image_size,
        mask_size=image_size,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if model == "dinov3_unetr_custom":
        teacher = dinov3_backbone_from_checkpoint(
            "/home/pwilson/projects/aip-medilab/pwilson/medAI/projects/dinov3/dinov3/configs/train/vitl_im1k_lin834_microUS.yaml",
            "/home/pwilson/projects/aip-medilab/pwilson/medAI/projects/dinov3/local_dino/eval/training_49999/teacher_checkpoint.pth",
            wrapper="feature_maps_list",
        )
    elif model.startswith("pretrained_dinov3"):
        teacher = dinov3_backbone(
            wrapper="feature_maps_list",
        )
    elif model == "dinov3_segmentor": 
        from dinov3.hub.segmentors import _make_dinov3_m2f_segmentor
        teacher = _make_dinov3_m2f_segmentor(
            backbone_name="dinov3_vitl16",
            pretrained=True,
            backbone_weights="lin834m_microUS",
            segmentor_weights="ade20k",
            autocast_dtype=torch.float16,
        )
    model = UNETR(
        image_encoder=teacher,
        embedding_size=1024,
        backbone_out_format="bchw",
        input_size=image_size,
        output_size=image_size,
    )

    wrapped_model = ProstNFoundWrapperForHeatmapModel(model).cuda()

    optimizer, lr_scheduler = pnf.setup_optimizers_and_schedulers(
        epochs,
        wrapped_model,
        train_loader,
        lr=1e-5,
        encoder_lr=1e-05,
        warmup_lr=0.0001,
        cnn_lr=1e-05,
    )
    scaler = torch.GradScaler()
    crit = build_loss(LossArgs(add_image_clf=False))

    pnf.run_training(
        wrapped_model,
        optimizer,
        lr_scheduler,
        train_loader,
        val_loader,
        test_loader=None,
        scaler=scaler,
        criterion=crit,
        cfg=cfg,
        log_fn=log_metrics,
    )


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fold", type=int, default=0, help="Which fold to use for training")
    p.add_argument(
        "--splits-seed", type=int, default=42, help="Random seed for splits generation"
    )
    p.add_argument(
        "--n-folds", type=int, default=5, help="Number of folds for cross-validation"
    )
    p.add_argument("--name", type=str, default="pnf_nct2013")
    p.add_argument("--output-dir")
    p.add_argument("--model", type=str, default="dinov3_unetr_custom")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--save-checkpoints", action="store_true")
    args = p.parse_args()

    train(
        name=args.name,
        seed=args.seed,
        splits_seed=args.splits_seed,
        fold=args.fold,
        n_folds=args.n_folds,
        epochs=args.epochs,
        output_dir=args.output_dir,
        model=args.model,
        image_size=args.image_size,
        save_checkpoints=args.save_checkpoints,
    )
