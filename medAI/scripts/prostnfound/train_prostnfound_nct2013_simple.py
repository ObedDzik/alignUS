from argparse import ArgumentParser
import sys
import os
import wandb
import logging
import torch

import medAI.engine.prostnfound.prostnfound_model_wrapper

sys.path.append(os.getcwd())

from medAI.datasets.nct2013.bmode_dataset import BModeDatasetV1
from medAI.engine.prostnfound import trainer as pnf
from medAI.transforms.prostnfound_transform import ProstNFoundTransform
from medAI.modeling.prostnfound import prostnfound_adapter_medsam_legacy
from medAI.factories.prostnfound.dataloaders_v0 import get_dataloaders_from_args
from medAI.datasets.nct2013.cohort_selection import select_cohort
from medAI.factories.utils.build_dataloader import build_dataloader
from medAI.losses.prostnfound import build_loss, LossArgs
from medAI.utils.reproducibility import set_global_seed
from medAI.datasets.optimum.cohort_selection import OptimumSplitsGenerator
from medAI.datasets.optimum.needle_trace_dataset import NeedleTraceImageFramesDataset


logging.basicConfig(level=logging.INFO)


def main(): 
    p = ArgumentParser()
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--noprompt", action="store_true", help="If set, do not use tabular prompts"
    )
    p.add_argument(
        "--fold", type=int, default=0, help="Which fold to use for training (0-4)"
    )
    p.add_argument("--name", type=str, default="pnf_nct2013_experiment")

    args = p.parse_args()

    wandb.init(
        project="prostnfound_nct2013_training",
        group=args.name,
        name=args.name + f"_fold{args.fold}",
    )

    set_global_seed(args.seed)
    epochs = args.epochs
    batch_size = 8

    cfg = pnf.ProstNFoundTrainingArgs(epochs=args.epochs, run_test=False)

    train_transform = ProstNFoundTransform(
        augment="translate", image_size=256, mask_size=64
    )
    val_transform = ProstNFoundTransform(augment="none", image_size=256, mask_size=64)
    train_cores, val_cores, test_cores = select_cohort(
        args.fold, 5, exclude_benign_cores_from_positive_patients=True, mode="kfold"
    )
    train_dataset = BModeDatasetV1(
        train_cores,
        train_transform,
        include_rf=False,
    )
    val_dataset = BModeDatasetV1(
        val_cores,
        val_transform,
        include_rf=False,
    )
    train_loader = build_dataloader(
        train_dataset, batch_size=8, shuffle=True, num_workers=4
    )
    val_loader = build_dataloader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    # build model
    prompts = [] if args.noprompt else ["age", "psa"]
    model = prostnfound_adapter_medsam_legacy(
        prompts=prompts
    )

    wrapped_model = medAI.engine.prostnfound.prostnfound_model_wrapper.ProstNFoundModelWrapper(model).cuda()
    optimizer, lr_scheduler = pnf.setup_optimizers_and_schedulers(
        args.epochs,
        wrapped_model,
        train_loader,
        lr=1e-5,
        encoder_lr=1e-05,
        warmup_lr=0.0001,
        cnn_lr=1e-05,
    )
    scaler = torch.GradScaler()

    crit = build_loss(LossArgs())

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
    )


def get_optimum_dataset(fold=0, n_folds=5): 
    g = pnf.OptimumSplitsGenerator(
        "/home/pwilson/projects/aip-medilab/pwilson/medAI/data/OPTIMUM/processed/UA_annotated_needles/mined_path_reports.csv"
    )
    splits = g.split_patients_kfold(k=n_folds, fold=fold).get_cine_id_splits()

    train_transform = ProstNFoundTransform(
        augment="translate", image_size=256, mask_size=64
    )
    val_transform = ProstNFoundTransform(augment="none", image_size=256, mask_size=64)
    train_dataset = pnf.NeedleTraceImageFramesDataset(
        "data/OPTIMUM/processed/UA_annotated_needles",
        out_fmt="np",
        cine_ids=splits["train"],
        transform=train_transform,
    )
    val_dataset = pnf.NeedleTraceImageFramesDataset(
        "data/OPTIMUM/processed/UA_annotated_needles",
        out_fmt="np",
        cine_ids=splits["val"],
        transform=val_transform,
    )
    return train_dataset, val_dataset


if __name__ == "__main__":
    main()