from argparse import ArgumentParser
import sys
import os
import wandb
import logging
import torch

import medAI.engine.prostnfound.prostnfound_model_wrapper
from medAI.factories.prostnfound.dataloaders_v1_simple import get_dataloaders_optimum_kfold

sys.path.append(os.getcwd())

from medAI.datasets.nct2013.bmode_dataset import BModeDatasetV1
from medAI.engine.prostnfound import trainer as pnf
from medAI.modeling.prostnfound import prostnfound_adapter_medsam_legacy, prostnfound_plus_final
from medAI.factories.prostnfound.dataloaders_v0 import get_dataloaders_from_args
from medAI.datasets.nct2013.cohort_selection import select_cohort
from medAI.losses.prostnfound import build_loss, LossArgs
from medAI.utils.reproducibility import set_global_seed


logging.basicConfig(level=logging.INFO)


def main():
    p = ArgumentParser()
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--prompts", nargs="+", default=[], help="List of prompts to use"
    )
    p.add_argument("--use-class-decoder", action="store_true", help="If set, use class decoder")
    p.add_argument(
        "--fold", type=int, default=0, help="Which fold to use for training"
    )
    p.add_argument("--splits-seed", type=int, default=42, help="Random seed for splits generation")
    p.add_argument("--n-folds", type=int, default=5, help="Number of folds for cross-validation")
    p.add_argument("--name", type=str, default="pnf_nct2013_experiment")
    p.add_argument("--use-pretrained-pnf", action="store_true", help="If set, use pretrained ProstNFound weights")
    p.add_argument("--output-dir")
    args = p.parse_args()


    wandb.init(
        project="prostnfound_Nov2025",
        group=args.name,
        name=args.name + f"_fold{args.fold}",
    )

    set_global_seed(args.seed)

    cfg = pnf.ProstNFoundTrainingArgs(epochs=args.epochs, run_test=False, checkpoint_dir=args.output_dir)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    train_loader, val_loader = get_dataloaders_optimum_kfold(
        n_folds=args.n_folds,
        fold=args.fold,
        splits_seed=args.splits_seed,
    )

    # MODEL, OPTIMIZER, SCHEDULER, LOSS
    kw = dict(prompts=args.prompts, use_class_decoder=args.use_class_decoder)
    if args.use_pretrained_pnf:
        model = prostnfound_plus_final(**kw, strict=False)
    else:
        model = prostnfound_adapter_medsam_legacy(**kw)
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
    crit = build_loss(LossArgs(add_image_clf=args.use_class_decoder))

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
        log_fn=wandb.log,
    )


if __name__ == "__main__":
    main()