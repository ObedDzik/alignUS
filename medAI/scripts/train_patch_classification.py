from argparse import ArgumentParser
from functools import partial
import os
from medAI.datasets.optimum.cohort_selection import OptimumSplitsGenerator
from medAI.engine.patch_classification_trainer import PatchClassificationTrainer
from timm import create_model
from torchvision.transforms import v2 as T
import torch
from medAI.datasets.optimum import NeedleTraceImageFramesDataset
from medAI.datasets.patches_dataset_wrapper import PatchesDatasetWrapper
from medAI.transforms.normalization import InstanceNormalizeImage
import wandb


def main():
    p = ArgumentParser()
    p.add_argument(
        "--dataset", type=str, choices=["optimum", "nct2013"], default="optimum"
    )
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--log-wandb", action="store_true")

    args = p.parse_args()

    if args.log_wandb:
        wandb.init(project="medAI_patch-classification")
        log_fn = wandb.log
    else:
        log_fn = None

    if args.dataset == "optimum":
        datasets = get_optimum_datasets()
    else:
        datasets = get_nct2013_datasets(fold=args.fold)

    model = create_model(
        "resnet18",
        pretrained=True,
        num_classes=2,
        in_chans=1 if args.dataset == "nct2013" else 3,
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    trainer = PatchClassificationTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=torch.utils.data.DataLoader(
            datasets["train"],
            batch_size=64,
            shuffle=True,
            num_workers=4,
        ),
        epochs=100,
        val_loader=torch.utils.data.DataLoader(
            datasets["val"],
            batch_size=1,
            shuffle=False,
            num_workers=4,
        ),
        logger=log_fn,
    )
    trainer.train()


def get_patch_transform(augmentations=False, instance_norm=False):

    if augmentations:
        augs = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomAffine(degrees=0, translate=(0.2, 0.2)),
            ]
        )
    else:
        augs = T.Identity()

    patch_t = T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Resize((224, 224)),
            augs,
            InstanceNormalizeImage() if instance_norm else T.Identity(),
            lambda p: torch.stack(p),
        ]
    )

    return patch_t


def get_optimum_datasets(fold=0, n_folds=5):

    g = OptimumSplitsGenerator(
        "/home/pwilson/projects/aip-medilab/pwilson/medAI/data/OPTIMUM/processed/UA_annotated_needles/mined_path_reports.csv"
    )
    splits = g.split_patients_kfold(k=n_folds, fold=fold).get_cine_id_splits()

    def transform_item(item, patch_t):
        output = {}
        output["patches"] = patch_t(item["patches"])
        output["label"] = int(item["info"]["Diagnosis"] != "Benign")
        return output

    train_ds = NeedleTraceImageFramesDataset(
        "data/OPTIMUM/processed/UA_annotated_needles", out_fmt="np", cine_ids=splits["train"]
    )
    val_ds = NeedleTraceImageFramesDataset(
        "data/OPTIMUM/processed/UA_annotated_needles", out_fmt="np", cine_ids=splits["val"]
    )
    train_ds = PatchesDatasetWrapper(
        train_ds,
        patch_size_mm=(5, 5),
        patch_stride_mm=(1, 1),
        image_key="image",
        mask_keys=["needle_mask"],
        physical_height_getter=lambda item: item["info"]["heightMm"],
        physical_width_getter=lambda item: item["info"]["widthMm"],
        mask_thresholds=[0.25],
        yield_one_patch_per_item=True,
        include_images=False,
        transform=partial(
            transform_item,
            patch_t=get_patch_transform(augmentations=True, instance_norm=True),
        ),
    )
    val_ds = PatchesDatasetWrapper(
        val_ds,
        patch_size_mm=(5, 5),
        patch_stride_mm=(1, 1),
        image_key="image",
        mask_keys=["needle_mask"],
        physical_height_getter=lambda item: item["info"]["heightMm"],
        physical_width_getter=lambda item: item["info"]["widthMm"],
        mask_thresholds=[0.25],
        yield_one_patch_per_item=False,
        include_images=False,
        transform=partial(
            transform_item,
            patch_t=get_patch_transform(augmentations=False, instance_norm=True),
        ),
    )
    return dict(
        train=train_ds,
        val=val_ds,
    )


def get_nct2013_datasets(fold=0, n_folds=5):
    from medAI.datasets.nct2013.bmode_dataset import BModeDatasetV1
    from medAI.datasets.nct2013.cohort_selection import select_cohort

    def transform_item(item, patch_t):
        output = {}
        output["patches"] = patch_t(item["patches"])
        output["label"] = int(item["grade"] != "Benign")
        return output

    train, val, test = select_cohort(
        fold=fold,
        n_folds=n_folds,
        exclude_benign_cores_from_positive_patients=True,
        involvement_threshold_pct=40,
    )

    train_ds = BModeDatasetV1(train)
    val_ds = BModeDatasetV1(val)

    if os.getenv("DEBUG", None):
        train_ds = torch.utils.data.Subset(train_ds, list(range(20)))
        val_ds = torch.utils.data.Subset(val_ds, list(range(10)))

    train_ds = PatchesDatasetWrapper(
        train_ds,
        image_key="bmode",
        mask_keys=["needle_mask", "prostate_mask"],
        mask_thresholds=[0.6, 0.9],
        patch_size_mm=(5, 5),
        patch_stride_mm=(1, 1),
        yield_one_patch_per_item=True,
        include_images=False,
        transform=partial(
            transform_item,
            patch_t=get_patch_transform(augmentations=True, instance_norm=True),
        ),
    )
    val_ds = PatchesDatasetWrapper(
        val_ds,
        image_key="bmode",
        mask_keys=["needle_mask", "prostate_mask"],
        mask_thresholds=[0.6, 0.9],
        patch_size_mm=(5, 5),
        patch_stride_mm=(1, 1),
        yield_one_patch_per_item=False,
        include_images=False,
        transform=partial(
            transform_item,
            patch_t=get_patch_transform(augmentations=False, instance_norm=True),
        ),
    )
    return dict(
        train=train_ds,
        val=val_ds,
    )


if __name__ == "__main__":
    main()
