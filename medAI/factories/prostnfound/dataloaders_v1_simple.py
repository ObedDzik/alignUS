from medAI.datasets.nct2013.bmode_dataset import BModeDatasetV1
from medAI.datasets.nct2013.cohort_selection import select_cohort
from medAI.datasets.optimum.cohort_selection import OptimumSplitsGenerator
from medAI.datasets.optimum.needle_trace_dataset import NeedleTraceImageFramesDataset
from medAI.factories.utils.build_dataloader import build_dataloader
from medAI.transforms.prostnfound_transform import ProstNFoundTransform


def get_dataloaders_optimum_kfold(
    n_folds=5,
    fold=0,
    splits_seed=42,
    image_size=256,
    mask_size=64,
    mean=[0, 0, 0],
    std=[1, 1, 1],
):
    # DATASETS
    g = OptimumSplitsGenerator(
        "/home/pwilson/projects/aip-medilab/pwilson/medAI/data/OPTIMUM/processed/UA_annotated_needles/mined_path_reports.csv"
    )
    splits = g.split_patients_kfold(
        k=n_folds, fold=fold, seed=splits_seed
    ).get_cine_id_splits()
    train_transform = ProstNFoundTransform(
        augment="translate",
        image_size=image_size,
        mask_size=mask_size,
        mean=mean,
        std=std,
        flip_ud=False,
    )
    val_transform = ProstNFoundTransform(
        augment="none",
        image_size=image_size,
        mask_size=mask_size,
        mean=mean,
        std=std,
        flip_ud=False,
    )
    train_dataset = NeedleTraceImageFramesDataset(
        "data/OPTIMUM/processed/UA_annotated_needles",
        out_fmt="np",
        cine_ids=splits["train"],
        transform=train_transform,
    )
    val_dataset = NeedleTraceImageFramesDataset(
        "data/OPTIMUM/processed/UA_annotated_needles",
        out_fmt="np",
        cine_ids=splits["val"],
        transform=val_transform,
    )
    train_loader = build_dataloader(
        train_dataset, batch_size=8, shuffle=True, num_workers=4
    )
    val_loader = build_dataloader(
        val_dataset, batch_size=8, shuffle=False, num_workers=4
    )

    return train_loader, val_loader


def get_dataloaders_nct_kfold(
    n_folds=5,
    fold=0,
    splits_seed=42,
    image_size=256,
    mask_size=64,
    mean=[0, 0, 0],
    std=[1, 1, 1],
):
    train_cores, val_cores, test_cores = select_cohort(
        fold, n_folds, exclude_benign_cores_from_positive_patients=True, mode="kfold"
    )
    train_transform = ProstNFoundTransform(
        augment="translate",
        image_size=image_size,
        mask_size=mask_size,
        mean=mean,
        std=std,
        flip_ud=False,
    )
    val_transform = ProstNFoundTransform(
        augment="none",
        image_size=image_size,
        mask_size=mask_size,
        mean=mean,
        std=std,
        flip_ud=False,
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
    val_loader = build_dataloader(
        val_dataset, batch_size=8, shuffle=False, num_workers=4
    )
    return train_loader, val_loader


def get_nct2013_dataset(fold=0, n_folds=5):
    train_cores, val_cores, test_cores = select_cohort(
        fold, n_folds, exclude_benign_cores_from_positive_patients=True, mode="kfold"
    )
    train_transform = ProstNFoundTransform(
        augment="translate", image_size=256, mask_size=64
    )
    val_transform = ProstNFoundTransform(augment="none", image_size=256, mask_size=64)
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

    return train_dataset, val_dataset
