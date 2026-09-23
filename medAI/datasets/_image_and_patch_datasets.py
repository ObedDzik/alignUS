from .registry import register_dataset
import logging


@register_dataset
def nct2013_image_and_patch_v2(
    data_path="/ssd005/projects/exactvu_pca/nct2013/processed/npy_archive/npy_archive_2024-08-04_flipped_256",
    image_size=256,
    patch_size_before_crop=256,
    crop_size=224,
    instance_norm_patches=False,
    augmentation_mode="none",
    fold=0,
    main_data_conf={},
    pretrain_data_conf={},
    mode="ssl",
    include_test=False,
    debug=False,
    **kwargs,
):
    from medAI.datasets.nct2013.nct2013_image_and_patches_v2 import (
        NCT2013PatchesDataset,
        NCT2013PatchesDatasetConf,
        get_image_and_patch_transforms,
    )

    # Load the dataset
    train_transform, val_transform = get_image_and_patch_transforms(
        image_size=image_size,
        patch_size=patch_size_before_crop,
        patch_crop_size=crop_size,
        instance_norm=instance_norm_patches,
        augmentations_mode=augmentation_mode,
    )
    conf = NCT2013PatchesDatasetConf(**main_data_conf)
    logging.info(f"Loading NCT2013 dataset from {data_path}")

    train_dataset = NCT2013PatchesDataset.from_fold(
        root=data_path,
        fold=fold,
        split="train",
        transform=val_transform if mode == "ssl" else train_transform,
        conf=conf,
        debug=debug,
        **kwargs,
    )
    logging.info(f"Loaded {len(train_dataset)} training samples")
    val_dataset = NCT2013PatchesDataset.from_fold(
        root=data_path,
        fold=fold,
        split="val",
        transform=val_transform,
        conf=conf,
        debug=debug,
        **kwargs,
    )
    logging.info(f"Loaded {len(val_dataset)} validation samples")
    if include_test:
        test_dataset = NCT2013PatchesDataset.from_fold(
            root=data_path,
            fold=fold,
            split="test",
            transform=val_transform,
            conf=conf,
            debug=debug,
            **kwargs,
        )
        logging.info(f"Loaded {len(test_dataset)} test samples")

    # pretrain dataset
    if mode == "ssl":
        pretrain_dataset = NCT2013PatchesDataset.from_fold(
            root=data_path,
            fold=fold,
            split="train",
            transform=train_transform,
            conf=NCT2013PatchesDatasetConf(**pretrain_data_conf),
            cohort_selection_kw={"min_involvement_train": None},
            debug=debug,
            **kwargs,
        )
        logging.info(f"Loaded {len(pretrain_dataset)} pretrain samples")

    return {
        "train": train_dataset,
        "val": val_dataset,
        "pretrain": pretrain_dataset if mode == "ssl" else None,
        "test": test_dataset if include_test else None,
    }


@register_dataset
def nct2013_image_and_patch_v3(
    data_path="/ssd005/projects/exactvu_pca/nct2013/processed/npy_archive/npy_archive_2024-08-04_flipped_256",
    image_size=256,
    patch_size_before_crop=256,
    crop_size=224,
    instance_norm_patches=False,
    augmentation_mode="none",
    fold=0,
    main_data_conf={},
    pretrain_data_conf={},
    mode="ssl",
    include_test=False,
    debug=False,
    test_center="UVA",
    **kwargs,
):
    from medAI.datasets.nct2013.nct2013_image_and_patches_v2 import (
        NCT2013PatchesDataset,
        NCT2013PatchesDatasetConf,
        get_image_and_patch_transforms,
    )

    # Load the dataset
    train_transform, val_transform = get_image_and_patch_transforms(
        image_size=image_size,
        patch_size=patch_size_before_crop,
        patch_crop_size=crop_size,
        instance_norm=instance_norm_patches,
        augmentations_mode=augmentation_mode,
    )
    conf = NCT2013PatchesDatasetConf(**main_data_conf)
    logging.info(f"Loading NCT2013 dataset from {data_path}")

    train_dataset = NCT2013PatchesDataset.from_cohort_selection(
        root=data_path,
        fold=fold,
        test_center=test_center,
        split="train",
        transform=val_transform if mode == "ssl" else train_transform,
        conf=conf,
        debug=debug,
        **kwargs,
    )
    logging.info(f"Loaded {len(train_dataset)} training samples")
    val_dataset = NCT2013PatchesDataset.from_cohort_selection(
        root=data_path,
        fold=fold,
        test_center=test_center,
        split="val",
        transform=val_transform,
        conf=conf,
        debug=debug,
        **kwargs,
    )
    assert set(train_dataset.core_ids).isdisjoint(set(val_dataset.core_ids))

    logging.info(f"Loaded {len(val_dataset)} validation samples")
    if include_test:
        test_dataset = NCT2013PatchesDataset.from_cohort_selection(
            root=data_path,
            fold=fold,
            test_center=test_center,
            split="test",
            transform=val_transform,
            conf=conf,
            debug=debug,
            **kwargs,
        )
        logging.info(f"Loaded {len(test_dataset)} test samples")
        assert set(train_dataset.core_ids).isdisjoint(set(test_dataset.core_ids))
        assert set(val_dataset.core_ids).isdisjoint(set(test_dataset.core_ids))

    # pretrain dataset
    if mode == "ssl":
        pretrain_dataset = NCT2013PatchesDataset.from_cohort_selection(
            root=data_path,
            fold=fold,
            test_center=test_center,
            split="pretrain",
            transform=train_transform,
            conf=NCT2013PatchesDatasetConf(**pretrain_data_conf),
            debug=debug,
            **kwargs,
        )
        logging.info(f"Loaded {len(pretrain_dataset)} pretrain samples")
        assert set(pretrain_dataset.core_ids).isdisjoint(set(val_dataset.core_ids))
        if include_test:
            assert set(pretrain_dataset.core_ids).isdisjoint(set(test_dataset.core_ids))

    return {
        "train": train_dataset,
        "val": val_dataset,
        "pretrain": pretrain_dataset if mode == "ssl" else None,
        "test": test_dataset if include_test else None,
    }


from medAI.datasets.bk.datasets import (
    BKPatchesAndImagesDataset,
    BKBmodePNGDataset,
    BKPatchesDataset,
    PatchOptions,
)


@register_dataset
def bk_image_and_patch_v0(
    fold=0,
    augmentations_mode="none",
    image_size=256, 
    mode="default",
    patch_options_main: PatchOptions = PatchOptions(),
    patch_options_pretrain: PatchOptions = PatchOptions(),
):
    def _get_dataset(split, options):
        # load images dataset
        images_dataset = BKBmodePNGDataset.from_splits_file(fold=fold, split=split)
        # load patches dataset
        patches_dataset = BKPatchesDataset.from_splits_file(
            fold=fold, positions_cache_file=".cache", patch_options=options, split=split
        )
        # combine them
        ds = BKPatchesAndImagesDataset(patches_dataset, images_dataset)
        return ds

    pretrain_dataset = _get_dataset("pretrain", options=patch_options_pretrain)
    train_dataset = _get_dataset("train", patch_options_main)
    val_dataset = _get_dataset("val", patch_options_main)
    test_dataset = _get_dataset("test", patch_options_main)

    train_transform, val_transform = pretrain_dataset.get_transforms_v1(
        augmentations_mode=augmentations_mode,
        image_size=image_size,
    )
    pretrain_dataset.transform = train_transform
    train_dataset.transform = val_transform if mode == "ssl" else train_transform
    val_dataset.transform = val_transform
    test_dataset.transform = val_transform

    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
        "pretrain": pretrain_dataset if mode == "ssl" else None,
    }
