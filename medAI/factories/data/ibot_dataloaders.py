import json
from types import SimpleNamespace


from medAI.factories.transforms import get_ibot_ssl_transform
import logging
import torch
from torchvision.datasets import ImageFolder
from medAI.datasets.misc.dataset_wrappers import MaskDatasetWrapper
from torch import distributed as dist
from torchvision.transforms import v2 as T
from torch import distributed as dist

from .ibot_datasets import get_dataset
from medAI import registry


def get_multicrop_ssl_dataloader_from_config(cfg, **kwargs):
    data_cfg = dict(**cfg.data)
    data_cfg.update(kwargs)
    return get_multicrop_ssl_dataloader(
        crop_augmentation_kw=cfg.transform,
        **data_cfg,
    )


def get_multicrop_ssl_dataloader(
    *,
    crop_augmentation_kw={},
    patch_size=16,
    pred_ratio=0.3,
    pred_ratio_var=0,
    pred_shape="block",
    pred_start_epoch=0,
    meanfill_masked_pixel=False,
    batch_size_per_gpu=32,
    num_workers=4,
    **dataset_kw,
):

    transform = get_ibot_ssl_transform(**crop_augmentation_kw)

    if 'path' in dataset_kw:
        dataset = ImageFolder(dataset_kw.pop('path'), transform=transform)
    else:
        dataset = registry.build("dataset", **dataset_kw, transform=transform)
    
    dataset = MaskDatasetWrapper(
        dataset,
        patch_size=patch_size,
        pred_ratio=pred_ratio,
        pred_ratio_var=pred_ratio_var,
        pred_aspect_ratio=(0.3, 1 / 0.3),
        pred_shape=pred_shape,
        pred_start_epoch=pred_start_epoch,
        meanfill_masked_pixel=meanfill_masked_pixel,
    )

    # sampler
    sampler = (
        torch.utils.data.DistributedSampler(dataset, shuffle=True)
        if dist.is_initialized()
        else None
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        shuffle=None if dist.is_initialized() else True,
        batch_size=batch_size_per_gpu,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logging.info(f"Data loaded: there are {len(dataset)} images.")

    return data_loader


def get_supervised_dataloaders_from_config(cfg):
    return get_supervised_dataloaders(crop_augmentation_kw=cfg.transform, **cfg.data)


def get_supervised_dataloaders(
    batch_size_per_gpu=32,
    num_workers=4,
    crop_augmentation_kw={},
    **dataset_kw,
):

    dataset_kw = dataset_kw.copy()
    extra_keys = [
        "patch_size",
        "pred_ratio",
        "pred_ratio_var",
        "pred_shape",
        "pred_start_epoch",
        "meanfill_masked_pixel",
    ]
    for key in extra_keys:
        dataset_kw.pop(key, None)

    crop_augmentation_kw_ = dict(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        global_crops_size=224,
    )
    crop_augmentation_kw_.update(crop_augmentation_kw)
    crop_augmentation_kw = SimpleNamespace(**crop_augmentation_kw_)

    transform = T.Compose(
        [
            T.Resize(
                (
                    crop_augmentation_kw.global_crops_size,
                    crop_augmentation_kw.global_crops_size,
                )
            ),
            T.ToTensor(),
            T.Normalize(mean=crop_augmentation_kw.mean, std=crop_augmentation_kw.std),
        ]
    )

    train_dataset = get_dataset(**dataset_kw, transform=transform, split="train")
    val_dataset = get_dataset(**dataset_kw, transform=transform, split="val")

    if dist.is_initialized():
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,
        num_workers=num_workers,
        sampler=train_sampler,
        shuffle=True if train_sampler is None else False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size_per_gpu,
        num_workers=num_workers,
        sampler=val_sampler,
        shuffle=False if val_sampler is None else False,
    )
    return dict(train=train_loader, val=val_loader)
