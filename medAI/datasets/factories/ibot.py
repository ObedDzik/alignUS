import json

import torchvision
from medAI.transforms.ibot import DataAugmentation
import logging
import torch
from torchvision.datasets import ImageFolder
from medAI.datasets.misc.dataset_wrappers import MaskDatasetWrapper
from torch import distributed as dist


def get_ssl_dataloader_from_config(cfg):
    return get_ssl_dataloader(
        crop_augmentation_kw=cfg.transform, 
        batch_size_per_gpu=cfg.batch_size_per_gpu, 
        num_workers=cfg.num_workers, 
        **cfg.data, 
    )


def get_ssl_dataloader(
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

    augmentations = DataAugmentation(**crop_augmentation_kw)
    logging.info(f"Data augmentation: {augmentations}")

    dataset = get_ssl_dataset(**dataset_kw, augmentations=augmentations)

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


def get_ssl_dataset(data_format="default", augmentations=None, **kwargs):
    if data_format == "default":
        dataset = ImageFolder(kwargs["path"], augmentations)
    elif data_format == "nct2013":
        return get_nct2013_ssl_dataset(augmentations=augmentations, **kwargs)
    else:
        raise NotImplementedError(data_format)

    return dataset


def get_nct2013_ssl_dataset(*, patient_ids_fpath, augmentations=None):
    with open(patient_ids_fpath) as f: 
        patient_ids = json.load(f)

    from medAI.datasets.nct2013.cohort_selection import get_core_ids
    core_ids = get_core_ids(patient_ids)
    
    from medAI.datasets.nct2013.bmode_dataset import BModeDatasetV1

    from PIL import Image

    ds = BModeDatasetV1(core_ids)
    
    class NCT2013Wrapper: 
        def __init__(self, ds, transform): 
            self.ds = ds 
            self.transform = transform 

        def __getitem__(self, idx):
            item = self.ds[idx]
            img = Image.fromarray(item['bmode']).convert("RGB")
            label = 0 if item['grade'] == 'Benign' else 1

            if self.transform: 
                img = self.transform(img)

            return img, label

        def __len__(self): 
            return len(self.ds)

    return NCT2013Wrapper(ds, augmentations)



