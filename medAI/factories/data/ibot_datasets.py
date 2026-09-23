import inspect
import json
from types import SimpleNamespace

from medAI.transforms.ibot import DataAugmentation, DataAugmentationDINOBasic
import logging
import torch
from torchvision.datasets import ImageFolder
from medAI.datasets.misc.dataset_wrappers import MaskDatasetWrapper
from torch import distributed as dist
from torchvision.transforms import v2 as T
from torch import distributed as dist
from medAI import registry


@registry.register("dataset", "dino_ssl_dataset")
def get_dataset(data_format="default", split="ssl", transform=None, **kwargs):
    if data_format == "default":
        dataset = ImageFolder(kwargs["path"], transform)
    elif data_format == "nct2013":
        return nct2013_ssl_dataset(transform=transform, split=split, **kwargs)
    elif data_format == "cifar10":
        return cifar10_ssl_dataset(transform=transform, split=split, **kwargs)
    elif data_format == "tiny-imagenet":
        return tiny_imagenet_ssl_dataset(transform=transform, split=split, **kwargs)
    elif data_format == "breastmnist":
        return medmnist_dataset(
            flag="breast", transform=transform, split=split, **kwargs
        )
    elif data_format == "tissuemnist":
        dataset = medmnist_dataset(
            flag="tissue", transform=transform, split=split, **kwargs
        )
    elif data_format == "bloodmnist":
        return medmnist_dataset(
            flag="blood", transform=transform, split=split, **kwargs
        )
    elif data_format == "pneumoniamnist":
        return medmnist_dataset(
            flag="pneumonia", transform=transform, split=split, **kwargs
        )
    elif data_format == "pathologymnist":
        return medmnist_dataset(
            flag="pathology", transform=transform, split=split, **kwargs
        )
    elif data_format == "imagenet100":
        return imagenet100_ssl_dataset(transform=transform, split=split, **kwargs)
    elif data_format == "bk":
        return bk_dataset(transform=transform, split=split, **kwargs)
    elif data_format == "bk_minh":
        return bk_dataset_minh_version(transform=transform, split=split, **kwargs)
    else:
        from medAI.datasets import create_dataset
        split = 'train' if split in ['train', 'ssl'] else split
        return create_dataset(name=data_format, split=split, transform=transform, **kwargs)

    return dataset


def nct2013_ssl_dataset(*, patient_ids_fpath, split, transform=None):
    # with open(patient_ids_fpath) as f:
    #     patient_ids = json.load(f)

    from medAI.datasets.nct2013.cohort_selection import get_core_ids, get_patient_splits
    patient_ids, *_ = get_patient_splits(validation_fold=0)

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
            img = Image.fromarray(item["bmode"]).convert("RGB")
            label = 0 if item["grade"] == "Benign" else 1

            if self.transform:
                img = self.transform(img)

            return img, label

        def __len__(self):
            return len(self.ds)

    return NCT2013Wrapper(ds, transform)


def cifar10_ssl_dataset(*, split="ssl", transform=None, **kwargs):
    from src.datasets.cifar10 import CIFAR10Subsample

    train = split in ["train", "ssl"]

    return CIFAR10Subsample(
        root="data",
        train=train,
        download=True,
        transform=transform,
        **kwargs,
    )


def tiny_imagenet_ssl_dataset(*, split="ssl", transform=None, **kwargs):
    from medAI.datasets.computer_vision.tiny_imagenet import TinyImageNetDataset

    train = split in ["train", "ssl"]

    return TinyImageNetDataset(
        path="/fs01/projects/exactvu_pca/tiny-imagenet-200",
        split="train" if train else "val",
        transform=transform,
    )


def medmnist_dataset(*, split="ssl", transform=None, flag="breast", **kwargs):
    if split in ["train", "ssl"]:
        split = "train"
    else:
        split = "val"

    from medmnist import BreastMNIST, TissueMNIST, BloodMNIST, PathMNIST, PneumoniaMNIST

    if flag == "breast":
        dataset = BreastMNIST(
            split=split,
            transform=transform,
            download=True,
            root="data",
            as_rgb=True,
            size=224,
            **kwargs,
        )
    elif flag == "tissue":
        dataset = TissueMNIST(
            split=split,
            transform=transform,
            download=True,
            root="data",
            as_rgb=True,
            size=224,
            **kwargs,
        )
    elif flag == "blood":
        dataset = BloodMNIST(
            split=split,
            transform=transform,
            download=True,
            root="data",
            as_rgb=True,
            size=224,
            **kwargs,
        )
    elif flag == "pathology":
        dataset = PathMNIST(
            split=split,
            transform=transform,
            download=True,
            root="data",
            as_rgb=True,
            size=224,
            **kwargs,
        )
    elif flag == "pneumonia":
        dataset = PneumoniaMNIST(
            split=split,
            transform=transform,
            download=True,
            root="data",
            as_rgb=True,
            size=224,
            **kwargs,
        )
    return dataset


def imagenet100_ssl_dataset(*, split="ssl", transform=None, **kwargs):
    from medAI.datasets.computer_vision.imagenet import ImageNet100

    if split in ["train", "ssl"]:
        split = "train"
    else:
        split = "val"

    return ImageNet100(split=split, transform=transform, **kwargs)


def bk_dataset(*, split="ssl", transform=None, **kwargs):

    if split in ["train", "ssl"]:
        split = "train"
    else:
        split = "val"
    from medAI.datasets.bk.datasets import BKBmodePNGDataset
    ds = BKBmodePNGDataset.from_splits_file(fold=0, split=split, api_compatibility='torchvision', transform=transform)
    return ds


def bk_dataset_minh_version(*, split='ssl', transform=None, fold_idx=0, **kwargs):
    from medAI.datasets.bk import bk_ubc as bk
    train = split in ['train', 'ssl']
    return bk.get_dataset(
        fold_idx=fold_idx,
        set_name='train' if train else 'val',
        transform=transform
    )