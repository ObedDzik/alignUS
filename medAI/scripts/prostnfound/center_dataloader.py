from abc import ABC, abstractmethod
import argparse
from dataclasses import dataclass, field
import json
import os
from typing import Literal
from sklearn.model_selection import StratifiedKFold, train_test_split

import numpy as np
import sklearn
import sklearn.model_selection
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import InterpolationMode
from torchvision.tv_tensors import Image, Mask
from tqdm import tqdm

from medAI.datasets.nct2013.bmode_dataset import BModeDatasetV1
from medAI.transforms.crop_to_mask import CropToMask
from medAI.transforms.pixel_augmentations import RandomContrast, RandomGamma
# from medAI.datasets.nct2013.cohort_selection import (
#     get_parser as get_cohort_selection_parser,
#     select_cohort_from_args,
# )
from medAI.datasets.nct2013.data_access import data_accessor
from medAI.transforms.prostnfound_transform import ProstNFoundTransform
from typing import List, Optional


def get_patient_splits_by_center(train_size=0.8, seed=42, centers=None):
    """returns the list of patient ids for the train, val, and test splits."""

    support_ds = {}
    query_ds = {}
    support = {}
    query = {}

    metadata_table = data_accessor.get_metadata_table()
    patient_table = metadata_table.drop_duplicates(subset=["patient_id"])
    table = patient_table[["patient_id", "center"]]
    # generator = torch.Generator().manual_seed(seed)

    for center in centers:
        center_data = table[table.center == center]
        support_size = int(train_size * len(center_data))
        query_size = len(center_data) - support_size

        # support[center], query[center] = random_split(
        #     center_data, [support_size, query_size], generator
        # )

        support[center], query[center] = train_test_split(
            center_data, test_size=query_size, random_state=seed, stratify=center_data["center"]
        )


        support_ds[center] = support[center].patient_id.values.tolist()
        query_ds[center] = query[center].patient_id.values.tolist()

    return support_ds, query_ds

def get_core_ids(patient_ids):
    """returns the list of core ids for the given patient ids."""
    metadata_table = data_accessor.get_metadata_table()
    return metadata_table[
        metadata_table.patient_id.isin(patient_ids)
    ].core_id.values.tolist()

def remove_benign_cores_from_positive_patients(core_ids):
    """Returns the list of cores in the given list that are either malignant or from patients with no malignant cores."""
    table = data_accessor.get_metadata_table().copy()
    table["positive"] = table.grade.apply(lambda g: 0 if g == "Benign" else 1)
    num_positive_for_patient = table.groupby("patient_id").positive.sum()
    num_positive_for_patient.name = "patients_positive"
    table = table.join(num_positive_for_patient, on="patient_id")
    ALLOWED = table.query("positive == 1 or patients_positive == 0").core_id.to_list()
    return [core for core in core_ids if core in ALLOWED]

def remove_cores_below_threshold_involvement(core_ids, threshold_pct):
    """Returns the list of cores with at least the given percentage of cancer cells."""
    table = data_accessor.get_metadata_table().copy()
    ALLOWED = table.query(
        "grade == 'Benign' or pct_cancer >= @threshold_pct"
    ).core_id.to_list()
    return [core for core in core_ids if core in ALLOWED]

def undersample_benign(cores, seed=0, benign_to_cancer_ratio=1):
    """Returns the list of cores with the same cancer cores and the benign cores undersampled to the given ratio."""
    table = data_accessor.get_metadata_table().copy()
    benign = table.query('grade == "Benign"').core_id.to_list()
    cancer = table.query('grade != "Benign"').core_id.to_list()
    import random
    cores_benign = [core for core in cores if core in benign]
    cores_cancer = [core for core in cores if core in cancer]
    rng = random.Random(seed)
    cores_benign = rng.sample(
        cores_benign, int(len(cores_cancer) * benign_to_cancer_ratio)
    )
    return [core for core in cores if core in cores_benign or core in cores_cancer]


def apply_core_filters(
    core_ids,
    exclude_benign_cores_from_positive_patients=False,
    involvement_threshold_pct=None,
    undersample_benign_ratio=None,
):
    if exclude_benign_cores_from_positive_patients:
        core_ids = remove_benign_cores_from_positive_patients(core_ids)

    if involvement_threshold_pct is not None:
        if involvement_threshold_pct < 0 or involvement_threshold_pct > 100:
            raise ValueError(
                f"involvement_threshold_pct must be between 0 and 100, but got {involvement_threshold_pct}"
            )
        core_ids = remove_cores_below_threshold_involvement(
            core_ids, involvement_threshold_pct
        )

    if undersample_benign_ratio is not None:
        core_ids = undersample_benign(
            core_ids, seed=0, benign_to_cancer_ratio=undersample_benign_ratio
        )
    return core_ids


def select_cohort(
    fold=None,
    centers = ["UVA", "CRCEO", "PCC", "PMCC", "JH"],
    exclude_benign_cores_from_positive_patients=False,
    involvement_threshold_pct=None,
    undersample_benign_ratio=None,
    splits_file=None,
    seed=0,
    train_size=0.8,
    mode: Literal[
        "kfold", "nested_kfold", "center", "train_only", None
    ] = None,  # Added mode for flexibility
    return_unfiltered_train_cores=False
):
    """Returns the list of core ids for the given cohort selection criteria.

    Default is to use the 5-fold split.

    Args:
        fold (int): If specified, the fold to use for the train/val/test split.
        n_folds (int): If specified, the number of folds to use for the train/val/test split.
        test_center (str): If specified, the center to use for the test set.

        The following arguments are used to filter the cores in the cohort, affecting
            only the train sets:
        remove_benign_cores_from_positive_patients (bool): If True, remove cores from patients with malignant cores that also have benign cores.
            Only applies to the training set.
        involvement_threshold_pct (float): If specified, remove cores with less than the given percentage of cancer cells.
            this should be a value between 0 and 100. Only applies to the training set.
        undersample_benign_ratio (float): If specified, undersample the benign cores to the given ratio. Only applies to the training set.
        seed (int): Random seed to use for the undersampling.
        splits_file: if specified, use the given csv file to load the train/val/test splits (kfold only)
    """
    support, query = get_patient_splits_by_center(
        train_size=train_size, seed=seed, centers=centers
    )
    support_cores = {}
    query_cores = {}
    unfiltered_support_cores = {}

    for center in centers:
        support_cores[center] = get_core_ids(support[center])
        unfiltered_support_cores[center] = support_cores[center].copy() 
        query_cores[center]= get_core_ids(query[center])

        support_cores[center] = apply_core_filters(
            support_cores[center],
            exclude_benign_cores_from_positive_patients=exclude_benign_cores_from_positive_patients,
            involvement_threshold_pct=involvement_threshold_pct,
            undersample_benign_ratio=undersample_benign_ratio,
        )
    if return_unfiltered_train_cores:
        return support_cores, query_cores, unfiltered_support_cores
    else: 
        return support_cores, query_cores


def get_dataloaders_from_args(args, mode: Literal["train", "test", "heatmap"] = "train"):

    SCHEMA_VERSION = args.get('schema_version', 1)
    if SCHEMA_VERSION != 1: 
        transform_flip_ud = args.flip_ud
    else: 
        transform_flip_ud = False

    centers = ["UVA", "CRCEO", "PCC", "PMCC", "JH"]

    train_transform = ProstNFoundTransform(
        augment=args.augmentations,
        image_size=args.image_size,
        mask_size=args.mask_size,
        mean=args.mean,
        std=args.std,
        crop_to_prostate=args.crop_to_prostate,
        first_downsample_size=args.first_downsample_size,
        return_raw_images=mode != "train",
        grade_group_for_positive_label=vars(args).get(
            "grade_group_for_positive_label", 1
        ),
        flip_ud=transform_flip_ud,
    )
    val_transform = ProstNFoundTransform(
        augment="none",
        image_size=args.image_size,
        mask_size=args.mask_size,
        mean=args.mean,
        std=args.std,
        crop_to_prostate=args.crop_to_prostate,
        first_downsample_size=args.first_downsample_size,
        return_raw_images=mode != "train",
        grade_group_for_positive_label=vars(args).get(
            "grade_group_for_positive_label", 1
        ),
        flip_ud=transform_flip_ud,
    )

    train_cores, val_cores = select_cohort()
    center_loaders = {}

    if args.limit_train_data is not None:
        cores = train_cores
        center = [core.split("-")[0] for core in cores]
        from sklearn.model_selection import StratifiedShuffleSplit

        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=1 - args.limit_train_data,
            random_state=args.train_subsample_seed,
        )
        for train_index, _ in sss.split(cores, center):
            train_cores = [cores[i] for i in train_index]

    for center in centers:

        train_dataset = BModeDatasetV1(
            train_cores[center],
            train_transform,
            rf_as_bmode=args.rf_as_bmode,
            include_rf=args.include_rf,
            flip_ud=args.flip_ud,
            frames=args.frames,
        )
        val_dataset = BModeDatasetV1(
            val_cores[center],
            val_transform,
            rf_as_bmode=args.rf_as_bmode,
            include_rf=args.include_rf,
            flip_ud=args.flip_ud,
            frames="first",
        )
        if center == args.val_center:
            support_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size if mode == "train" else 1,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            query_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size if mode == "train" else 1,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        else:
            support_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size if mode == "train" else 1,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            query_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size if mode == "train" else 1,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
            )


        center_loaders[center]=(support_loader, query_loader)

    return center_loaders