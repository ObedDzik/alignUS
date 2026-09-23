from abc import ABC, abstractmethod
import argparse
from dataclasses import dataclass, field
import json
import os
from typing import Literal

import numpy as np
import sklearn
import sklearn.model_selection
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import InterpolationMode
from torchvision.tv_tensors import Image, Mask
from tqdm import tqdm

from medAI.datasets.nct2013.bmode_dataset import BModeDatasetV1
from medAI.transforms.crop_to_mask import CropToMask
from medAI.transforms.pixel_augmentations import RandomContrast, RandomGamma
from medAI.datasets.nct2013.cohort_selection import (
    get_parser as get_cohort_selection_parser,
    select_cohort_from_args,
)
from medAI.datasets.nct2013.data_access import data_accessor
from medAI.transforms.prostnfound_transform import ProstNFoundTransform
from typing import List, Optional


def get_dataloaders_from_args(args, mode: Literal["train", "test", "heatmap"] = "train"):

    SCHEMA_VERSION = args.get('schema_version', 1)
    if SCHEMA_VERSION != 1: 
        transform_flip_ud = args.flip_ud
    else: 
        transform_flip_ud = False

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

    if SCHEMA_VERSION == 2: 
        from medAI import registry
        train_dataset = registry.build("dataset", split="train", transform=train_transform, **args.dataset)
        val_dataset = registry.build("dataset", split="val", transform=val_transform, **args.dataset)
        try:
            test_dataset = registry.build("dataset", split="test", transform=val_transform, **args.dataset)
        except:
            test_dataset = val_dataset

    elif args.dataset == "optimum":
        from medAI.datasets.optimum import NeedleTraceImageFramesDataset

        # args.root_dir = "/h/pwilson/projects/medAI/data/OPTIMUM/processed/UA_annotated_needles"

        if args.cohort_selection_mode is None or args.cohort_selection_mode in [
            "all",
            "train_only",
        ]:
            train_dataset = NeedleTraceImageFramesDataset(
                root_dir=args.root_dir,
                transform=train_transform,
                needle_mask_fname=(
                    "needle_mask.png" if mode != "heatmap" else "needle_mask_full.png"
                ),
            )
            val_dataset = NeedleTraceImageFramesDataset(
                root_dir=args.root_dir,
                transform=val_transform,
                needle_mask_fname=(
                    "needle_mask.png" if mode != "heatmap" else "needle_mask_full.png"
                ),
            )
            test_dataset = NeedleTraceImageFramesDataset(
                root_dir=args.root_dir,
                transform=val_transform,
                needle_mask_fname=(
                    "needle_mask.png" if mode != "heatmap" else "needle_mask_full.png"
                ),
            )

        elif args.cohort_selection_mode == "train_val":
            cases = [
                p
                for p in os.listdir(args.root_dir)
                if os.path.isdir(os.path.join(args.root_dir, p))
            ]
            train_cases, val_cases = sklearn.model_selection.train_test_split(
                cases,
                test_size=0.2,
                random_state=args.train_subsample_seed,
            )
            train_dataset = NeedleTraceImageFramesDataset(
                root_dir=args.root_dir,
                transform=train_transform,
                case_ids=train_cases,
                needle_mask_fname=(
                    "needle_mask.png" if mode != "heatmap" else "needle_mask_full.png"
                ),
            )
            val_dataset = NeedleTraceImageFramesDataset(
                root_dir=args.root_dir,
                transform=val_transform,
                case_ids=val_cases,
                needle_mask_fname=(
                    "needle_mask.png" if mode != "heatmap" else "needle_mask_full.png"
                ),
            )
            test_dataset = NeedleTraceImageFramesDataset(
                root_dir=args.root_dir,
                transform=val_transform,
                case_ids=[],
                needle_mask_fname=(
                    "needle_mask.png" if mode != "heatmap" else "needle_mask_full.png"
                ),
            )

        elif args.cohort_selection_mode == "kfold":
            cases = [
                p
                for p in os.listdir(args.root_dir)
                if os.path.isdir(os.path.join(args.root_dir, p))
            ]
            skf = sklearn.model_selection.KFold(
                n_splits=args.n_folds,
                shuffle=True,
                random_state=args.train_subsample_seed,
            )
            train_cases, val_cases = [], []
            for fold, (train_index, val_index) in enumerate(skf.split(cases)):
                if fold == args.fold:
                    train_cases = [cases[i] for i in train_index]
                    val_cases = [cases[i] for i in val_index]
                    break
            train_dataset = NeedleTraceImageFramesDataset(
                root_dir=args.root_dir,
                transform=train_transform,
                case_ids=train_cases,
                needle_mask_fname=(
                    "needle_mask.png" if mode != "heatmap" else "needle_mask_full.png"
                ),
            )
            val_dataset = NeedleTraceImageFramesDataset(
                root_dir=args.root_dir,
                transform=val_transform,
                case_ids=val_cases,
                needle_mask_fname=(
                    "needle_mask.png" if mode != "heatmap" else "needle_mask_full.png"
                ),
            )
            test_dataset = NeedleTraceImageFramesDataset(
                root_dir=args.root_dir,
                transform=val_transform,
                case_ids=val_cases,
                needle_mask_fname=(
                    "needle_mask.png" if mode != "heatmap" else "needle_mask_full.png"
                ),
            )

        elif args.cohort_selection_mode == "splits_file":
            splits_file = args.splits_file
            with open(splits_file, "r") as f:
                splits = json.load(f)

            train_cases = splits.get("train", [])
            val_cases = splits.get("val", [])
            test_cases = splits.get("test", [])

            train_dataset = NeedleTraceImageFramesDataset(
                root_dir=args.root_dir,
                transform=train_transform,
                case_ids=train_cases,
                needle_mask_fname=(
                    "needle_mask.png" if mode != "heatmap" else "needle_mask_full.png"
                ),
            )
            val_dataset = NeedleTraceImageFramesDataset(
                root_dir=args.root_dir,
                transform=val_transform,
                case_ids=val_cases,
                needle_mask_fname=(
                    "needle_mask.png" if mode != "heatmap" else "needle_mask_full.png"
                ),
            )
            test_dataset = NeedleTraceImageFramesDataset(
                root_dir=args.root_dir,
                transform=val_transform,
                case_ids=test_cases,
                needle_mask_fname=(
                    "needle_mask.png" if mode != "heatmap" else "needle_mask_full.png"
                ),
            )

        else:
            raise ValueError(
                f"Unknown cohort selection mode: {args.cohort_selection_mode}"
            )

    else:
        train_cores, val_cores, test_cores = select_cohort_from_args(args)

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

        train_dataset = BModeDatasetV1(
            train_cores,
            train_transform,
            rf_as_bmode=args.rf_as_bmode,
            include_rf=args.include_rf,
            flip_ud=args.flip_ud,
            frames=args.frames,
        )
        val_dataset = BModeDatasetV1(
            val_cores,
            val_transform,
            rf_as_bmode=args.rf_as_bmode,
            include_rf=args.include_rf,
            flip_ud=args.flip_ud,
            frames="first",
        )
        test_dataset = BModeDatasetV1(
            test_cores,
            val_transform,
            rf_as_bmode=args.rf_as_bmode,
            include_rf=args.include_rf,
            flip_ud=args.flip_ud,
            frames="first",
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size if mode == "train" else 1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size if mode == "train" else 1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size if mode == "train" else 1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return dict(train=train_loader, val=val_loader, test=test_loader)


def get_kfold_nct2013_prostnfound_datasets(): 
    ...