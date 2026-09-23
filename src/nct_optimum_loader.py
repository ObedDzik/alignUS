from abc import ABC, abstractmethod
import argparse
from dataclasses import dataclass, field
import json
import os
from typing import Literal
import pandas as pd

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
from collections import defaultdict
from src.bmode_dataset import BModeDatasetV2
from src.needle_trace_dataset_ttt import NeedleTraceImageFramesDataset
from src.nct_optimum_dataset import MergedDataset
from src.isup_stratify import GradeStratifiedSampler, GradeBalancedBatchSampler



def _optimum_adapter(item: dict) -> dict:
    """
    Normalize a NeedleTraceImageFramesDataset item to the shared schema.
    Works whether the sub-dataset returns PIL or numpy (out_fmt either way).
    """
    import numpy as np

    image = item["image"]
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    # Grayscale: take first channel if RGB
    if image.ndim == 3:
        image = image[..., 0]

    needle_mask = item["needle_mask"]
    if not isinstance(needle_mask, np.ndarray):
        needle_mask = np.array(needle_mask)

    prostate_mask = item.get("prostate_mask")
    if prostate_mask is not None:
        if not isinstance(prostate_mask, np.ndarray):
            prostate_mask = np.array(prostate_mask)
    else:
        prostate_mask = np.ones_like(needle_mask, dtype=np.uint8)

    info = item.get("info", {})

    return {
        "bmode": image,
        "needle_mask": needle_mask,
        "prostate_mask": prostate_mask,
        "grade": info.get("Diagnosis", "Unknown"),
        "pct_cancer": float(info.get("% Cancer", 0.0)),
        "psa": float(info.get("psa", 0.0)),
        "age": float(info.get("age", 0.0)),
        "approx_psa_density": float(info.get("approx_psa_density", 0.0)),
        "family_history": info.get("family_history", float("nan")),
        "center": info.get("center", "Unknown"),
        "all_cores_benign": info.get("all_cores_benign", False),
        "core_id": info.get("cine_id", "Unknown"),
        "patient_id": info.get("case", "Unknown"),
        "loc": info.get("Sample ID", "Unknown"),
        "grade_group": int(0 if pd.isna(info.get("GG")) else info.get("GG")), #int(info.get("GG", 0)),
        "clinically_significant": info.get("clinically_significant", False),
        "involvement": float(info.get("% Cancer", 0.0)) / 100.0,
    }


def _build_hist_mri_groups(hist_csv: str, mri_csv: str, data_provider: str):
    """
    Build the grade-keyed sample groups consumed by MergedDataset's samplers.
    Mirrors the logic previously inlined in BModeDatasetV2.__init__.
    """
    hist_df = pd.read_csv(hist_csv)
    # hist_df = hist_df[hist_df["data_provider"] == "karolinska"]
    hist_df = hist_df[hist_df["data_provider"] == data_provider]

    mri_df = pd.read_csv(mri_csv)
    mri_df = mri_df[
        (mri_df["merged_ISUP"] <= 1) |
        ((mri_df["merged_ISUP"] >= 2) & (mri_df["has_lesion"] == 1))
    ]

    hist_groups = {}
    for grade in hist_df["isup_grade"].unique():
        # if grade != 1:
        hist_groups[grade] = (
            hist_df[hist_df["isup_grade"] == grade][["image_id", "cancer_percentage"]]
            .values.tolist()
        )

    mri_groups = {}
    for grade in mri_df["merged_ISUP"].unique():
        # if grade != 1:
        mri_groups[grade] = (
            mri_df[mri_df["merged_ISUP"] == grade][["case_id", "z"]]
            .values.tolist()
        )

    return hist_groups, mri_groups

def debug_collate(batch):
    from torch.utils.data._utils.collate import default_collate
    for key in batch[0].keys():
        vals = [item[key] for item in batch]
        try:
            default_collate([{key: v} for v in vals])
        except Exception as e:
            shapes = [v.shape if hasattr(v, 'shape') else type(v) for v in vals]
            raise RuntimeError(f"Collate failed on key '{key}': shapes={shapes}") from e
    return default_collate(batch)


def get_dataloaders(args, mode: Literal["train", "test", "heatmap"] = "train"):
    from torch.utils.data import DataLoader
    from sklearn.model_selection import KFold, train_test_split

    train_transform = ProstNFoundTransform(
        augment=args.augmentations,
        image_size=args.image_size,
        mask_size=args.mask_size,
        mean=args.mean,
        std=args.std,
        crop_to_prostate=args.crop_to_prostate,
        first_downsample_size=args.first_downsample_size,
        return_raw_images=mode != "train",
        grade_group_for_positive_label=vars(args).get("grade_group_for_positive_label", 1),
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
        grade_group_for_positive_label=vars(args).get("grade_group_for_positive_label", 1),
    )

    hist_groups, mri_groups = _build_hist_mri_groups(
        args.hist_csv,   # path to clean_labels_with_cancer_pct.csv
        args.mri_csv,    # path to slices_manifest.csv
        args.hist_data_provider,
    )

    shared_sampler_kwargs = dict(
        hist_groups=hist_groups,
        mri_groups=mri_groups,
        hist_root=args.histo_emb_dir,
        mri_root=args.mri_emb_dir,
    )

    # ------------------------------------------------------------------
    # Build center-1 sub-datasets (BModeDatasetV2) — no adapter needed
    # ------------------------------------------------------------------
    train_cores_c1, val_cores_c1, test_cores_c1 = select_cohort_from_args(args)

    # Optionally subsample training cores
    if args.limit_train_data is not None:
        from sklearn.model_selection import StratifiedShuffleSplit
        centers = [c.split("-")[0] for c in train_cores_c1]
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=1 - args.limit_train_data,
            random_state=args.train_subsample_seed
        )
        for train_idx, _ in sss.split(train_cores_c1, centers):
            train_cores_c1 = [train_cores_c1[i] for i in train_idx]

    # Note: histo/MRI sampling is now in MergedDataset, so we pass
    # histo_emb_dir=None / mri_emb_dir=None to avoid double-sampling in
    # BModeDatasetV2.  If your BModeDatasetV2 raises on None, gate the
    # sampling there behind `if self.hist_root is not None`.
    def make_bmode(cores, frames="first"):
        return BModeDatasetV2(
            cores, transform=None,
            rf_as_bmode=args.rf_as_bmode,
            include_rf=args.include_rf,
            flip_ud=args.flip_ud,
            frames=frames,
        )

    # ------------------------------------------------------------------
    # Build center-2 and center-3 sub-datasets (NeedleTraceImageFrames)
    # ------------------------------------------------------------------
    if args.cohort_selection_mode == "kfold":
        all_cases_c23 = [
            p for p in os.listdir(args.root_dir_c23)
            if os.path.isdir(os.path.join(args.root_dir_c23, p))
        ]

        all_cases_c3 = [
            p for p in os.listdir(args.root_dir_c3)
            if os.path.isdir(os.path.join(args.root_dir_c3, p)) and p[:2]==args.optimum_center
        ]

        skf = KFold(n_splits=args.n_folds, shuffle=True,
                    random_state=args.train_subsample_seed)
        train_cases_c23, val_cases_c23 = [], []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(all_cases_c23)):
            if fold == args.fold:
                train_cases_c23 = [all_cases_c23[i] for i in tr_idx]
                val_cases_c23   = [all_cases_c23[i] for i in va_idx]
                break
        train_cases_c3, val_cases_c3 = [], []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(all_cases_c3)):
            if fold == args.fold:
                train_cases_c3 = [all_cases_c3[i] for i in tr_idx]
                val_cases_c3   = [all_cases_c3[i] for i in va_idx]
                break

    elif args.cohort_selection_mode == "splits_file":
        with open(args.splits_file) as f:
            splits = json.load(f)
        train_cases_c23 = splits.get("train", [])
        val_cases_c23   = splits.get("val", [])
        test_cases_c23  = splits.get("test", [])

    else:  # "train_val" or None — simple 80/20
        all_cases_c23 = [
            p for p in os.listdir(args.root_dir_c23)
            if os.path.isdir(os.path.join(args.root_dir_c23, p))
        ]
        train_cases_c23, val_cases_c23 = train_test_split(
            all_cases_c23, test_size=0.2,
            random_state=args.train_subsample_seed
        )
        test_cases_c23 = []

    needle_mask_fname = "needle_mask.png" if mode != "heatmap" else "needle_mask_full.png"

    def make_needle(cases, root=None):
        return NeedleTraceImageFramesDataset(
            root_dir=root or args.root_dir_c23,
            case_ids=cases if cases else [],
            needle_mask_fname=needle_mask_fname,
            out_fmt="np",   # MergedDataset / adapter expects numpy
            transform=None
        )

    # Center 3 test dataset (separate root if different dir)
    # c3_test_cases = [
    #     p for p in os.listdir(args.root_dir_c3)
    #     if os.path.isdir(os.path.join(args.root_dir_c3, p)) and p[:2]==args.optimum_center
    # ]

    # ------------------------------------------------------------------
    # Assemble MergedDatasets
    # ------------------------------------------------------------------
    # def make_merged(bmode_ds, needle_ds, transform):
    #     sub_datasets = [
    #         (bmode_ds, None),           # center 1 — no adapter, already normalized
    #         (needle_ds, _optimum_adapter),  # centers 2/3
    #     ]
    #     return MergedDataset(
    #         sub_datasets=sub_datasets,
    #         transform=transform,
    #         **shared_sampler_kwargs,
    #     )

    train_dataset = MergedDataset(
        sub_datasets=[
            (make_bmode(train_cores_c1, frames=args.frames), None),
            (make_needle(train_cases_c23), _optimum_adapter),
            (make_needle(train_cases_c3, root=args.root_dir_c3), _optimum_adapter),  # <-- add
        ],
        transform=train_transform,
        **shared_sampler_kwargs,
    )
    val_dataset = MergedDataset(
        sub_datasets=[
            (make_bmode(val_cores_c1, frames="first"), None),
            (make_needle(val_cases_c23), _optimum_adapter),
            (make_needle(val_cases_c3, root=args.root_dir_c3), _optimum_adapter),    # <-- add
        ],
        transform=val_transform,
        **shared_sampler_kwargs,
    )
# test_dataset is now empty / unused
    # test_dataset = MergedDataset(
    #     sub_datasets=[
    #         (make_needle(c3_test_cases, root=args.root_dir_c3), _optimum_adapter),
    #     ],
    #     transform=val_transform,
    #     **shared_sampler_kwargs,
    #     )

    # sampler = GradeStratifiedSampler(train_dataset).build()
    batch_sampler = GradeBalancedBatchSampler(
        train_dataset,
        samples_per_grade=args.samples_per_grade,  # e.g. 4, giving batch size 24
        num_grades=6,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,  # replaces batch_size + sampler
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=debug_collate,
    )

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size if mode == "train" else 1,
    #     sampler=sampler,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     collate_fn=debug_collate
    # )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size if mode == "train" else 1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_loader = []
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=args.batch_size if mode == "train" else 1,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    # )
    print(f"train: {len(train_dataset)}  val: {len(val_dataset)}")

    # print(f"train: {len(train_dataset)}  val: {len(val_dataset)}  test: {len(test_dataset)}")
    return dict(train=train_loader, val=val_loader, test=test_loader)