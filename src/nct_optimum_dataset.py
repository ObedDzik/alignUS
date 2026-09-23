import os
import json
import random
import typing as tp
from collections import Counter
import math
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class MergedDataset(Dataset):
    """
    Unified dataset that merges BModeDatasetV2 (center 1) and one or more
    NeedleTraceImageFramesDataset instances (centers 2, 3) into a single
    interface compatible with GradeStratifiedSampler.

    All items are normalized to the same output dict schema.
    Histopathology and MRI contrastive sampling is applied uniformly
    regardless of which sub-dataset the item came from.

    Attributes exposed for GradeStratifiedSampler (unchanged):
        .df         — pd.DataFrame with columns [core_id, grade_group]
        .core_ids   — list of str, one per unique core
        ._indices   — list of (core_idx, frame_idx) tuples
    """

    def __init__(
        self,
        # Sub-datasets: each entry is (dataset_instance, adapter_fn_or_None)
        # adapter_fn receives the raw item dict and returns a normalized dict
        sub_datasets: list,           # list of (Dataset, callable|None)
        hist_groups: dict,            # {isup_grade: [[image_id, cancer_pct], ...]}
        mri_groups: dict,             # {isup_grade: [[case_id, z], ...]}
        hist_root: str,
        mri_root: str,
        transform=None,
    ):
        self.sub_datasets = sub_datasets
        self.hist_groups = hist_groups
        self.mri_groups = mri_groups
        self.hist_root = hist_root
        self.mri_root = mri_root
        self.transform = transform

        # Build unified index: (sub_dataset_idx, local_item_idx)
        # Each local_item_idx maps 1-to-1 to a core (no multi-frame expansion
        # here — frames are handled inside each sub-dataset already).
        self._sub_indices = []   # list of (sub_ds_idx, local_idx)
        core_records = []        # for building self.df

        for sub_idx, (ds, adapter) in enumerate(sub_datasets):
            for local_idx in range(len(ds)):
                self._sub_indices.append((sub_idx, local_idx))

            # Try to extract grade/core metadata for GradeStratifiedSampler.
            # BModeDatasetV2 exposes .df and .core_ids; for
            # NeedleTraceImageFramesDataset we pull from .data[i]["info"].
            if hasattr(ds, "df") and hasattr(ds, "core_ids"):
                # BModeDatasetV2 path
                for core_idx, frame_idx in ds._indices:
                    core_id = str(ds.core_ids[core_idx]).strip()
                    row = ds.df[ds.df["core_id"].astype(str).str.strip() == core_id]
                    if not row.empty:
                        grade = int(row.iloc[0]["grade_group"])
                    else:
                        grade = 0
                    core_records.append({
                        "core_id": f"sub{sub_idx}__{core_id}",
                        "grade_group": grade,
                    })
            else:
                # NeedleTraceImageFramesDataset path
                for local_idx in range(len(ds)):
                    info = ds.data[local_idx]["info"]
                    core_id = f"sub{sub_idx}__{info.get('cine_id', str(local_idx))}"
                    # grade = int(info.get("GG", 0))
                    value = info.get("GG", 0)
                    grade = 0 if math.isnan(value) else int(value)
                    core_records.append({
                        "core_id": core_id,
                        "grade_group": grade,
                    })

        # Unified metadata frame — GradeStratifiedSampler reads .df and .core_ids
        self.df = pd.DataFrame(core_records)
        self.core_ids = self.df["core_id"].tolist()

        # ._indices expected by GradeStratifiedSampler:
        # list of (core_idx, anything) — we use 0 as a dummy frame_idx
        self._indices = [(i, 0) for i in range(len(self._sub_indices))]

    # ------------------------------------------------------------------
    # Sampling helpers (same logic as BModeDatasetV2, centralised here)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_hard_negative_label(label: int, valid_classes: list) -> int:
        candidates = [c for c in valid_classes if c != label]
        if random.random() < 0.8:
            adjacent = []
            lower = [c for c in valid_classes if c < label]
            upper = [c for c in valid_classes if c > label]
            if lower:
                adjacent.append(max(lower))
            if upper:
                adjacent.append(min(upper))
            return random.choice(adjacent) if adjacent else random.choice(candidates)
        return random.choice(candidates)

    def _select_hist(self, grade: int, involvement: float, max_tries: int = 10):
        group = self.hist_groups.get(grade, [])
        if not group:
            raise ValueError(f"No hist samples for grade {grade}")
            logging.info("No grade")
        if grade == 0:
            candidates = list(group)
        else:
            candidates = [e for e in group if abs(e[1] - involvement) <= 15]
            if not candidates:
                candidates = [e for e in group if abs(e[1] - involvement) <= 30]
            if not candidates:
                candidates = list(group)
        random.shuffle(candidates)
        tried_ids = []
        for tried, entry in enumerate(candidates[:max_tries]):
            sample_id = entry[0]
            tried_ids.append(sample_id)
            path = os.path.join(self.hist_root, f"{sample_id}.npz")
            if os.path.isfile(path):
                try:
                    emb = np.load(path)["embedding"]
                    return sample_id, emb
                except Exception:
                    pass
        raise FileNotFoundError(f"No valid hist file for grade={grade}, tried {tried_ids}")

    def _select_mri(self, grade: int, max_tries: int = 10):
        group = self.mri_groups.get(grade, [])
        if not group:
            raise ValueError(f"No mri samples for grade {grade}")
        candidates = list(group)
        random.shuffle(candidates)
        for entry in candidates[:max_tries]:
            case_id, z = entry
            path = os.path.join(self.mri_root, f"{case_id}_{z}.npy")
            if os.path.isfile(path):
                try:
                    emb = np.load(path, mmap_mode="r")
                    return (case_id, z), emb
                except Exception:
                    pass
        raise FileNotFoundError(f"No valid mri file for grade={grade}")

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self._sub_indices)

    def __getitem__(self, idx):
        sub_ds_idx, local_idx = self._sub_indices[idx]
        ds, adapter = self.sub_datasets[sub_ds_idx]
        raw = ds[local_idx]

        # Normalize via adapter if provided
        item = adapter(raw) if adapter is not None else raw
        # At this point item must contain at minimum:
        #   bmode, needle_mask, prostate_mask, grade_group, pct_cancer
        # (BModeDatasetV2 already returns these; the adapter does the same
        #  for NeedleTraceImageFramesDataset items.)

        if self.transform is not None:
            item = self.transform(item)

        isup = int(item.get("grade_group", 0))
        involvement = float(item.get("pct_cancer", 0.0))
        if isup == 0:
            involvement = 0.0

        valid_classes = [g for g in self.hist_groups if g in [0, 1, 2, 3, 4, 5]]

        # Positive hist
        _, pos_hist_emb = self._select_hist(isup, involvement)
        pos_hist = torch.from_numpy(pos_hist_emb.astype(np.float32).copy())

        # Negative hist
        neg_hist_grade = self._get_hard_negative_label(isup, valid_classes)
        _, neg_hist_emb = self._select_hist(neg_hist_grade, involvement)
        neg_hist = torch.from_numpy(neg_hist_emb.astype(np.float32).copy())

        # Positive MRI
        _, pos_mri_emb = self._select_mri(isup)
        pos_mri = (
            torch.from_numpy(pos_mri_emb.astype(np.float32).copy())
            if pos_mri_emb is not None else None
        )

        # Negative MRI
        neg_mri_grade = self._get_hard_negative_label(isup, valid_classes)
        _, neg_mri_emb = self._select_mri(neg_mri_grade)
        neg_mri = (
            torch.from_numpy(neg_mri_emb.astype(np.float32).copy())
            if neg_mri_emb is not None else None
        )

        item.update({
            "positive_hist": pos_hist,
            "negative_hist": neg_hist,
            "positive_mri": pos_mri,
            "negative_mri": neg_mri,
            "negative_hist_grade": neg_hist_grade,
            "negative_mri_grade": neg_mri_grade,
            "bucket_label": isup,
        })

        return item