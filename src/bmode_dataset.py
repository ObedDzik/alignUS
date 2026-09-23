import typing as tp
from enum import StrEnum

import numpy as np
from PIL import Image
import skimage
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import torch
import os
import random

from medAI.datasets.nct2013.utils import apply_colormap
from medAI.datasets.registry import register_dataset

from medAI.datasets.nct2013.data_access import data_accessor


class DataKeys(StrEnum):
    BMODE = "bmode"
    PROSTATE_MASK = "prostate_mask"
    NEEDLE_MASK = "needle_mask"
    CENTER = "center"
    RF = "rf"
    CORE_ID = "core_id"
    FRAME_IDX = "frame_idx"
    PSA = "psa"
    PRIMARY_GRADE = "primary_grade"
    SECONDARY_GRADE = "secondary_grade"
    AGE = "age"
    FAMILY_HISTORY = "family_history"


class BModeDatasetV1(Dataset):
    """Dataset for B-mode images.

    Samples are dictionaries with the following keys:
        bmode_frames (np.ndarray): Shape (H, W) uint8
        prostate_mask (np.ndarray): Shape (H, W)
        needle_mask (np.ndarray): Shape (H, W)
        core_id (str): Core id of the sample.
        frame_idx (int): Frame index of the sample.
        psa (float): PSA value of the sample.
        primary_grade (int): Primary grade of the sample.
        secondary_grade (int): Secondary grade of the sample.
        age (int): Age of the sample.
        family_history (int): Family history of the sample.
        ... etc (other metadata)

    Args:
        core_ids (list): List of core ids to include in the dataset.
        transform (callable): Transform to apply to each sample.
        frames (str): If 'first', only the first frame is returned. If 'all', all frames are returned.
    """

    def __init__(
        self,
        core_ids=None,
        transform=None,
        frames: tp.Literal["first", "all", "all_concat"] = "first",
        include_rf=False,
        rf_as_bmode=False,
        api_compatibility=None,
        apply_colormap=True,
        flip_ud=False,
        core_selection_kw={}, 
        split=None,
        output_fmt='numpy',
    ):

        if core_ids is None:
            if split is not None: 
                from medAI.datasets.nct2013.cohort_selection import select_cohort
                train, val, test = select_cohort(**core_selection_kw)
                if split == "train":
                    core_ids = train
                elif split == "val":
                    core_ids = val
                elif split == "test":
                    core_ids = test
            else:
                print("Using all cores.")
                core_ids = data_accessor.get_metadata_table().core_id.unique().tolist()

        self.metadata = data_accessor.get_metadata_table().copy()
        self.metadata = self.metadata[self.metadata.core_id.isin(core_ids)]

        self._indices = []
        self.core_ids = core_ids
        for i, core_id in enumerate(tqdm(self.core_ids, desc="Loading dataset")):
            if frames == "first":
                self._indices.append((i, 0))
            elif frames == "all":
                n = data_accessor.get_num_frames(core_id)
                self._indices.extend([(i, j) for j in range(n)])
            elif frames == "all_concat": 
                n = data_accessor.get_num_frames(core_id)
                self._indices.append((i, np.arange(n)))

        self.transform = transform
        self.include_rf = include_rf
        self.rf_as_bmode = rf_as_bmode
        self.api_compatibility = api_compatibility
        self.apply_colormap = apply_colormap
        self.flip_ud = flip_ud
        self.frames = frames
        self.output_fmt = output_fmt

    def __len__(self):
        return len(self._indices)

    def get_patient_id(self, idx):
        """
        Extract patient ID from core_id.
        core_id format: "center-patientID-coreNumber"
        e.g. "UVA-00123-4" -> patient_id = "UVA-00123"
        Adjust the split logic to match your actual core_id format.
        """
        core_idx, frame_idx = self._indices[idx]
        core_id = self.core_ids[core_idx]
        patient_id = core_id.split("_")[0]
        return patient_id

    def __getitem__(self, idx):
        core_idx, frame_idx = self._indices[idx]
        core_id = self.core_ids[core_idx]

        if not self.rf_as_bmode:
            bmode = data_accessor.get_bmode_image(core_id, frame_idx)
        else:
            bmode = data_accessor.get_rf_image(core_id, frame_idx)
        h, w, *_ = bmode.shape

        if self.apply_colormap:
            bmode = apply_colormap(bmode)

        prostate_mask = data_accessor.get_prostate_mask(core_id)
        prostate_mask = skimage.transform.resize(
            prostate_mask, (h, w), order=0, preserve_range=True, anti_aliasing=False
        ).astype(np.uint8)
        needle_mask = data_accessor.get_needle_mask(core_id)
        needle_mask = skimage.transform.resize(
            needle_mask, (h, w), order=0, preserve_range=True, anti_aliasing=False
        ).astype(np.uint8)

        if self.flip_ud: 
            bmode = np.flipud(bmode)
            prostate_mask = np.flipud(prostate_mask)
            needle_mask = np.flipud(needle_mask)

        metadata = self.metadata[self.metadata.core_id == core_id].iloc[0].to_dict()
        metadata["frame_idx"] = frame_idx

        output = {
            "bmode": bmode,
            "prostate_mask": prostate_mask,
            "needle_mask": needle_mask,
            "orientation": "anterior_is_top" if self.flip_ud else "anterior_is_bottom",
            **metadata,
        }
        if self.output_fmt == 'pil': 
            output["bmode"] = Image.fromarray(output["bmode"]).convert("RGB")
            output["prostate_mask"] = Image.fromarray(output["prostate_mask"])
            output["needle_mask"] = Image.fromarray(output["needle_mask"])

        if self.include_rf:
            rf = data_accessor.get_rf_image(core_id, frame_idx)
            output["rf"] = rf

        if self.api_compatibility == "torchvision":
            image = Image.fromarray(output["bmode"]).convert("RGB")
            label = int(output["grade"] != "Benign")
            if self.transform:
                image = self.transform(image)
            return image, label

        if self.transform:
            output = self.transform(output)
        return output


class BModeDatasetV2(Dataset):
    def __init__(
        self,
        core_ids=None,
        transform=None,
        frames: tp.Literal["first", "all", "all_concat"] = "first",
        include_rf=False,
        rf_as_bmode=False,
        api_compatibility=None,
        apply_colormap=True,
        flip_ud=False,
        core_selection_kw={}, 
        split=None,
        output_fmt='numpy',
        histo_emb_dir=None,
        mri_emb_dir=None
    ):

        if core_ids is None:
            if split is not None: 
                from medAI.datasets.nct2013.cohort_selection import select_cohort
                train, val, test = select_cohort(**core_selection_kw)
                if split == "train":
                    core_ids = train
                elif split == "val":
                    core_ids = val
                elif split == "test":
                    core_ids = test
            else:
                print("Using all cores.")
                core_ids = data_accessor.get_metadata_table().core_id.unique().tolist()
        self.metadata = data_accessor.get_metadata_table().copy()
        self.metadata = self.metadata[self.metadata.core_id.isin(core_ids)]
        self._indices = []
        self.core_ids = core_ids
        self.df = pd.read_csv(os.path.join(os.environ["USALIGN_METADATA_DIR"], "metadata.csv"))
        for i, core_id in enumerate(tqdm(self.core_ids, desc="Loading dataset")):
            if frames == "first":
                self._indices.append((i, 0))
            elif frames == "all":
                n = data_accessor.get_num_frames(core_id)
                self._indices.extend([(i, j) for j in range(n)])
            elif frames == "all_concat": 
                n = data_accessor.get_num_frames(core_id)
                self._indices.append((i, np.arange(n)))

        self.transform = transform
        self.include_rf = include_rf
        self.rf_as_bmode = rf_as_bmode
        self.api_compatibility = api_compatibility
        self.apply_colormap = apply_colormap
        self.flip_ud = flip_ud
        self.frames = frames
        self.output_fmt = output_fmt

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        core_idx, frame_idx = self._indices[idx]
        core_id = self.core_ids[core_idx]

        matches = self.df.loc[self.df["core_id"].astype(str).str.strip().eq(core_id)]
        if matches.empty:
            raise KeyError(f"{core_id} not found")
        row = matches.iloc[0]

        isup = row["grade_group"]
        involvement = row["pct_cancer"] / 100
        if isup == 0:
            involvement = 0
        
        if not self.rf_as_bmode:
            bmode = data_accessor.get_bmode_image(core_id, frame_idx)
        else:
            bmode = data_accessor.get_rf_image(core_id, frame_idx)
        h, w, *_ = bmode.shape
        
        if self.apply_colormap: 
            bmode = apply_colormap(bmode)

        prostate_mask = data_accessor.get_prostate_mask(core_id)
        prostate_mask = skimage.transform.resize(
            prostate_mask, (h, w), order=0, preserve_range=True, anti_aliasing=False
        ).astype(np.uint8)
        needle_mask = data_accessor.get_needle_mask(core_id)
        needle_mask = skimage.transform.resize(
            needle_mask, (h, w), order=0, preserve_range=True, anti_aliasing=False
        ).astype(np.uint8)

        if self.flip_ud: 
            bmode = np.flipud(bmode)
            prostate_mask = np.flipud(prostate_mask)
            needle_mask = np.flipud(needle_mask)

        metadata = self.metadata[self.metadata.core_id == core_id].iloc[0].to_dict()
        metadata["frame_idx"] = frame_idx

        output = {
            "bmode": bmode,
            "prostate_mask": prostate_mask,
            "needle_mask": needle_mask,
            "age": metadata['age'],
            "psa": metadata['psa'],
            "approx_psa_density": metadata['approx_psa_density'],
            "involvement": metadata['pct_cancer'],
            "grade_group": metadata['grade_group'],
            "center": metadata["center"],
            "core_id": metadata["core_id"],
            "patient_id": metadata["patient_id"],
            "loc": metadata["loc"],
            "grade": metadata["grade"],
            "family_history": metadata["family_history"],
            "pct_cancer": metadata["pct_cancer"],
            "clinically_significant": metadata["clinically_significant"],
            "all_cores_benign": metadata["all_cores_benign"]
        }
        
        if self.output_fmt == 'pil': 
            output["bmode"] = Image.fromarray(output["bmode"]).convert("RGB")
            output["prostate_mask"] = Image.fromarray(output["prostate_mask"])
            output["needle_mask"] = Image.fromarray(output["needle_mask"])

        if self.include_rf:
            rf = data_accessor.get_rf_image(core_id, frame_idx)
            output["rf"] = rf

        if self.api_compatibility == "torchvision":
            image = Image.fromarray(output["bmode"]).convert("RGB")
            label = int(output["grade"] != "Benign")
            if self.transform:
                image = self.transform(image)
            return image, label
        
        if self.transform:
            output = self.transform(output)
        return output


ISUP_TO_GROUP = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2, 5: 2}
GROUP_NAMES   = {0: "Benign (ISUP 0)", 1: "Low-grade (ISUP 2)", 2: "Significant (ISUP 3-5)"}


class BModeDatasetV3(Dataset):
    def __init__(
        self,
        core_ids=None,
        transform=None,
        frames: tp.Literal["first", "all", "all_concat"] = "first",
        include_rf=False,
        rf_as_bmode=False,
        api_compatibility=None,
        apply_colormap=True,
        flip_ud=False,
        core_selection_kw={},
        split=None,
        output_fmt='numpy',
        histo_emb_dir=None,
        mri_emb_dir=None,
    ):
        if core_ids is None:
            if split is not None:
                from medAI.datasets.nct2013.cohort_selection import select_cohort
                train, val, test = select_cohort(**core_selection_kw)
                if split == "train":
                    core_ids = train
                elif split == "val":
                    core_ids = val
                elif split == "test":
                    core_ids = test
            else:
                print("Using all cores.")
                core_ids = data_accessor.get_metadata_table().core_id.unique().tolist()

        self.metadata = data_accessor.get_metadata_table().copy()
        self.metadata = self.metadata[self.metadata.core_id.isin(core_ids)]

        self._indices = []
        self.core_ids = core_ids
        for i, core_id in enumerate(tqdm(self.core_ids, desc="Loading dataset")):
            if frames == "first":
                self._indices.append((i, 0))
            elif frames == "all":
                n = data_accessor.get_num_frames(core_id)
                self._indices.extend([(i, j) for j in range(n)])
            elif frames == "all_concat":
                n = data_accessor.get_num_frames(core_id)
                self._indices.append((i, np.arange(n)))

        self.transform       = transform
        self.include_rf      = include_rf
        self.rf_as_bmode     = rf_as_bmode
        self.api_compatibility = api_compatibility
        self.apply_colormap  = apply_colormap
        self.flip_ud         = flip_ud
        self.frames          = frames
        self.output_fmt      = output_fmt

        self.df = pd.read_csv(os.path.join(os.environ["USALIGN_METADATA_DIR"], "metadata.csv"))
        self.train_df = pd.read_csv(os.path.join(os.environ["USALIGN_METADATA_DIR"], "panda_split", "train.csv"))
        self.hist_df = pd.read_csv(os.environ["HIST_CSV"])
        self.mri_df = pd.read_csv(os.environ["MRI_CSV"])

        # Karolinska histo only
        self.hist_df = self.hist_df[self.hist_df["data_provider"] == "karolinska"]

        # Hard-negative map keyed by original ISUP grade
        self.neg_map = {
            0: [2],
            2: [3],
            3: [4, 5],
            4: [3, 5],
            5: [3, 4],
        }

        # Histo groups keyed by original ISUP grade (sampling still uses raw grades)
        self.hist_groups = {}
        for grade in self.hist_df['isup_grade'].unique():
            if grade != 1:
                self.hist_groups[grade] = (
                    self.hist_df[self.hist_df['isup_grade'] == grade][['image_id', 'cancer_percentage']]
                    .values.tolist()
                )
        for grade, group in self.hist_groups.items():
            print(f"Histopathology Grade {grade}: {len(group)} samples")
            if group:
                print(f"  Example: {group[0]}")

        # MRI groups keyed by original ISUP grade
        self.mri_groups = {}
        self.mri_df = self.mri_df[
            (self.mri_df["merged_ISUP"] <= 1) |
            ((self.mri_df["merged_ISUP"] >= 2) & (self.mri_df["has_lesion"] == 1))
        ]
        for grade in self.mri_df['merged_ISUP'].unique():
            if grade != 1:
                self.mri_groups[grade] = (
                    self.mri_df[self.mri_df['merged_ISUP'] == grade][['case_id', 'z']]
                    .values.tolist()
                )
        for grade, group in self.mri_groups.items():
            print(f"MRI Grade {grade}: {len(group)} samples")
            if group:
                print(f"  Example: {group[0]}")

        self.hist_root = histo_emb_dir
        self.mri_root  = mri_emb_dir

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        core_idx, frame_idx = self._indices[idx]
        core_id = self.core_ids[core_idx]

        matches = self.df.loc[self.df["core_id"].astype(str).str.strip().eq(core_id)]
        if matches.empty:
            raise KeyError(f"{core_id} not found")
        row = matches.iloc[0]

        # Original ISUP grade — used for histo/MRI bank lookups
        isup = row["grade_group"]
        involvement = row["pct_cancer"] / 100
        if isup == 0:
            involvement = 0

        # 3-class group label — used as the training target
        bucket_label = ISUP_TO_GROUP.get(int(isup), 0)

        if not self.rf_as_bmode:
            bmode = data_accessor.get_bmode_image(core_id, frame_idx)
        else:
            bmode = data_accessor.get_rf_image(core_id, frame_idx)
        h, w, *_ = bmode.shape

        if self.apply_colormap:
            bmode = apply_colormap(bmode)

        prostate_mask = data_accessor.get_prostate_mask(core_id)
        prostate_mask = skimage.transform.resize(
            prostate_mask, (h, w), order=0, preserve_range=True, anti_aliasing=False
        ).astype(np.uint8)
        needle_mask = data_accessor.get_needle_mask(core_id)
        needle_mask = skimage.transform.resize(
            needle_mask, (h, w), order=0, preserve_range=True, anti_aliasing=False
        ).astype(np.uint8)

        if self.flip_ud:
            bmode         = np.flipud(bmode)
            prostate_mask = np.flipud(prostate_mask)
            needle_mask   = np.flipud(needle_mask)

        metadata = self.metadata[self.metadata.core_id == core_id].iloc[0].to_dict()
        metadata["frame_idx"] = frame_idx

        output = {
            "bmode":                bmode,
            "prostate_mask":        prostate_mask,
            "needle_mask":          needle_mask,
            "age":                  metadata['age'],
            "psa":                  metadata['psa'],
            "approx_psa_density":   metadata['approx_psa_density'],
            "involvement":          metadata['pct_cancer'],
            "grade_group":          metadata['grade_group'],        # original ISUP
            "bucket_label":         bucket_label,                   # 3-class group
            "center":               metadata["center"],
            "core_id":              metadata["core_id"],
            "patient_id":           metadata["patient_id"],
            "loc":                  metadata["loc"],
            "grade":                metadata["grade"],
            "family_history":       metadata["family_history"],
            "pct_cancer":           metadata["pct_cancer"],
            "clinically_significant": metadata["clinically_significant"],
            "all_cores_benign":     metadata["all_cores_benign"],
        }

        if self.output_fmt == 'pil':
            output["bmode"]         = Image.fromarray(output["bmode"]).convert("RGB")
            output["prostate_mask"] = Image.fromarray(output["prostate_mask"])
            output["needle_mask"]   = Image.fromarray(output["needle_mask"])

        if self.include_rf:
            output["rf"] = data_accessor.get_rf_image(core_id, frame_idx)

        if self.api_compatibility == "torchvision":
            image = Image.fromarray(output["bmode"]).convert("RGB")
            label = int(output["grade"] != "Benign")
            if self.transform:
                image = self.transform(image)
            return image, label

        if self.transform:
            output = self.transform(output)

        # ------------------------------------------------------------------ #
        # Helpers — operate on original ISUP grades for bank key correctness  #
        # ------------------------------------------------------------------ #

        def get_hard_negative_label(label, valid_classes):
            """Sample a hard negative original ISUP grade."""
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

        def select_id_with_existing_file(grade, involvement, max_tries=10, modality='hist'):
            if modality == 'hist':
                group = self.hist_groups.get(grade)
                if not group:
                    raise ValueError(f"No histo samples for ISUP grade {grade}")

                if grade == 0:
                    candidates = list(group)
                else:
                    candidates = [e for e in group if abs(e[1] - involvement) <= 15]
                    if not candidates:
                        candidates = [e for e in group if abs(e[1] - involvement) <= 30]
                    if not candidates:
                        candidates = list(group)

                candidates = candidates.copy()
                random.shuffle(candidates)
                tried = 0
                while candidates and tried < max_tries:
                    sample_id = candidates.pop()[0]
                    path = os.path.join(self.hist_root, f"{sample_id}.npz")
                    if os.path.isfile(path):
                        try:
                            data = np.load(path)
                            _    = data["embedding"]
                            return sample_id, path
                        except Exception:
                            print(f"Can't load {path}")
                    tried += 1
                raise FileNotFoundError(
                    f"Could not find a valid hist file for grade={grade} after {tried} tries."
                )

            elif modality == 'mri':
                group = self.mri_groups.get(grade, [])
                if not group:
                    raise ValueError(f"No MRI samples for ISUP grade {grade}")

                candidates = list(group)
                random.shuffle(candidates)
                tried = 0
                while candidates and tried < max_tries:
                    case_id, z = candidates.pop()
                    path = os.path.join(self.mri_root, f"{case_id}_{z}.npy")
                    if os.path.isfile(path):
                        try:
                            _ = np.load(path, mmap_mode="r")
                            return (case_id, z), path
                        except Exception:
                            print(f"Can't load {path}")
                    tried += 1
                raise FileNotFoundError(
                    f"Could not find a valid MRI file for grade={grade} after {tried} tries."
                )

        # ------------------------------------------------------------------ #
        # Histo sampling — keyed by original ISUP grade                       #
        # ------------------------------------------------------------------ #
        if isup not in self.hist_groups or not self.hist_groups[isup]:
            raise ValueError(f"No histo samples for ISUP grade {isup}")

        positive_hist_id, positive_hist_path = select_id_with_existing_file(
            isup, involvement, modality='hist'
        )
        data_pos      = np.load(positive_hist_path)
        positive_hist = data_pos["embedding"]

        neg_grade = get_hard_negative_label(isup, valid_classes=[0, 2, 3, 4, 5])
        negative_hist_id, negative_hist_path = select_id_with_existing_file(
            neg_grade, involvement, modality='hist'
        )
        data_neg      = np.load(negative_hist_path)
        negative_hist = data_neg["embedding"]

        # ------------------------------------------------------------------ #
        # MRI sampling — keyed by original ISUP grade                         #
        # ------------------------------------------------------------------ #
        if isup in self.mri_groups and self.mri_groups[isup]:
            positive_mri_id, positive_mri_path = select_id_with_existing_file(
                isup, involvement, modality='mri'
            )
            positive_mri = np.load(positive_mri_path)
        else:
            print(f"Warning: No MRI samples for ISUP grade {isup}, using dummy")
            positive_mri = None

        neg_mri_grade = get_hard_negative_label(isup, valid_classes=[0, 2, 3, 4, 5])
        negative_mri_id, negative_mri_path = select_id_with_existing_file(
            neg_mri_grade, involvement, modality='mri'
        )
        negative_mri = np.load(negative_mri_path)

        # ------------------------------------------------------------------ #
        # Convert to tensors                                                   #
        # ------------------------------------------------------------------ #
        positive_hist = torch.from_numpy(positive_hist.astype(np.float32).copy())
        negative_hist = torch.from_numpy(negative_hist.astype(np.float32).copy())

        if positive_mri is not None:
            positive_mri = torch.from_numpy(positive_mri.astype(np.float32).copy())
        if negative_mri is not None:
            negative_mri = torch.from_numpy(negative_mri.astype(np.float32).copy())

        output["positive_hist"]       = positive_hist
        output["negative_hist"]       = negative_hist
        output["positive_mri"]        = positive_mri
        output["negative_mri"]        = negative_mri
        output["negative_hist_grade"] = neg_grade
        output["negative_mri_grade"]  = neg_mri_grade
        # bucket_label already set above (3-class group)

        return output


@register_dataset
def nct2013_bmode_dataset(split=None, **kwargs): 
    if split: 
        ...