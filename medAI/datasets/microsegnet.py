import argparse
import json
import os
from pathlib import Path
import SimpleITK as sitk
from typing import Callable, Literal
import pandas as pd
from torch import Tensor
import torch
from torch.utils.data import Dataset
from PIL import Image
import re
import numpy as np
from torchvision import transforms as T
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.nn import functional as F
import logging


class MicroSegNetDataset(Dataset):
    def __init__(
        self,
        root: Path | str,
        transform: Callable | None = None,
        split: Literal["train", "test"] = "train",
        preprocess: bool | None = None,
        raw_data_dir: Path | str | None = None,
    ):
        """
        Args:
            transform: Takes tuple of image, mask (PIL.Image) and returns transformed image, mask
        """

        self.root = root if isinstance(root, Path) else Path(root)
        self.transform = transform
        self.split = split
        if self.split == 'val': 
            self.split = 'test' # test is val

        if preprocess:
            # should try to preprocess the data
            assert raw_data_dir is not None, "Raw data directory must be provided"
            self.root.mkdir(exist_ok=True, parents=True)
            raw_data_dir = (
                raw_data_dir if isinstance(raw_data_dir, Path) else Path(raw_data_dir)
            )
            self._preprocess_data(self.root, raw_data_dir)

        self._image_folder = self.root / self.split / "micro_ultrasound_scans"
        self._mask_folder = self.root / self.split / "expert_annotations"

        self._image_paths = sorted(
            self._image_folder.iterdir(), key=self._extract_indices
        )
        self._mask_paths = sorted(
            self._mask_folder.iterdir(), key=self._extract_indices
        )

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, index):
        image_path = self._image_paths[index]
        mask_path = self._mask_paths[index]
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        data = dict(
            image=image, mask=mask
        )

        if self.transform is not None:
            data = self.transform(data)
        return data

    def set_ids(self, ids): 
        self._image_paths = self._filter_by_ids(self._image_paths, ids)
        self._mask_paths = self._filter_by_ids(self._mask_paths, ids)

    def get_ids(self): 
        return set([self._extract_indices(path)[0] for path in self._image_paths])

    def _filter_by_ids(self, paths, ids: list[int]):
        out = [] 
        for path in paths: 
            if self._extract_indices(path)[0] in ids: 
                out.append(path)
        return out

    def _extract_indices(self, path: Path):
        return list(map(int, re.search(r"(\d+)_(\d+)", path.name).groups()))

    def _preprocess_data(self, target_dir: Path, source_dir: Path):
        target_dir.joinpath("train").mkdir(exist_ok=True)
        target_dir.joinpath("test").mkdir(exist_ok=True)
        for split in ["train", "test"]:
            scans_dir = source_dir.joinpath(split).joinpath("micro_ultrasound_scans")
            annotations_dir = source_dir.joinpath(split).joinpath("expert_annotations")
            target_scans_dir = target_dir.joinpath(split).joinpath(
                "micro_ultrasound_scans"
            )
            target_annotations_dir = target_dir.joinpath(split).joinpath(
                "expert_annotations"
            )
            target_scans_dir.mkdir(exist_ok=True)
            target_annotations_dir.mkdir(exist_ok=True)

            def _read_id_from_path(path):
                id = int(re.search("(\d+)", str(path.name)).groups()[0])
                return id

            scans_paths = sorted(scans_dir.iterdir(), key=_read_id_from_path)
            annotations_paths = sorted(
                annotations_dir.iterdir(), key=_read_id_from_path
            )

            for id, (scan, target) in enumerate(
                tqdm(
                    zip(scans_paths, annotations_paths),
                    desc=f"Preprocessing {split} data",
                    total=len(scans_paths),
                )
            ):
                scan = sitk.GetArrayFromImage(sitk.ReadImage(scan)).astype("uint8")
                annotation = sitk.GetArrayFromImage(sitk.ReadImage(target)).astype(
                    "uint8"
                )

                for frame_idx, (frame, target_frame) in enumerate(
                    zip(scan, annotation)
                ):
                    scan_output_path = target_scans_dir / f"{id}_{frame_idx}.png"
                    annotations_path = target_annotations_dir / f"{id}_{frame_idx}.png"
                    frame = Image.fromarray(frame)
                    target_frame = Image.fromarray(target_frame)
                    frame.save(scan_output_path)
                    target_frame.save(annotations_path)