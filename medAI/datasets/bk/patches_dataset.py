import argparse
from dataclasses import dataclass, asdict
from email.mime import image
import logging
import mmap
import os
import pickle
from typing import Literal
import warnings
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm
from medAI.datasets.bk.datasets import BK_SPLITS_FILE_DEFAULT
from medAI.utils.data.patch_extraction import PatchView
from medAI.global_config import RESOURCES_PATH
from medAI.datasets.bk import RawDataAccessor
from medAI.utils.data.nct_preprocessing import to_bmode
import random
import numpy as np
import json
from torch.utils.data import Dataset
from torchvision.transforms.v2.functional import (
    crop_bounding_boxes,
    resize_bounding_boxes,
)
from torchvision.tv_tensors import BoundingBoxFormat
from torchvision import tv_tensors as tvt
import os
import psutil


@dataclass
class PatchOptions:
    patch_size_px: tuple = (32, 32)
    vertical_stride: int = 16
    horizontal_stride: int = 16
    patch_extraction_mode: Literal["sliding_window", "needle_com"] = "needle_com"
    prostate_mask_threshold: float = 0.8
    needle_mask_threshold: float = 0.2


class PositionsGenerator:
    def __init__(
        self, accessor: RawDataAccessor, patch_options: PatchOptions, cache_file=None
    ):
        self.accessor = accessor
        self.patch_options = patch_options
        if cache_file is not None and os.path.exists(cache_file):
            self._cache = pickle.load(open(cache_file, "rb"))
        else:
            self._cache = {}
        self.cache_file = cache_file

    def __call__(self, core_id):
        if core_id in self._cache:
            return self._cache[core_id]

        prostate_mask = self.accessor.load_data(core_id, key="prostate", mmap_mode="r")
        needle_mask = self.accessor.load_data(core_id, key="needle", mmap_mode="r")

        if self.patch_options.patch_extraction_mode == "sliding_window":
            patch_view = PatchView.from_sliding_window(
                None,
                self.patch_options.patch_size_px,
                (
                    self.patch_options.vertical_stride,
                    self.patch_options.horizontal_stride,
                ),
                image_shape=prostate_mask.shape,
            )
        elif self.patch_options.patch_extraction_mode == "needle_com":
            patch_view = PatchView.from_horizontal_mask_center_of_mass(
                self.patch_options.patch_size_px,
                needle_mask,
                self.patch_options.vertical_stride,
                image_size=prostate_mask.shape,
            )
        patch_view.apply_mask(
            prostate_mask, threshold=self.patch_options.prostate_mask_threshold
        )
        patch_view.apply_mask(
            needle_mask, threshold=self.patch_options.needle_mask_threshold
        )
        self._cache[core_id] = patch_view.positions
        return patch_view.positions

    def preload(self, core_ids):
        for core_id in tqdm(core_ids):
            self(core_id)

    def save_cache(self):
        if self.cache_file is None:
            return

        with open(self.cache_file, "wb") as f:
            pickle.dump(self._cache, f)


class SimpleInMemoryPatches(Dataset):
    def __init__(
        self,
        core_ids,
        accessor=None,
        patch_options: PatchOptions = PatchOptions(),
        cache_file=None,
        mode="individual_patches",
        last_n_frames=100,
        aggregation="none",
        transform=None,
    ):
        self.accessor = accessor or RawDataAccessor()
        self.patch_options = patch_options
        self.positions_generator = PositionsGenerator(
            self.accessor, self.patch_options, cache_file
        )
        self.mode = mode
        self.transform = transform

        self._patches = {}
        self.core_ids = []
        self._indices = []

        with tqdm(core_ids, desc="Loading patches into memory") as pbar:
            for core_id in pbar:
                pid = os.getpid()
                py = psutil.Process(pid)
                memory_use = py.memory_info()[0] / 2.0**30
                pbar.set_postfix(memory_use=memory_use)

                try:
                    positions = self.positions_generator(core_id)
                    image = self.accessor.get_rf(core_id)
                    patch_view = PatchView(image, positions)
                    patches = []
                    for i in range(len(patch_view)):
                        patch = patch_view[i]
                        patch = patch[:, :, -last_n_frames:]
                        if aggregation == "mean":
                            patch = patch.mean(-1)
                        patches.append(patch)

                    patches = np.stack(patches)
                    self._patches[core_id] = patches
                    self.core_ids.append(core_id)
                    if self.mode == "individual_patches":
                        self._indices.extend(
                            [(core_id, i) for i in range(len(patch_view))]
                        )
                    else:
                        self._indices.append(core_id, None)
                except Exception as e:
                    logging.error(f"Error loading core {core_id}: {e}")
                    continue

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index):
        core_id, patch_index = self._indices[index]
        metadata = self.accessor.get_metadata_for_id(core_id)
        patch = (
            self._patches[core_id][patch_index]
            if patch_index is not None
            else self._patches[core_id]
        )
        item = {
            "patch": patch,
            "metadata": metadata,
        }
        if self.transform:
            item = self.transform(item)
        return item

    @classmethod
    def from_splits_file(
        cls, splits_file=BK_SPLITS_FILE_DEFAULT, fold=0, split="train", **kwargs
    ):
        with open(splits_file, "r") as f:
            splits = json.load(f)
        core_ids = splits[fold][split]
        return cls(core_ids, **kwargs)
