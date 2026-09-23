import argparse
import json
import logging
import mmap
import os
import pickle
import random
import warnings
from dataclasses import asdict, dataclass
from email.mime import image
from typing import Literal

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors as tvt
from torchvision.transforms.v2.functional import (
    crop_bounding_boxes,
    resize_bounding_boxes,
)
from torchvision.tv_tensors import BoundingBoxFormat
from tqdm import tqdm

from medAI.global_config import RESOURCES_PATH
from medAI.datasets.bk import RawDataAccessor
from medAI.utils.data.nct_preprocessing import to_bmode
from medAI.utils.data.patch_extraction import PatchView

logging.basicConfig(level=logging.INFO)


BK_SPLITS_FILE_DEFAULT = os.path.join(RESOURCES_PATH, "bk_core_splits.json")


class BKBmodePNGDataset(Dataset):
    def __init__(
        self,
        core_ids,
        root=os.environ.get("BK_BMODE_IMAGES_DIR_PNG"),
        transform=None,
        target_transform=None,
        num_frames_per_core=1,
        return_frame_mode: Literal["single_frame", "all_frames"] = "single_frame",
        return_tensors=False,
        as_rgb=True, 
        api_compatibility: Literal['torchvision'] | None = None, 
        n_samples=None,
        label_key: Literal['positive', 'pct_cancer'] = 'positive'
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.frame_mode = return_frame_mode
        self.num_frames = num_frames_per_core
        self.return_tensors = return_tensors
        self.as_rgb = as_rgb    
        self.api_compatibility = api_compatibility
        self.label_key = label_key

        assert (
            self.root is not None
        ), "Please set the BK_BMODE_IMAGES_DIR_PNG environment variable or pass the root argument."
        assert os.path.exists(self.root), f"Directory {self.root} does not exist."

        self.metadata = pd.read_csv(os.path.join(self.root, "metadata.csv"))
        self._crop_params = json.load(
            open(os.path.join(self.root, "crop_params.json"), "r")
        )

        if n_samples: 
            self.set_core_ids([core_ids[i] for i in range(n_samples)])
        else: 
            self.set_core_ids(core_ids)

    def __len__(self):
        return len(self._indices)

    def set_core_ids(self, core_ids):
        self.core_ids = [
            core_id
            for core_id in core_ids
            if self.metadata.loc[
                self.metadata["unique_core_id"] == core_id, "processed"
            ].values[0]
        ]
        print(f"Loaded {len(self.core_ids)}/{len(core_ids)} cores")
        self._indices = []
        for core_id in self.core_ids:
            try: 
                self.get_item(core_id, 0)
            except Exception as e: 
                logging.info(f"Error loading core {core_id}: {e}. Skipping...")
                continue

            if self.frame_mode == "single_frame":
                self._indices.extend(
                    [(core_id, frame) for frame in range(self.num_frames)]
                )
            else:
                self._indices.append((core_id, 0))
        self.metadata = self.metadata.loc[self.metadata['unique_core_id'].isin(self.core_ids)]
        self.metadata.reset_index(inplace=True, drop=True)

    def get_item(self, core_id, frame):
        if self.frame_mode == "single_frame":
            bmode = Image.open(self.get_filepath("bmode", core_id, frame))
            if self.as_rgb:
                bmode = bmode.convert("RGB")
        else:
            bmode = [
                Image.open(self.get_filepath("bmode", core_id, frame))
                for frame in range(self.num_frames)
            ]
            if self.as_rgb:
                bmode = [frame.convert("RGB") for frame in bmode]
        prostate_mask = Image.open(self.get_filepath("prostate", core_id, frame))
        needle_mask = Image.open(self.get_filepath("needle", core_id, frame))
        cancer_mask = Image.open(self.get_filepath("cancer", core_id, frame))
        metadata = (
            self.metadata.loc[self.metadata["unique_core_id"] == core_id]
            .iloc[0]
            .to_dict()
        )
        crop_params = self._crop_params[core_id]

        item = {
            "bmode": bmode,
            "prostate_mask": prostate_mask,
            "needle_mask": needle_mask,
            "cancer_mask": cancer_mask,
            "metadata": metadata,
            "crop_params": crop_params,
            "frame": frame,
        }
        return item

    def __getitem__(self, idx):
        core_id, frame = self._indices[idx]

        item = self.get_item(core_id, frame)
        if self.return_tensors: 
            item['bmode'] = tvt.Image(item['bmode'])
            item['prostate_mask'] = tvt.Mask(item['prostate_mask'])
            item['needle_mask'] = tvt.Mask(item['needle_mask'])
            item['cancer_mask'] = tvt.Mask(item['cancer_mask'])

        if self.api_compatibility == 'torchvision':
            label = item['metadata'][self.label_key]
            img = item['bmode']
            item = img, label
            if self.transform:
                img = self.transform(img)
            if self.target_transform: 
                label = self.target_transform(label)
            return img, label

        if self.transform:
            item = self.transform(item)

        return item

    def get_filepath(self, data_type, core_id, frame=0):
        return os.path.join(self.root, data_type, f"{core_id}_{frame}.png")

    @classmethod
    def from_splits_file(
        cls, splits_file=BK_SPLITS_FILE_DEFAULT, fold=0, split="train", **kwargs
    ):
        with open(splits_file, "r") as f:
            splits = json.load(f)
        core_ids = splits[fold][split]
        return cls(core_ids, **kwargs)


@dataclass
class PatchOptions:
    patch_size_px: tuple = (32, 32)
    vertical_stride: int = 16
    horizontal_stride: int = 16
    patch_extraction_mode: Literal["sliding_window", "needle_com"] = "needle_com"
    prostate_mask_threshold: float = 0.8
    needle_mask_threshold: float = 0.2



#class PatchPositionsGenerator: 
#    def __init__(self, prostate_mask_loader, needle_mask_loader, patch_options: PatchOptions):
#        self.prostate_mask_loader = prostate_mask_loader
#        self.needle_mask_loader = needle_mask_loader
#        self.patch_options = patch_options 
#
#    def __call__(self, core_ids):
#        class ItemLoader: 
#            def __len__(_): 
#                return len(core_ids)
#
#            def __getitem__(_, idx):
#                return self._get_positions_single_core(core_ids[idx])
#
#        dataloader = torch.utils.data.DataLoader(ItemLoader(), batch_size=1, num_workers=8, collate_fn=lambda x: x)
#        positions = []
#        for item in tqdm(dataloader, desc="Generating patch positions"):
#            positions.append(item[0])
#        return positions
#
#    def _get_positions_single_core(self, core_id):
#        prostate_mask = self.prostate_mask_loader(core_id)
#        needle_mask = self.needle_mask_loader(core_id)
#
#        if self.patch_options.patch_extraction_mode == "sliding_window":
#            patch_view = PatchView.from_sliding_window(
#                None,
#                self.patch_options.patch_size_px,
#                (
#                    self.patch_options.vertical_stride,
#                    self.patch_options.horizontal_stride,
#                ),
#                image_shape=prostate_mask.shape,
#            )
#        elif self.patch_options.patch_extraction_mode == "needle_com":
#            patch_view = PatchView.from_horizontal_mask_center_of_mass(
#                self.patch_options.patch_size_px,
#                needle_mask,
#                self.patch_options.vertical_stride,
#                image_size=prostate_mask.shape,
#            )
#        patch_view.apply_mask(
#            prostate_mask, threshold=self.patch_options.prostate_mask_threshold
#        )
#        patch_view.apply_mask(
#            needle_mask, threshold=self.patch_options.needle_mask_threshold
#        )
#        return patch_view.positions


class PositionsGenerator: 
    def __init__(self, accessor: RawDataAccessor, patch_options: PatchOptions, cache_file=None):
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


class RawDataRFImageLoader: 
    def __init__(self, accessor, mmap=True, cache_output=False):
        self.accessor = accessor
        self.mmap = mmap
        self.cache_output = cache_output
        self._cache = {}

    def __call__(self, core_id):
        if core_id in self._cache: 
            return self._cache[core_id]

        rf = self.accessor.get_rf(core_id, mmap_mode="r" if self.mmap else None)
        if self.cache_output: 
            self._cache[core_id] = rf
        return rf


class PatchLoader: 
    def __call__(self, core_id, patch_index, positions, image): 
        return np.stack([patch for patch in PatchView(image, positions)])


class BKPatchesDataset(Dataset):
    def __init__(
        self,
        core_ids,
        positions_loader: PositionsGenerator,
        image_loader: RawDataRFImageLoader, 
        patch_loader=PatchLoader(),
        accessor=RawDataAccessor,
        transform=None,
        mode: Literal[
            "individual_patches", "all_patches_per_core"
        ] = "individual_patches",
    ):
        assert mode in ["individual_patches", "all_patches_per_core"]
        logging.info(f"Loading dataset with mode {mode}")
        self.transform = transform
        accessor = accessor()
        self.mode = mode
        self.image_loader = image_loader
        self.patch_loader = patch_loader
        self.positions_loader = positions_loader

        self._indices = []

        # setup patch views
        logging.info("Setting up patch views")
        successful_core_ids = []
        for core_id in tqdm(core_ids, desc="Loading patch positions"):
            try:
                positions = positions_loader(core_id)
                if mode == "individual_patches":
                    self._indices.extend([(core_id, i) for i in range(len(positions))])
                else:
                    self._indices.append((core_id, None))
                successful_core_ids.append(core_id)
            except Exception as e:
                logging.error(f"Error loading core {core_id}: {e}")
                continue
        logging.info(f"Loaded {len(successful_core_ids)}/{len(core_ids)} cores")
        positions_loader.save_cache()
        self.core_ids = successful_core_ids

        # self.patch_views = {}
        # for core_id in tqdm(core_ids, desc="Loading patch positions"):
        #     try:
        #         prostate_mask = accessor.load_data(
        #             core_id, key="prostate", mmap_mode="r"
        #         )
        #         needle_mask = accessor.load_data(core_id, key="needle", mmap_mode="r")
        #     except Exception as e:
        #         warnings.warn(f"Error loading core {core_id}: {e}")
        #         continue
# 
        #     if (
        #         core_id in self.positions_cache
        #         and (tuple(asdict(patch_options).values()))
        #         in self.positions_cache[core_id]
        #     ):
        #         # load from cache
        #         positions = self.positions_cache[core_id][
        #             tuple(asdict(patch_options).values())
        #         ]
        #         self.patch_views[core_id] = PatchView(
        #             None, positions, prostate_mask.shape
        #         )
        #     else:
        #         # compute positions
        #         if patch_options.patch_extraction_mode == "sliding_window":
        #             patch_view = PatchView.from_sliding_window(
        #                 None,
        #                 patch_options.patch_size_px,
        #                 (
        #                     patch_options.vertical_stride,
        #                     patch_options.horizontal_stride,
        #                 ),
        #                 image_shape=prostate_mask.shape,
        #             )
        #         elif patch_options.patch_extraction_mode == "needle_com":
        #             patch_view = PatchView.from_horizontal_mask_center_of_mass(
        #                 patch_options.patch_size_px,
        #                 needle_mask,
        #                 patch_options.vertical_stride,
        #                 image_size=prostate_mask.shape,
        #             )
        #         patch_view.apply_mask(
        #             prostate_mask, threshold=patch_options.prostate_mask_threshold
        #         )
        #         patch_view.apply_mask(
        #             needle_mask, threshold=patch_options.needle_mask_threshold
        #         )
        #         self.patch_views[core_id] = patch_view
        #         self.positions_cache.setdefault(core_id, {})[
        #             tuple(asdict(patch_options).values())
        #         ] = patch_view.positions

    def get_image(self, core_id):
        return self.image_loader(core_id)

    def get_item(self, core_id, patch_idx):
        positions = self.positions_loader(core_id)
        image = self.image_loader(core_id)
        patch = self.patch_loader(core_id, patch_idx, positions, image)
        metadata = self.accessor.get_metadata_for_id(core_id)

        # conventions for bbox are xyxy, but we use yxyx internally
        if position.ndim == 1:
            position = position[np.newaxis, :]
        patch_bbox_xyxy = np.stack(
            [position[:, 1], position[:, 0], position[:, 3], position[:, 2]], axis=1
        )

        item = {
            "patch": patch,
            "position": position,
            "patch_bbox_xyxy": patch_bbox_xyxy,
            "reference_image_shape": reference_shape_for_patch,
            "metadata": metadata,
        }
        return item

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        core_idx, patch_idx = self._indices[idx]
        item = self.get_item(core_idx, patch_idx)

        if self.transform:
            item = self.transform(item)

        return item

    def show_item(self, idx):
        core_id, patch_idx = self._indices[idx]
        patch_view = self.patch_views[core_id]
        fig, ax = plt.subplots(1, 2)
        patch_view.set_image(
            np.log(np.abs(self.accessor.get_rf(core_id, mmap_mode="r")[..., -1]) + 1)
        )
        patch_view.show(highlight_idx=patch_idx, ax=ax[0], aspect="auto")
        patch_view.set_image(None)
        protste_mask = self.accessor.load_data(core_id, key="prostate", mmap_mode="r")
        needle_mask = self.accessor.load_data(core_id, key="needle", mmap_mode="r")
        ax[0].imshow(
            protste_mask, alpha=0.5 * (protste_mask > 0), cmap="Reds", aspect="auto"
        )
        ax[0].imshow(
            needle_mask, alpha=0.5 * (needle_mask > 0), cmap="Blues", aspect="auto"
        )
        patch_view.set_image(self.get_image(core_id))
        patch = patch_view[patch_idx]
        if patch.ndim == 3:
            patch = patch[..., 0]
        ax[1].imshow(patch, cmap="magma")

    @classmethod
    def from_splits_file(
        cls, splits_file=BK_SPLITS_FILE_DEFAULT, fold=0, split="train", **kwargs
    ):
        with open(splits_file, "r") as f:
            splits = json.load(f)
        core_ids = splits[fold][split]
        return cls(core_ids, **kwargs)


class BKPatchesAndImagesDataset(Dataset):
    def __init__(
        self,
        patch_dataset: BKPatchesDataset,
        image_dataset: BKBmodePNGDataset,
        transform=None,
    ):
        self.patch_dataset = patch_dataset
        self.image_dataset = image_dataset
        self.transform = transform
        self.patch_dataset.set_core_ids(self.image_dataset.core_ids)

    def __len__(self):
        return len(self.patch_dataset)

    def __getitem__(self, idx):
        core_id, patch_idx = self.patch_dataset._indices[idx]
        image_dataset_item = self.image_dataset.get_item(core_id, 0)
        patch_dataset_item = self.patch_dataset.get_item(core_id, patch_idx)

        assert (
            image_dataset_item["metadata"]["unique_core_id"]
            == patch_dataset_item["metadata"]["unique_core_id"]
        )

        # since patch dataset has the same core_ids as the image dataset, we use the image dataset metadata
        patch_dataset_item.pop("metadata")

        # since the image dataset returns images which were cropped, we need to include the adjusted
        # patch position in the image space
        top = image_dataset_item["crop_params"]["top"]
        left = image_dataset_item["crop_params"]["left"]
        height = image_dataset_item["crop_params"]["height"]
        width = image_dataset_item["crop_params"]["width"]

        bboxes = patch_dataset_item["patch_bbox_xyxy"]
        bboxes = torch.tensor(bboxes)
        adjusted_bboxes, canvas_size = crop_bounding_boxes(
            bboxes, BoundingBoxFormat.XYXY, top, left, height, width
        )
        # also, since the image was resized, we need to further adjust the boxes
        # to match the resized image
        H, W = image_dataset_item["bmode"].size
        adjusted_bboxes, _ = resize_bounding_boxes(adjusted_bboxes, canvas_size, (H, W))
        adjusted_bboxes = adjusted_bboxes.numpy()

        item = {
            **image_dataset_item,
            **patch_dataset_item,
            "patch_bbox_xyxy": adjusted_bboxes,
        }

        if self.transform:
            item = self.transform(item)

        return item

    class _TransformV1:
        def __init__(self, image_transform, patch_transform):
            self.patch_transform = patch_transform
            self.image_transform = image_transform

        def __call__(self, item):
            img = item["bmode"]
            prostate_mask = item["prostate_mask"]
            needle_mask = item["needle_mask"]
            metadata = item["metadata"]
            patch = item["patch"]
            box = item["patch_bbox_xyxy"]
            cancer_label = item["metadata"]["positive"]
            item_id = item["metadata"]["unique_core_id"]

            img, box = self.image_transform(img, prostate_mask, needle_mask, box)

            patch = self.patch_transform(patch)
            cancer_label = torch.tensor(cancer_label).long()

            out = {
                "image": img,
                #"prostate_mask": prostate_mask,
                #"needle_mask": needle_mask,
                "patch": patch,
                "patch_position_xyxy": box,
                "label": cancer_label,
                #"metadata": metadata,
                "item_id": item_id,
            }

            return out

    @classmethod
    def get_transforms_v1(
        cls,
        image_size: int = 256,
        patch_size: int = 32,
        patch_crop_size: int = 32,
        instance_norm: bool = False,
        augmentations_mode: str = "none",
    ):
        from medAI.transforms import BModeImageAndMaskTransformV0, RFPatchTransformV0

        patch_kw = {
            "initial_resize": patch_size,
            "random_crop": False,
            "horizontal_flip": False,
            "crop_size": patch_crop_size,
            "instance_norm": instance_norm,
            "aggregation": "mean",
        }
        image_kw = {
            "image_size": image_size,
        }
        patch_transform_val = RFPatchTransformV0(**patch_kw)
        image_transform_val = BModeImageAndMaskTransformV0(**image_kw)

        if augmentations_mode == "weak":
            patch_kw["random_crop"] = True
            patch_kw["horizontal_flip"] = True
            image_kw["random_translation"] = (0.1, 0.1)
            image_kw["random_resized_crop"] = (0.8, 1.0)
        elif augmentations_mode == "medium":
            patch_kw["random_crop"] = True
            patch_kw["horizontal_flip"] = True
            image_kw["random_translation"] = (0.1, 0.1)
            image_kw["random_resized_crop"] = (0.8, 1.0)
            image_kw["pixel_level_augmentations_mode"] = "weak"
        elif augmentations_mode == "strong":
            patch_kw["random_crop"] = True
            patch_kw["horizontal_flip"] = True
            patch_kw["vertical_flip"] = True
            patch_kw["random_crop_scale"] = (0.6, 1.0)
            image_kw["random_translation"] = (0.8, 0.1)
            image_kw["random_resized_crop"] = (0.8, 1.0)
            image_kw["pixel_level_augmentations_mode"] = "strong"

        patch_transform_train = RFPatchTransformV0(**patch_kw)
        image_transform_train = BModeImageAndMaskTransformV0(**image_kw)

        return (
            cls._TransformV1(image_transform_train, patch_transform_train),
            cls._TransformV1(image_transform_val, patch_transform_val),
        )


class BKPreExtractedPatchesDataset(Dataset): 
    ...