import json
import logging
import pandas as pd
import os
import numpy as np
from medAI.datasets.nct2013.cohort_selection import (
    select_cohort,
)
from medAI.utils.data.patch_extraction import PatchView
from dataclasses import dataclass
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import torch
from torchvision.transforms import v2 as T
from torchvision import tv_tensors
from typing import Literal
import torch
from torchvision.transforms import v2 as T
from torchvision import tv_tensors
import matplotlib.pyplot as plt
from medAI.transforms.transforms import get_pixel_level_augmentations_by_mode
from medAI.utils.data.patch_extraction import PatchView


@dataclass
class NCT2013PatchesDatasetConf:
    patch_height_mm: float = 5
    patch_width_mm: float = 5
    stride_height_mm: float = 1
    stride_width_mm: float = 1
    patch_source: str = "rf"
    image_source: str = "bmode"
    prostate_mask_threshold: float = 0.9
    needle_mask_threshold: float = 0.66


class NCT2013PatchesDataset:
    IMAGE_HEIGHT_MM = 28
    IMAGE_WIDTH_MM = 46.06

    DATA_KEY_PATCH = "patch"
    DATA_KEY_PROSTATE_MASK = "prostate_mask"
    DATA_KEY_NEEDLE_MASK = "needle_mask"
    DATA_KEY_IMAGE = "image"
    DATA_KEY_METADATA = "metadata"
    DATA_KEY_CANCER_LABEL = "cancer_label"
    DATA_KEY_PATCH_POSITION_HWHW_RELATIVE = "patch_position"
    DATA_KEY_PATCH_POSITION_XYXY = "patch_position_xyxy"

    def __init__(
        self,
        root,
        core_ids,
        conf: NCT2013PatchesDatasetConf,
        transform=None,
        debug=False,
        patch_mode="individual_patches",
    ):
        self.root = root
        self.conf = conf
        self.core_ids = core_ids
        assert patch_mode in ['individual_patches', 'patch_views'], "Invalid mode"
        self.mode = patch_mode

        if os.environ.get("DEBUG", False) or debug:
            logging.info(f"DEBUG MODE: Selecting 10 random core ids...")
            self.core_ids = np.random.choice(self.core_ids, 10, replace=False)

        self.transform = transform

        self.metadata = pd.read_csv(os.path.join(root, "metadata.csv"))
        self._bmode_data = np.load(os.path.join(root, "bmode.npy"), mmap_mode="r")
        self._rf_data = np.load(os.path.join(root, "rf.npy"), mmap_mode="r")
        self._prostate_mask_data = np.load(
            os.path.join(root, "prostate_mask.npy"), mmap_mode="r"
        )
        self._needle_mask_data = np.load(
            os.path.join(root, "needle_mask.npy"), mmap_mode="r"
        )
        self._core_id_to_idx = json.load(
            open(os.path.join(root, "core_id_to_idx.json"))
        )

        self.patch_views = {}
        self.setup_patch_views(self.core_ids)

        self._indices = []
        for i, core_id in enumerate(self.core_ids):
            view = self.patch_views[core_id]
            if self.mode == 'individual_patches':
                for j in range(len(view)):
                    self._indices.append((i, j))
            elif self.mode == 'patch_views':
                self._indices.append((i, 0))

    def bmode(self, core_id):
        idx = self._core_id_to_idx[core_id]
        return self._bmode_data[idx]

    def rf(self, core_id):
        idx = self._core_id_to_idx[core_id]
        return self._rf_data[idx]

    def prostate_mask(self, core_id):
        idx = self._core_id_to_idx[core_id]
        return self._prostate_mask_data[idx]

    def needle_mask(self, core_id):
        idx = self._core_id_to_idx[core_id]
        return self._needle_mask_data[idx]

    def metadata_for_core_id(self, core_id):
        return self.metadata.loc[self.metadata.core_id == core_id].iloc[0].to_dict()

    def setup_patch_views(self, core_ids):
        print(f"Setting up patch views for {len(core_ids)} cores from scratch.")
        patch_height_px, patch_width_px, stride_height_px, stride_width_px = (
            self.compute_patch_sizes()
        )

        if self.conf.patch_source == "rf":
            patch_images = [self.rf(core_id) for core_id in core_ids]
        else:
            patch_images = [self.bmode(core_id) for core_id in core_ids]

        needle_masks = [self.needle_mask(core_id) for core_id in core_ids]
        prostate_masks = [self.prostate_mask(core_id) for core_id in core_ids]

        patch_views = PatchView.build_collection_from_images_and_masks(
            patch_images,
            (patch_height_px, patch_width_px),
            (stride_height_px, stride_width_px),
            "topright",
            mask_lists=[needle_masks, prostate_masks],
            thresholds=[
                self.conf.needle_mask_threshold,
                self.conf.prostate_mask_threshold,
            ],
        )

        for core_id, view in zip(core_ids, patch_views):
            self.patch_views[core_id] = view

    def compute_patch_sizes(self):
        # Computing patch sizes in pixels
        core_id = self.core_ids[0]

        if self.conf.patch_source == "rf":
            patch_ref = self.rf(core_id)
        else:
            patch_ref = self.bmode(core_id)

        height_px = patch_ref.shape[0]
        width_px = patch_ref.shape[1]

        height_mm = self.IMAGE_HEIGHT_MM
        width_mm = self.IMAGE_WIDTH_MM

        patch_height_px = int(self.conf.patch_height_mm / height_mm * height_px)
        patch_width_px = int(self.conf.patch_width_mm / width_mm * width_px)
        stride_height_px = int(self.conf.stride_height_mm / height_mm * height_px)
        stride_width_px = int(self.conf.stride_width_mm / width_mm * width_px)

        print(f"stride (pixels): {stride_height_px}, {stride_width_px}")
        print(f"height (pixels): {patch_height_px}, width: {patch_width_px}")

        return patch_height_px, patch_width_px, stride_height_px, stride_width_px

    def __len__(self):
        return len(self._indices)

    def get_single_patch(self, core_id, patch_idx):
        patch_view = self.patch_views[core_id]
        patch = patch_view[patch_idx]
        position = patch_view.positions[patch_idx]
        prostate_mask = self.prostate_mask(core_id)
        needle_mask = self.needle_mask(core_id)
        metadata_row = self.metadata_for_core_id(core_id)
        if self.conf.image_source == "rf":
            image = self.rf(core_id)
        else:
            image = self.bmode(core_id)

        # make position be relative to the image size rather than absolute
        hmin, wmin, hmax, wmax = position
        h, w = patch_view.image.shape
        position = (hmin / h, wmin / w, hmax / h, wmax / w)
        position = np.array(position)

        # also add position in xyxy format relative to main image (bounding box style)
        hmin, wmin, hmax, wmax = position
        h, w = image.shape
        position_xyxy = (int(wmin * w), int(hmin * h), int(wmax * w), int(hmax * h))

        item = {
            self.DATA_KEY_PATCH: patch,
            self.DATA_KEY_PROSTATE_MASK: prostate_mask,
            self.DATA_KEY_NEEDLE_MASK: needle_mask,
            self.DATA_KEY_IMAGE: image,
            self.DATA_KEY_METADATA: metadata_row,
            self.DATA_KEY_CANCER_LABEL: metadata_row["grade"] != "Benign",
            self.DATA_KEY_PATCH_POSITION_HWHW_RELATIVE: position,
            self.DATA_KEY_PATCH_POSITION_XYXY: position_xyxy,
        }

        if self.transform:
            item = self.transform(item)

        return item

    def __getitem__(self, index):
        i, j = self._indices[index]
        core_id = self.core_ids[i]

        if self.mode == 'individual_patches':
            return self.get_single_patch(core_id, j)
        elif self.mode == 'patch_views':
            patch_view = self.patch_views[core_id]
            items = []
            for j in range(len(patch_view)):
                items.append(self.get_single_patch(core_id, j))

            def stack(items): 
                if isinstance(items[0], torch.Tensor): 
                    return torch.stack(items)
                elif isinstance(items[0], np.ndarray):
                    return np.stack(items)
                else: 
                    return items

            item = {}
            item['image'] = items[0]['image']
            item['patch'] = stack([i['patch'] for i in items])
            item['prostate_mask'] = items[0]['prostate_mask']
            item['needle_mask'] = items[0]['needle_mask']
            item['metadata'] = items[0]['metadata']
            item['cancer_label'] = items[0]['cancer_label']
            item['patch_position_xyxy'] = stack([i['patch_position_xyxy'] for i in items])
            return item
        else: 
            raise ValueError("Invalid mode")

    def show_item(self, idx):
        _tmp_transform = self.transform
        self.transform = None
        item = self[idx]
        self.transform = _tmp_transform

        patch = item[self.DATA_KEY_PATCH]
        prostate_mask = item[self.DATA_KEY_PROSTATE_MASK]
        needle_mask = item[self.DATA_KEY_NEEDLE_MASK]
        image = item[self.DATA_KEY_IMAGE]
        metadata = item[self.DATA_KEY_METADATA]
        cancer_label = item[self.DATA_KEY_CANCER_LABEL]
        position = item[self.DATA_KEY_PATCH_POSITION_HWHW_RELATIVE]

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        IMG_EXTENT = (0, self.IMAGE_WIDTH_MM, self.IMAGE_HEIGHT_MM, 0)
        axs[0, 0].imshow(patch, cmap="gray", aspect="auto")
        axs[0, 0].set_title("Patch")
        axs[1, 1].imshow(prostate_mask, cmap="Purples", extent=IMG_EXTENT)
        axs[1, 1].imshow(needle_mask, cmap="Greens", extent=IMG_EXTENT, alpha=0.5)
        axs[1, 1].set_title("Masks")
        view = self.patch_views[metadata["core_id"]]
        view.show(axs[1, 0])
        axs[1, 0].set_title("Patch view")
        axs[0, 1].imshow(image, cmap="gray", extent=IMG_EXTENT)
        axs[0, 1].set_title("Image")

        # make box where patch came from
        hmin, wmin, hmax, wmax = position
        h, w = self.IMAGE_HEIGHT_MM, self.IMAGE_WIDTH_MM
        hmin, wmin, hmax, wmax = hmin * h, wmin * w, hmax * h, wmax * w
        rect = plt.Rectangle(
            (wmin, hmin), wmax - wmin, hmax - hmin, edgecolor="r", facecolor="none"
        )
        axs[1, 1].add_patch(rect)
        rect = plt.Rectangle(
            (wmin, hmin), wmax - wmin, hmax - hmin, edgecolor="r", facecolor="none"
        )
        axs[0, 1].add_patch(rect)

        plt.show()

    @classmethod
    def from_fold(
        cls,
        root,
        fold,
        split="train",
        conf: NCT2013PatchesDatasetConf = NCT2013PatchesDatasetConf(),
        transform=None,
        cohort_selection_kw={},
        **kwargs,
    ):
        train, val, test = select_cohort(
            CohortSelectionOptions(fold=fold, n_folds=5, **cohort_selection_kw)
        )
        core_ids = {"train": train, "val": val, "test": test}[split]
        return cls(root, core_ids, conf, transform, **kwargs)

    @classmethod
    def from_test_center(
        cls,
        root,
        test_center,
        split="train",
        conf: NCT2013PatchesDatasetConf = NCT2013PatchesDatasetConf(),
        transform=None,
        cohort_selection_kw={},
        **kwargs,
    ):
        train, val, test = select_cohort(
            CohortSelectionOptions(test_center=test_center, **cohort_selection_kw)
        )
        core_ids = {"train": train, "val": val, "test": test}[split]
        return cls(root, core_ids, conf, transform, **kwargs)

    @classmethod
    def from_cohort_selection_options(
        cls,
        root,
        cohort_selection_options,
        split="train",
        conf: NCT2013PatchesDatasetConf = NCT2013PatchesDatasetConf(),
        transform=None,
        **kwargs,
    ):
        train, val, test = select_cohort(cohort_selection_options)
        core_ids = {"train": train, "val": val, "test": test}[split]
        return cls(root, core_ids, conf, transform, **kwargs)

    @classmethod
    def from_cohort_selection(
        cls,
        root, 
        cohort_selection_mode="test_center_validation_folds", 
        test_center='UVA', 
        fold=0, 
        split="train",
        transform=None,
        cohort_selection_kw={},
        **kwargs,
    ):
        from .cohort_selection import select_cohort_by_mode
        splits = select_cohort_by_mode(cohort_selection_mode, test_center=test_center, fold=fold, **cohort_selection_kw)
        core_ids = splits[split]
        return cls(root, core_ids, transform=transform, **kwargs)


DATA_KEY_PATCH = "patch"
DATA_KEY_PROSTATE_MASK = "prostate_mask"
DATA_KEY_NEEDLE_MASK = "needle_mask"
DATA_KEY_IMAGE = "image"
DATA_KEY_METADATA = "metadata"
DATA_KEY_CANCER_LABEL = "cancer_label"
DATA_KEY_PATCH_POSITION_HWHW_RELATIVE = "patch_position"
DATA_KEY_PATCH_POSITION_XYXY = "patch_position_xyxy"


class NCT2013PatchTransform:
    def __init__(
        self,
        size=256,
        instance_norm=True,
        crop_size=None,
        random_crop=False,
        horizontal_flip=False,
        vertical_flip=False,
        random_crop_scale=None,
        random_erasing=False,
    ):
        self.size = size
        self.instance_norm = instance_norm
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.horizontal_flip = horizontal_flip
        self.random_crop_scale = random_crop_scale
        self.vertical_flip = vertical_flip
        self.random_erasing = random_erasing

    def __call__(self, patch):
        patch = torch.tensor(patch.copy()).float().unsqueeze(0)
        if self.instance_norm:
            patch = (patch - patch.mean()) / patch.std()
        patch = T.Resize((self.size, self.size))(patch)

        if self.crop_size is not None:
            if self.crop_size > self.size:
                raise ValueError(f"crop_size must be less than or equal to {self.size}")
            if self.random_crop:
                if self.random_crop_scale is not None:
                    patch = T.RandomResizedCrop(
                        (self.crop_size, self.crop_size), scale=self.random_crop_scale
                    )(patch)
                else:
                    patch = T.RandomCrop((self.crop_size, self.crop_size))(patch)
            else:
                patch = T.CenterCrop((self.crop_size, self.crop_size))(patch)

        if self.horizontal_flip:
            patch = T.RandomHorizontalFlip()(patch)
        if self.vertical_flip:
            patch = T.RandomVerticalFlip()(patch)
        if self.random_erasing:
            patch = T.RandomErasing()(patch)

        return patch


class NCT2013ImageAndMaskTransform:
    def __init__(
        self,
        image_size=256,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        random_translation: tuple[float, float] | None = None,
        random_resized_crop: tuple[int, int] | None = None,
        pixel_level_augmentations_mode: Literal["weak", "strong", "none"] = "none",
        return_masks: bool = False,
    ):
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.random_translation = random_translation
        self.random_resized_crop = random_resized_crop
        self.pixel_level_augmentations_mode = pixel_level_augmentations_mode
        self.pixel_level_augmentations = get_pixel_level_augmentations_by_mode(
            pixel_level_augmentations_mode
        )
        self.return_masks = return_masks

    def __call__(self, image, prostate_mask, needle_mask, box):
        image = torch.tensor(image).float().unsqueeze(0).repeat_interleave(3, dim=0)
        image = (image - image.min()) / (image.max() - image.min())

        if self.pixel_level_augmentations_mode != "none":
            image = T.ToDtype(torch.uint8, scale=True)(image)
            image = T.ToPILImage()(image)
            image = self.pixel_level_augmentations(image)
            image = T.ToImage()(image)
            image = T.ToDtype(torch.float32, scale=True)(image)

        image = T.Normalize(mean=self.mean, std=self.std)(image)

        image = tv_tensors.Image(image)
        prostate_mask = torch.tensor(prostate_mask).float().unsqueeze(0)
        prostate_mask = tv_tensors.Mask(prostate_mask)
        box = torch.tensor(box).unsqueeze(0).long()
        box = tv_tensors.BoundingBoxes(box, format="xyxy", canvas_size=image.shape[-2:])

        needle_mask = torch.tensor(needle_mask).float().unsqueeze(0)
        needle_mask = tv_tensors.Mask(needle_mask)

        H, W = image.shape[-2:]

        if self.return_masks:
            # resize masks to image shape
            prostate_mask = T.Resize((H, W))(prostate_mask)
            needle_mask = T.Resize((H, W))(needle_mask)

            # geometric transformations
            image, prostate_mask, needle_mask, box = T.Resize(
                (self.image_size, self.image_size)
            )(image, prostate_mask, needle_mask, box)

            if self.random_resized_crop:
                image, prostate_mask, needle_mask, box = T.RandomResizedCrop(
                    self.image_size, scale=self.random_resized_crop
                )(image, prostate_mask, needle_mask, box)

            if self.random_translation:
                image, prostate_mask, needle_mask, box = T.RandomAffine(
                    degrees=0, translate=self.random_translation
                )(image, prostate_mask, needle_mask, box)

            return image, prostate_mask, needle_mask, box
        else:
            image, box = T.Resize((self.image_size, self.image_size))(image, box)
            if self.random_resized_crop:
                image, box = T.RandomResizedCrop(
                    self.image_size, scale=self.random_resized_crop
                )(image, box)

            if self.random_translation:
                image, box = T.RandomAffine(
                    degrees=0, translate=self.random_translation
                )(image, box)

            return image, box


class NCT2013FullTransform:
    def __init__(self, patch_transform, image_transform):
        self.patch_transform = patch_transform
        self.image_transform = image_transform

    def __call__(self, item):
        img = item[DATA_KEY_IMAGE]
        prostate_mask = item[DATA_KEY_PROSTATE_MASK]
        needle_mask = item[DATA_KEY_NEEDLE_MASK]

        metadata = item[DATA_KEY_METADATA]
        patch = item[DATA_KEY_PATCH]
        box = item[DATA_KEY_PATCH_POSITION_XYXY]
        cancer_label = item[DATA_KEY_CANCER_LABEL]

        img, box = self.image_transform(img, prostate_mask, needle_mask, box)

        patch = self.patch_transform(patch)
        cancer_label = torch.tensor(cancer_label).long()

        out = {
            DATA_KEY_IMAGE: img,
            DATA_KEY_PROSTATE_MASK: prostate_mask,
            DATA_KEY_NEEDLE_MASK: needle_mask,
            DATA_KEY_PATCH: patch,
            DATA_KEY_PATCH_POSITION_XYXY: box,
            DATA_KEY_CANCER_LABEL: cancer_label,
            DATA_KEY_METADATA: metadata,
        }

        return out


def get_image_and_patch_transforms(
    image_size: int = 256,
    patch_size: int = 256,
    patch_crop_size: int = 224,
    instance_norm: bool = True,
    augmentations_mode: str = "none",
):
    patch_kw = {
        "size": patch_size,
        "random_crop": False,
        "horizontal_flip": False,
        "crop_size": patch_crop_size,
        "instance_norm": instance_norm,
    }
    image_kw = {
        "image_size": image_size,
    }
    patch_transform_val = NCT2013PatchTransform(**patch_kw)
    image_transform_val = NCT2013ImageAndMaskTransform(**image_kw)

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

    patch_transform_train = NCT2013PatchTransform(**patch_kw)
    image_transform_train = NCT2013ImageAndMaskTransform(**image_kw)

    return NCT2013FullTransform(
        patch_transform_train, image_transform_train
    ), NCT2013FullTransform(patch_transform_val, image_transform_val)
