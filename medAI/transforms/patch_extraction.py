from copy import deepcopy

from medAI.utils.data.patch_extraction import PatchView

import numpy as np
from torchvision.transforms import v2 as T
from torchvision.tv_tensors import BoundingBoxes as TVBoundingBoxes
from torchvision import tv_tensors
import torch


class AddPatchesFromSlidingWindowPhysicalCoordinates:
    def __init__(
        self,
        patch_size,
        patch_stride,
        mask_keys=[],
        mask_thresholds=[],
        height_key="image_height_mm",
        width_key="image_width_mm",
        image_key="image",
        n_patches_per_sample=None,
    ):
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.mask_keys = mask_keys
        self.mask_thresholds = mask_thresholds
        self.height_key = height_key
        self.width_key = width_key
        self.image_key = image_key
        self.n_patches_per_sample = n_patches_per_sample

    def __call__(self, sample):
        # Generate patches
        pv = self._create_patch_view(sample)

        if self.n_patches_per_sample is not None:
            n_patches_total = len(pv)
            if n_patches_total >= self.n_patches_per_sample:
                indices = np.random.choice(
                    n_patches_total, self.n_patches_per_sample, replace=False
                )
            else:
                indices = np.random.choice(
                    n_patches_total, self.n_patches_per_sample, replace=True
                )
        else:
            indices = np.arange(len(pv))

        patches = np.stack([pv[i] for i in indices], axis=0)
        positions = []
        for i in indices:
            x1, y1, x2, y2 = pv.get_position_xyxy(i)
            positions.append((x1, y1, x2, y2))
        positions = np.array(positions)

        patches = [patch for patch in patches]
        patches = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float, scale=True),
            ]
        )(patches)

        positions = TVBoundingBoxes(
            positions, format="XYXY", canvas_size=sample["image"].shape[:2]
        )
        sample["patches"] = patches
        sample["patch_positions_xyxy"] = positions

        return sample

    def _create_patch_view(self, sample):
        image = sample[self.image_key]
        height = sample[self.height_key]
        width = sample[self.width_key]
        masks = [sample[mask_key] for mask_key in self.mask_keys]

        pv = PatchView.from_sliding_window_physical_coordinate(
            image,
            image_physical_size=(height, width),
            window_physical_size=self.patch_size,
            stride_physical_size=self.patch_stride,
            masks=masks,
            thresholds=self.mask_thresholds,
        )

        return pv


class RandomSamplePatches:
    def __init__(self, num_patches=4, patch_size=(224, 224), image_key="image"):
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.image_key = image_key

    def __call__(self, sample):
        image = sample[self.image_key]
        boxes = []
        patches = []
        for _ in range(self.num_patches):
            top, left, height, width = T.RandomCrop.get_params(image, self.patch_size)
            boxes.append((left, top, left + width, top + height))
            patches.append(T.functional.crop(image, top, left, height, width))

        boxes = tv_tensors.BoundingBoxes(
            boxes, format="XYXY", canvas_size=T.functional.get_size(image)
        )
        sample["patches"] = patches
        sample["boxes"] = boxes
        return sample