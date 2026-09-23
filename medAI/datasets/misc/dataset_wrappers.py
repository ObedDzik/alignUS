# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import math
from typing import Sequence
import numpy as np

import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T
from einops import rearrange


class ReturnsIndexDatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return tuple([*self.dataset[index], index])


class MaskDatasetWrapper(Dataset):
    """
    Wrapper for the dataset to apply a mask to the images.

    Args:
        dataset: The dataset to wrap.
        patch_size: The patch size.
        pred_ratio: The ratio of the masked area. For example, setting pred_ratio to 0.3 will mask 30% of the image.
            Setting to 0 will disable masking.
        pred_ratio_var: The variance of the masked area. 
        pred_aspect_ratio: The aspect ratio of the masked area. 
        pred_shape: The shape of the masked area. Can be "block" or "rand".
        pred_start_epoch: The epoch to start the masked area. Before this epoch, the masked area will be 0.
        meanfill_masked_pixel: Whether to meanfill the masked pixel.
    """

    def __init__(
        self,
        dataset,
        patch_size,
        pred_ratio=0.3,
        pred_ratio_var=0.1,
        pred_aspect_ratio=(0.3, 1 / 0.3),
        pred_shape="block",
        pred_start_epoch=0,
        image_keys=None,
        meanfill_masked_pixel=False
    ):
        self.dataset = dataset
        self.image_keys = image_keys

        if image_keys is None:
            try: 
                images, label = self.dataset[0]
            except: 
                raise ValueError(f"Dataset should return image, label tuples. Got {type(self.dataset[0])}")
            if not isinstance(images, Sequence):
                raise ValueError(f"Dataset should return lists of images (one per crop)")
            if not isinstance(images[0], torch.Tensor):
                raise ValueError(f"Dataset should return images in Tensor format.")
        else: 
            images = [self.dataset[0][k] for k in image_keys]
            if not isinstance(images[0], torch.Tensor):
                raise ValueError(f"Dataset should return images in Tensor format.")

        self.psz = patch_size
        self.pred_ratio = (
            pred_ratio[0]
            if isinstance(pred_ratio, list) and len(pred_ratio) == 1
            else pred_ratio
        )
        self.pred_ratio_var = (
            pred_ratio_var[0]
            if isinstance(pred_ratio_var, list) and len(pred_ratio_var) == 1
            else pred_ratio_var
        )
        if isinstance(self.pred_ratio, list) and not isinstance(
            self.pred_ratio_var, list
        ):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch
        self.meanfill_masked_pixel = meanfill_masked_pixel

    def get_pred_ratio(self):
        if hasattr(self, "epoch") and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = (
                random.uniform(
                    self.pred_ratio - self.pred_ratio_var,
                    self.pred_ratio + self.pred_ratio_var,
                )
                if self.pred_ratio_var > 0
                else self.pred_ratio
            )

        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        output = self.dataset[index]
        masks = []

        if self.image_keys is None:
            images = output[0]
        else:
            images = [output[k] for k in self.image_keys]

        for img in images:
            try:
                H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
            except:
                # skip non-image
                continue

            high = self.get_pred_ratio() * H * W

            if self.pred_shape == "block":
                # following BEiT (https://arxiv.org/abs/2106.08254), see at
                # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[top : top + h, left : left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if mask[i, j] == 0:
                                            mask[i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta

            elif self.pred_shape == "rand":
                mask = np.hstack(
                    [
                        np.zeros(H * W - int(high)),
                        np.ones(int(high)),
                    ]
                ).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)

            else:
                # no implementation
                assert False

            masks.append(mask)

        if self.meanfill_masked_pixel:
            for img, mask in zip(output[0], masks):
                img_mean = img.mean((-1, -2))[:, None, None].repeat(1, self.psz, self.psz)
                img_view_as_patches = rearrange(
                    img, 'c (h psz1) (w psz2) -> h w c psz1 psz2', psz1=self.psz, psz2=self.psz
                )
                img_view_as_patches[mask] = img_mean

        if self.image_keys is None:
            return output + (masks,)
        else:
            output['masks'] = masks
            return output

    def __len__(self): 
        return len(self.dataset)