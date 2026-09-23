from collections import defaultdict
import PIL
import numpy as np
import torch
from torchvision.transforms import v2 as T
from torchvision.tv_tensors import (
    Image as TVImage,
    Mask as TVMask,
)


class ImageTransform:
    def __init__(
        self,
        image_key="image",
        mask_keys=[],
        box_keys=["patch_positions_xyxy"],
        output_size_px=None,
        mean=None,
        std=None,
        geometric_aug=None,
        pixel_aug=None,
    ):
        self.image_key = image_key
        self.mask_keys = mask_keys
        self.box_keys = box_keys
        self.output_size_px = output_size_px
        self.mean = mean
        self.std = std
        self.geometric_aug = geometric_aug
        self.pixel_aug = pixel_aug

    def __call__(self, sample):
        image = sample[self.image_key]
        image = T.ToImage()(image)  # to [0, 1]
        image = T.ToDtype(torch.float, scale=True)(image)  # to float tensor
        image = TVImage(image)  # to TVImage (C, H, W)

        masks = {}
        for mask_key in self.mask_keys:
            masks[mask_key] = sample[mask_key]
            masks[mask_key] = T.ToImage()(masks[mask_key])  # to [0, 1]
            masks[mask_key] = T.ToDtype(torch.float, scale=False)(
                masks[mask_key]
            )  # to float tensor
            masks[mask_key] = TVMask(masks[mask_key])

        boxes = {}
        for box_key in self.box_keys:
            boxes[box_key] = sample[box_key]

        image, masks, boxes = self.inner_transform(image, masks, boxes)

        sample[self.image_key] = image  # (C, H, W)
        for mask_key in self.mask_keys:
            sample[mask_key] = masks[mask_key]
        for box_key in self.box_keys:
            sample[box_key] = boxes[box_key]

        return sample

    def inner_transform(self, image, masks, boxes):
        if self.pixel_aug is not None:
            image = self.pixel_aug(image)

        if self.output_size_px is not None:
            if self.geometric_aug is not None:
                image, masks, boxes = self.geometric_aug(image, masks, boxes)
            else:
                image, masks, boxes = T.Resize(self.output_size_px)(image, masks, boxes)

        if self.mean is not None and self.std is not None:
            image = T.Normalize(self.mean, self.std)(image)

        return image, masks, boxes


class ImageTransformMultiCrop(ImageTransform):
    def __init__(
        self,
        num_crops=2,
        num_local_crops=0,
        geometric_aug_local=None,
        output_size_px_local=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_crops = num_crops
        self.num_local_crops = num_local_crops
        self.geometric_aug_local = geometric_aug_local
        self.output_size_px_local = output_size_px_local

    def __call__(self, sample):

        image = sample[self.image_key]
        image = T.ToImage()(image)  # to [0, 1]
        image = T.ToDtype(torch.float, scale=True)(image)  # to float tensor
        image = TVImage(image)  # to TVImage (C, H, W)

        masks = {}
        for mask_key in self.mask_keys:
            masks[mask_key] = sample[mask_key]
            masks[mask_key] = T.ToImage()(masks[mask_key])  # to [0, 1]
            masks[mask_key] = T.ToDtype(torch.float, scale=False)(
                masks[mask_key]
            )  # to float tensor
            masks[mask_key] = TVMask(masks[mask_key])

        boxes = {}
        for box_key in self.box_keys:
            boxes[box_key] = sample[box_key]

        boxes_multi = defaultdict(list)
        masks_multi = defaultdict(list)
        image_multi = []

        for _ in range(self.num_crops):
            img_copy = image.clone()
            masks_copy = {k: v.clone() for k, v in masks.items()}
            boxes_copy = {k: v.clone() for k, v in boxes.items()}

            img_crop, masks_crop, boxes_crop = self.inner_transform(
                img_copy, masks_copy, boxes_copy
            )

            image_multi.append(img_crop)
            for mask_key in self.mask_keys:
                masks_multi[mask_key].append(masks_crop[mask_key])
            for box_key in self.box_keys:
                boxes_multi[box_key].append(boxes_crop[box_key])

        if self.num_local_crops > 0 and self.geometric_aug_local is not None:
            for _ in range(self.num_local_crops):
                img_copy = image.clone()
                masks_copy = {k: v.clone() for k, v in masks.items()}
                boxes_copy = {k: v.clone() for k, v in boxes.items()}

                img_crop, masks_crop, boxes_crop = self.geometric_aug_local(
                    img_copy, masks_copy, boxes_copy
                )

                if self.output_size_px_local is not None:
                    img_crop, masks_crop, boxes_crop = T.Resize(
                        self.output_size_px_local
                    )(img_crop, masks_crop, boxes_crop)

                if self.mean is not None and self.std is not None:
                    img_crop = T.Normalize(self.mean, self.std)(img_crop)

                image_multi.append(img_crop)
                for mask_key in self.mask_keys:
                    masks_multi[mask_key].append(masks_crop[mask_key])
                for box_key in self.box_keys:
                    boxes_multi[box_key].append(boxes_crop[box_key])

        sample[self.image_key] = image_multi
        for mask_key in self.mask_keys:
            sample[mask_key] = masks_multi[mask_key]
        for box_key in self.box_keys:
            sample[box_key] = boxes_multi[box_key]

        return sample


class PatchTransform:
    def __init__(
        self,
        output_size_px=(224, 224),
        instance_norm=False,
        mean=None,
        std=None,
        geometric_aug=None,
        pixel_aug=None,
    ):
        self.output_size_px = output_size_px
        self.instance_norm = instance_norm
        self.mean = mean
        self.std = std
        self.geometric_aug = geometric_aug
        self.pixel_aug = pixel_aug

    def __call__(self, sample):
        patches = sample["patches"]
        patches = self.transform(patches)
        sample["patches"] = torch.stack(patches, dim=0)  # (N, C, H, W)

        return sample

    def transform(self, patches):
        # apply patch augmentations
        if self.pixel_aug is not None:
            patches = self.pixel_aug(patches)
        if self.geometric_aug is not None:
            patches = self.geometric_aug(patches)
        else:
            patches = T.Resize(self.output_size_px)(patches)

        # apply patch normalization
        if self.instance_norm:
            patches_norm = []
            for i in range(len(patches)):
                patch = patches[i]
                patch_mean = patch.mean(dim=(-2, -1), keepdim=True)
                patch_std = patch.std(dim=(-2, -1), keepdim=True)
                patch_std = torch.clamp(patch_std, min=1e-6)
                patch = (patch - patch_mean) / patch_std
                patches_norm.append(patch)
            patches = patches_norm
        elif self.mean is not None and self.std is not None:
            patches = T.Normalize(self.mean, self.std)(patches)

        return patches


class PatchTransformMultiCrop(PatchTransform):
    def __init__(self, num_crops=2, **kwargs):
        super().__init__(**kwargs)
        self.num_crops = num_crops

    def __call__(self, sample):
        patches = sample["patches"]
        patches_multi = []
        for _ in range(self.num_crops):
            patches_multi.append(torch.stack(self.transform(patches), dim=0))
        sample["patches"] = patches_multi
        return sample


class NormalizeDataFormat:
    def __init__(self):
        pass

    def __call__(self, sample):
        # convert to expected format
        if isinstance(sample["image"], PIL.Image.Image):
            image = np.array(sample["image"].convert("RGB"))
            sample["image"] = image

        if "info" in sample:
            sample["image_height_mm"] = float(sample["info"]["heightMm"])
            sample["image_width_mm"] = float(sample["info"]["widthMm"])

            if "GG" in sample["info"]:
                gg = sample["info"]["GG"]
                if np.isnan(gg):
                    sample["grade_group"] = 0
                else:
                    sample["grade_group"] = int(sample["info"]["GG"])
                sample["pca"] = int(sample["grade_group"] >= 1)
                sample["cspca"] = int(sample["grade_group"] >= 2)

            sample.pop("info")

        if "path" in sample:
            sample["sample_id"] = sample["path"]

        return sample


def show_output_batch(batch):
    import matplotlib.pyplot as plt

    images = batch["image"]

    from torchvision.utils import make_grid

    grid = make_grid(images, nrow=4)
    plt.figure()
    plt.imshow(grid.permute(1, 2, 0))

    if "needle_mask" in batch:
        needle_masks = batch["needle_mask"]
        grid = make_grid(needle_masks, nrow=4)
        plt.figure()
        plt.imshow(grid.permute(1, 2, 0), vmax=1, vmin=0)

    image = batch["image"][0]

    pos = batch["patch_positions_xyxy"][0]

    from torchvision.utils import draw_bounding_boxes

    image_with_box = draw_bounding_boxes(
        image,
        boxes=pos,
        colors="red",
        labels=[str(i) for i in range(len(pos))],
    )
    plt.figure()
    plt.imshow(image_with_box.permute(1, 2, 0))
    plt.figure()

    patches = batch["patches"][0]
    grid = make_grid(patches, nrow=8)
    plt.imshow(grid.permute(1, 2, 0), vmax=1, vmin=0)

    # plt.imshow(batch['patches'][0][0].permute(1, 2, 0), vmax=1, vmin=0)
