from dataclasses import dataclass, field
from typing import Sequence
import torch
from torchvision.transforms import v2 as T
from medAI.registry import register
from medAI.transforms.geometric_augmentations import GeometricAugmentations
from medAI.transforms.patch_extraction import RandomSamplePatches
from medAI.transforms.normalization import InstanceNormalizeImage
from medAI.transforms.pixel_augmentations import simple_build_pixel_augmentations
from medAI.transforms.common import *


__all__ = [
    "MultiCropPatchConfig",
    "build_multicrop_transform_with_patches",
]


@dataclass
class MultiCropPatchConfig:
    num_global_crops: int = 2
    global_crop_size: int = 224
    global_crop_scale: Sequence[float] = (0.4, 1.0)
    global_crop_kw: dict = field(default_factory=dict)

    num_local_crops: int = 0  # currently not used
    local_crop_size: int = 96
    local_crop_scale: Sequence[float] = (0.05, 0.4)
    local_crop_kw: dict = field(default_factory=dict)

    num_patches: int = 4
    patch_size: int = 224
    patch_crop_scale: Sequence[float] = (0.4, 1.0)

    image_mean: Sequence[float] = (0, 0, 0)
    image_std: Sequence[float] = (1, 1, 1)
    patch_mean: Sequence[float] = (0, 0, 0)
    patch_std: Sequence[float] = (1, 1, 1)

    image_pixel_augmentations: str = "none"
    patch_pixel_augmentations: str = "none"

    instance_norm_patches: bool = True


def build_multicrop_transform_with_patches(config: MultiCropPatchConfig):

    patch_norm = T.Normalize(config.patch_mean, config.patch_std)
    if config.instance_norm_patches:
        patch_norm = InstanceNormalizeImage()

    patch_pixel_augmentations = simple_build_pixel_augmentations(
        config.patch_pixel_augmentations
    )
    image_pixel_augmentations = simple_build_pixel_augmentations(
        config.image_pixel_augmentations
    )
    transform = T.Compose(
        [
            RandomSamplePatches(
                num_patches=config.num_patches, patch_size=(config.patch_size, config.patch_size)
            ),
            ApplyTransformToKeys(
                ApplyNTimes(
                    GeometricAugmentations(
                        random_crop_scale=config.global_crop_scale,
                        size=(config.global_crop_size, config.global_crop_size),
                        **config.global_crop_kw
                    ),
                    2,
                ),
                keys=["image", "boxes"],
            ),
            ApplyTransformToKeys(
                ApplyNTimes(
                    GeometricAugmentations(
                        random_crop_scale=config.local_crop_scale,
                        size=(config.patch_size, config.patch_size),
                        **config.local_crop_kw
                    ),
                    2,
                ),
                keys=["patches"],
            ),
            ApplyTransformToKeys(
                T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]),
                keys=["image", "patches"],
            ),
            ApplyTransformToKeys(image_pixel_augmentations, keys=["image"]),
            ApplyTransformToKeys(patch_pixel_augmentations, keys=["patches"]),
            ApplyTransformToKeys(T.Normalize(config.image_mean, config.image_std), keys=["image"]),
            ApplyTransformToKeys(patch_norm, keys=["patches"]),
        ]
    )

    return transform


def show_transformed_sample(sample):
    from torchvision.utils import draw_bounding_boxes

    image = sample["image"][0]
    boxes = sample["boxes"][0]
    image_with_boxes = draw_bounding_boxes(
        image, boxes, labels=[str(i) for i in range(len(boxes))], width=2
    )

    image2 = sample["image"][1]
    boxes2 = sample["boxes"][1]
    image_with_boxes2 = draw_bounding_boxes(
        image2, boxes2, labels=[str(i) for i in range(len(boxes2))], width=2
    )

    from matplotlib import patches, pyplot as plt

    plt.imshow(image_with_boxes.permute(1, 2, 0))
    plt.figure()
    plt.imshow(image_with_boxes2.permute(1, 2, 0))

    plt.figure()
    for i, patch in enumerate(sample["patches"][0]):
        plt.subplot(1, len(sample["patches"][0]), i + 1)
        plt.imshow(patch.permute(1, 2, 0))

    plt.figure()
    for i, patch in enumerate(sample["patches"][1]):
        plt.subplot(1, len(sample["patches"][1]), i + 1)
        plt.imshow(patch.permute(1, 2, 0))