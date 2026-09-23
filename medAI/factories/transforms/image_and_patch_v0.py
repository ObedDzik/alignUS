from medAI import registry
from medAI.registry import build, register
from medAI.transforms import image_patch_transform as IPT
from medAI.transforms.geometric_augmentations import GeometricAugmentations
from medAI.transforms import pixel_augmentations as PXL
from medAI.transforms.ibot import DataAugmentation, DataAugmentationDINOBasic
from medAI.transforms.mask_sampling import MaskSampler
import medAI.transforms.patch_extraction
from medAI.transforms.weak_segmentation import MapLabelToSegmentationTarget
from medAI.transforms.api_compatibility import AddAliasKeyToDict
from typing import Any, Dict, Optional, Tuple
from torchvision.transforms import Compose


__all__ = [
    "multicrop_image_and_patch_v1",
    "single_crop_image_and_patch_v1",
]


def multicrop_image_and_patch_v1(
    *,
    num_local_crops: int = 0,
    output_size_px: Tuple[int, int] = (224, 224),
    output_size_px_local: Tuple[int, int] = (96, 96),
    patch_size: Tuple[int, int] = (7, 7),
    patch_stride: Tuple[int, int] = (1, 1),
    n_patches_per_sample: int = 16,
    mask_keys_for_patches: list[str] = [],
    mask_thresholds_for_patches: list[float] = [],
    patch_output_size_px: Tuple[int, int] = (128, 128),
    geometric_aug_kw: Optional[Dict[str, Any]] = None,  # global crops
    geometric_aug_local_kw: Optional[Dict[str, Any]] = None,  # local crops
    pixel_aug_kw: Optional[Dict[str, Any]] = None,  # gamma/contrast, etc.
    patch_output_geometric_aug_kw: Optional[Dict[str, Any]] = None,
    mask_shape: str = "block",
    output_adapter: str | None = None,
    mean=[0, 0, 0],
    std=[1, 1, 1],
):

    # Defaults that mirror your YAML
    if geometric_aug_kw is None:
        geometric_aug_kw = dict(
            random_crop_scale=(0.4, 1.0), horizontal_flip=True, vertical_flip=False
        )
    if geometric_aug_local_kw is None:
        geometric_aug_local_kw = dict(
            random_crop_scale=(0.05, 0.4), horizontal_flip=True, vertical_flip=True
        )
    if pixel_aug_kw is None:
        pixel_aug_kw = dict(gamma_range=(0.7, 2.0), contrast_range=(0.6, 3.0))
    if patch_output_geometric_aug_kw is None:
        patch_output_geometric_aug_kw = dict(
            random_crop_scale=(0.4, 1.0), horizontal_flip=True, vertical_flip=True
        )

    return Compose(
        [
            IPT.NormalizeDataFormat(),
            medAI.transforms.patch_extraction.AddPatchesFromSlidingWindowPhysicalCoordinates(
                patch_size=patch_size,  # ✅ not swapped
                patch_stride=patch_stride,
                n_patches_per_sample=n_patches_per_sample,
                mask_keys=mask_keys_for_patches,
                mask_thresholds=mask_thresholds_for_patches,
            ),
            IPT.ImageTransformMultiCrop(
                output_size_px=output_size_px,
                num_local_crops=num_local_crops,
                output_size_px_local=output_size_px_local,
                geometric_aug=GeometricAugmentations(**geometric_aug_kw),
                geometric_aug_local=GeometricAugmentations(**geometric_aug_local_kw),
                pixel_aug=PXL.PixelAugmentations(**pixel_aug_kw),  # key name: pixel_aug
                mean=mean,
                std=std,
            ),
            IPT.PatchTransformMultiCrop(
                output_size_px=patch_output_size_px,
                geometric_aug=GeometricAugmentations(**patch_output_geometric_aug_kw),
            ),
            MaskSampler(mask_shape=mask_shape),
            build("adapter", output_adapter) if output_adapter else lambda x: x,
        ]
    )


def single_crop_image_and_patch_v1(
    *,
    output_size_px: Tuple[int, int] = (224, 224),
    patch_size: Tuple[int, int] = (7, 7),
    patch_stride: Tuple[int, int] = (1, 1),
    n_patches_per_sample: int = 16,
    mask_keys_for_patches: list[str] = [],
    mask_thresholds_for_patches: list[float] = [],
    patch_output_size_px: Tuple[int, int] = (128, 128),
    geometric_aug_kw: Optional[Dict[str, Any]] = None,  # global crops
    pixel_aug_kw: Optional[Dict[str, Any]] = None,  # gamma/contrast, etc.
    patch_geometric_aug_kw: Optional[Dict[str, Any]] = None,
    label_key="pca",
    mean=[0, 0, 0],
    std=[1, 1, 1],
):

    return Compose(
        [
            IPT.NormalizeDataFormat(),
            medAI.transforms.patch_extraction.AddPatchesFromSlidingWindowPhysicalCoordinates(
                patch_size=patch_size,  # ✅ not swapped
                patch_stride=patch_stride,
                n_patches_per_sample=n_patches_per_sample,
                mask_keys=mask_keys_for_patches,
                mask_thresholds=mask_thresholds_for_patches,
            ),
            IPT.ImageTransform(
                output_size_px=output_size_px,
                geometric_aug=(
                    GeometricAugmentations(**geometric_aug_kw)
                    if geometric_aug_kw is not None
                    else None
                ),
                pixel_aug=(
                    PXL.PixelAugmentations(**pixel_aug_kw)
                    if pixel_aug_kw is not None
                    else None
                ),
                mask_keys=["needle_mask"],
                mean=mean,
                std=std,
            ),
            IPT.PatchTransform(
                output_size_px=patch_output_size_px,
                geometric_aug=(
                    GeometricAugmentations(**patch_geometric_aug_kw)
                    if patch_geometric_aug_kw is not None
                    else None
                ),
            ),
            MapLabelToSegmentationTarget(
                "needle_mask", label_key=label_key, output_key="target_mask"
            ),
            AddAliasKeyToDict(label_key, "label"),
        ]
    )
