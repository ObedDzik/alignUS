from medAI import registry
from medAI.transforms.ibot import DataAugmentation, DataAugmentationDINOBasic
from medAI.transforms.mask_sampling import MaskSampler
import inspect
import logging


__all__ = [
    "get_ibot_ssl_transform",
]


def get_ibot_ssl_transform(
    input_format="torchvision",
    output_format="tuple",
    add_masks=False,
    mask_kw={},
    **crop_augmentation_kw,
):
    if crop_augmentation_kw.pop("mode", None) == "basic":
        crop_augmentation_kw = crop_augmentation_kw.copy()
        argnames = inspect.signature(DataAugmentationDINOBasic).parameters.keys()
        crop_augmentation_kw = {k: crop_augmentation_kw[k] for k in argnames}
        augmentations = DataAugmentationDINOBasic(**crop_augmentation_kw)
    else:
        augmentations = DataAugmentation(**crop_augmentation_kw)
    logging.info(f"Data augmentation: {augmentations}")

    def transform(sample):
        if input_format == "torchvision":
            image = sample
            return augmentations(image)

        elif input_format == "dict":
            if add_masks:
                sample = MaskSampler(**mask_kw)(sample)

            image = sample["image"] if "image" in sample else sample["images"]
            label = sample.get("label", 0)
            image = augmentations(image)

            if add_masks:
                masks = sample["masks"]

            if output_format == "dict":
                out_sample = dict(
                    image=image,
                    label=label,
                )
                if add_masks:
                    out_sample["masks"] = masks
                return out_sample
            elif output_format == "tuple":
                return (image, label, masks) if add_masks else (image, label)

        else:
            raise ValueError("Unknown input_format")
        # elif input_format == "dict":

    return transform