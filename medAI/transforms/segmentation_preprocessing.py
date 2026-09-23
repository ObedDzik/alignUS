# from medAI.modeling.ibot.medsam_ibot import MedSAMIBot
from medAI.transforms.transforms import (
    get_pixel_level_augmentations_for_greyscale_images,
)

# from src.models import get_model, MODEL_REGISTRY
from torchvision.transforms import v2 as T
from torchvision import tv_tensors as tvt
import numpy as np
import torch


class SegmentationTransform:
    """Default transform for image-mask pairs, PIL to tensor"""

    def __init__(
        self,
        size: int = 512,
        mean: tuple[float, float, float] = (0, 0, 0),
        std: tuple[float, float, float] = (1, 1, 1),
        random_crop_scale: tuple[float, float] | None = None,
        to_tensor: bool = True,
        translate: bool = False,
        pixel_level_augmentations_mode: str = "none",
        random_border_erase_range: tuple[float, float] | None = None,
    ):
        self.size = size
        self.mean = mean
        self.std = std
        self.to_tensor = to_tensor
        self.translate = translate
        self.random_crop_scale = random_crop_scale
        self.random_border_erase_range = random_border_erase_range

        if pixel_level_augmentations_mode == "weak":
            self.pixel_level_augmentations = (
                get_pixel_level_augmentations_for_greyscale_images(
                    speckle_prob=0.2,
                    salt_and_pepper_prob=0.2,
                    random_gamma_prob=0.4,
                    random_contrast_prob=0.2,
                )
            )
        elif pixel_level_augmentations_mode == "strong":
            self.pixel_level_augmentations = (
                get_pixel_level_augmentations_for_greyscale_images(
                    speckle_prob=0.5,
                    salt_and_pepper_prob=0.5,
                    random_gamma_prob=0.5,
                    random_contrast_prob=0.5,
                )
            )
        else:
            self.pixel_level_augmentations = T.Identity()

    def __call__(self, item: dict):
        item = item.copy()  # Avoid modifying the original item
        image = item["image"]
        image = np.array(image.convert("RGB"))
        H, W, C = image.shape
        mask = item["mask"] if 'mask' in item else np.zeros((H, W, 1), dtype=np.uint8)

        image = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])(image)
        image = tvt.Image(image)  # Convert to tvt.Image

        mask = T.ToImage()(mask)  # Convert mask to tvt.Image
        mask = tvt.Mask(mask)

        if self.translate:
            image, mask = T.RandomAffine(degrees=0, translate=(0.1, 0.1))(image, mask)

        if self.random_crop_scale is not None:
            image, mask = T.RandomResizedCrop(self.size, self.random_crop_scale)(
                image, mask
            )
        else:
            image, mask = T.Resize((self.size, self.size))(image, mask)

        # pixel level augmentations expect PIL
        image = T.ToPILImage()(image)
        image = self.pixel_level_augmentations(image)
        image = T.ToTensor()(image)

        if self.random_border_erase_range is not None:
            image = self._random_border_erase(image)

        if not self.to_tensor:
            image, mask = T.ToPILImage()(image, mask)
            return image, mask

        image, mask = T.ToTensor()(image, mask)
        image = T.ToDtype(torch.float32)(image)
        image = T.Normalize(mean=self.mean, std=self.std)(image)

        item["image"] = image
        item["mask"] = mask[0].long()

        return item

    def _random_border_erase(self, image: torch.Tensor):
        H, W = image.shape[-2:]
        border_erase_width = np.random.uniform(*self.random_border_erase_range)
        border_erase_width = int(border_erase_width * W)
        image[:, :, :border_erase_width] = 0
        image[:, :, -border_erase_width:] = 0
        return image