from torchvision.transforms import v2 as T
import torch
import numpy as np
from torchvision import tv_tensors as tvt
from PIL import Image

T.Resize


class RandomGamma(T.Transform):
    _transformed_types = (tvt.Image, Image.Image, torch.Tensor)

    def __init__(self, gamma_range=(0.5, 4)):
        super().__init__()
        self.gamma_range = gamma_range

    def transform(self, img, *args, **kwargs):
        gamma = np.random.uniform(*self.gamma_range)
        return T.functional.adjust_gamma(img, gamma)


class RandomContrast(T.Transform):
    _transformed_types = (tvt.Image, Image.Image, torch.Tensor)

    def __init__(self, contrast_range=(0.6, 3)):
        super().__init__()
        self.contrast_range = contrast_range

    def transform(self, img, *args, **kwargs):
        contrast = np.random.uniform(*self.contrast_range)
        return T.functional.adjust_contrast(img, contrast)


class SaltAndPepperNoise(T.Transform):
    _transformed_types = (tvt.Image, Image.Image, torch.Tensor)

    def __init__(self, amount=0.1, scale=1):
        super().__init__()
        self.amount = amount
        self.scale = scale

    def transform(self, img, *args, **kwargs):
        mask = torch.rand(*img.shape) < self.amount
        img[mask] = torch.randint(0, 256, (mask.sum(),)) / 255 * self.scale
        return img


class RandomSpeckleNoise(T.Transform):
    _transformed_types = (tvt.Image, Image.Image, torch.Tensor)

    def __init__(self, amount=(0.1, 0.5)):
        super().__init__()
        self.amount = amount

    def transform(self, img, *args, **kwargs):
        if img.min() < 0 or img.max() > 1:
            raise ValueError("Image must be in the range [0, 1]")
        amount = np.random.uniform(*self.amount)
        img = img + amount * img.std() * torch.randn(*img.shape)
        img = torch.clip(img, 0, 1)
        return img


class PixelAugmentations(T.Transform):
    _transformed_types = (tvt.Image, Image.Image, torch.Tensor)

    def __init__(
        self,
        gamma_range=None,
        contrast_range=None,
        salt_and_pepper_amount=None,
        speckle_noise_amount=None,
    ):
        super().__init__()
        self.augmentations = []
        if gamma_range is not None:
            self.augmentations.append(RandomGamma(gamma_range))
        if contrast_range is not None:
            self.augmentations.append(RandomContrast(contrast_range))
        if salt_and_pepper_amount is not None:
            self.augmentations.append(SaltAndPepperNoise(salt_and_pepper_amount))
        if speckle_noise_amount is not None:
            self.augmentations.append(RandomSpeckleNoise(speckle_noise_amount))

    def transform(self, img, *args, **kwargs):
        for aug in self.augmentations:
            img = aug(img)
        return img


def simple_build_pixel_augmentations(level='medium'): 
    if level == 'none':
        return T.Identity()
    elif level == 'low':
        return PixelAugmentations(
            gamma_range=(0.8, 1.2),
            contrast_range=(0.8, 1.2),
            salt_and_pepper_amount=0.01,
            speckle_noise_amount=None,
        )
    elif level == 'medium':
        return PixelAugmentations(
            gamma_range=(0.5, 1.5),
            contrast_range=(0.6, 1.5),
            salt_and_pepper_amount=0.05,
            speckle_noise_amount=0.1,
        )
    elif level == 'high':
        return PixelAugmentations(
            gamma_range=(0.3, 2.0),
            contrast_range=(0.4, 2.0),
            salt_and_pepper_amount=0.1,
            speckle_noise_amount=0.2,
        )
    else:
        raise ValueError(f"Unknown augmentation level: {level}")