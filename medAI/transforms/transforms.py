from dataclasses import dataclass
import random
from typing import Literal
from PIL import Image
from torchvision import transforms
import torchvision
from torchvision.transforms.v2.functional import horizontal_flip
# from . import utils
import torch
import numpy as np
from torchvision.transforms.v2 import Lambda
from torchvision.transforms import v2 as T
from torchvision import tv_tensors as tvt


def NormalizeToTensor(mean=(0.485, 0.456, 0.406), std=(0.485, 0.456, 0.406)):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


class SpeckleNoise:
    def __init__(self, amount=0.1, apply_to_greyscale=True):
        self.amount = amount
        self.apply_to_greyscale = apply_to_greyscale

    def __call__(self, img):
        fmt = img.format
        if self.apply_to_greyscale:
            img = img.convert("L")
        img = np.array(img) / 255.0
        img = img + self.amount * img.std() * np.random.randn(*img.shape)
        img = np.clip(img, 0, 1)
        return Image.fromarray((img * 255).astype(np.uint8)).convert(fmt)


class SaltAndPepperNoise:
    def __init__(self, amount=0.1, apply_to_greyscale=True):

        self.amount = amount
        self.apply_to_greyscale = apply_to_greyscale

    def __call__(self, img):
        if self.apply_to_greyscale:
            img = img.convert("L")
        img = np.array(img)
        mask = np.random.rand(*img.shape) < self.amount
        img[mask] = np.random.randint(0, 256, mask.sum())
        return Image.fromarray(img).convert("RGB")


class RandomNoise:
    def __init__(self, amount_range=(0, 1), apply_to_greyscale=True, type="speckle"):
        self.amount_range = amount_range
        self.apply_to_greyscale = apply_to_greyscale
        self.type = type

    def __call__(self, img):
        if self.type == "speckle":
            return SpeckleNoise(
                np.random.uniform(*self.amount_range), self.apply_to_greyscale
            )(img)
        elif self.type == "salt_and_pepper":
            return SaltAndPepperNoise(
                np.random.uniform(*self.amount_range), self.apply_to_greyscale
            )(img)
        else:
            raise ValueError(f"Unknown noise type: {self.type}")


class RandomGamma:
    def __init__(self, gamma_range=(0.5, 4)):
        self.gamma_range = gamma_range

    def __call__(self, img):
        gamma = np.random.uniform(*self.gamma_range)
        return T.functional.adjust_gamma(img, gamma)


class RandomContrast:
    def __init__(self, contrast_range=(0.6, 3)):
        self.contrast_range = contrast_range

    def __call__(self, img):
        contrast = np.random.uniform(*self.contrast_range)
        return T.functional.adjust_contrast(img, contrast)


def get_pixel_level_augmentations_for_greyscale_images(
    blur_prob: float = 0.0,
    speckle_prob: float = 0.0,
    speckle_amount: tuple[float, float] = (0.01, 0.1),
    salt_and_pepper_prob: float = 0.0,
    salt_and_pepper_amount: tuple[float, float] = (0.01, 0.1),
    random_gamma_prob: float = 0.0,
    random_gamma_range: tuple[float, float] = (0.5, 4),
    random_contrast_prob: float = 0.0,
    random_contrast_range: tuple[float, float] = (0.6, 3),
):
    return T.Compose(
        [
            T.RandomApply(
                [RandomNoise((speckle_amount), apply_to_greyscale=True, type="speckle")],
                p=speckle_prob,
            ) if speckle_prob > 0 else T.Identity(),
            T.RandomApply(
                [
                    RandomNoise(
                        (salt_and_pepper_amount), apply_to_greyscale=True, type="salt_and_pepper"
                    )
                ],
                p=salt_and_pepper_prob,
            ) if salt_and_pepper_prob > 0 else T.Identity(),
            T.RandomApply([RandomGamma(random_gamma_range)], p=random_gamma_prob) if random_gamma_prob > 0 else T.Identity(),
            T.RandomApply([RandomContrast(random_contrast_range)], p=random_contrast_prob) if random_contrast_prob > 0 else T.Identity(),
            T.RandomApply([T.GaussianBlur(3)], p=blur_prob) if blur_prob > 0 else T.Identity(),
        ]
    )


def get_pixel_level_augmentations_by_mode(mode: Literal["none", "weak", "strong"]): 
    if mode == 'weak': 
        pixel_level_augmentations = get_pixel_level_augmentations_for_greyscale_images(
            speckle_prob=0.2, 
            salt_and_pepper_prob=0.2,
            random_gamma_prob=0.4, 
            random_contrast_prob=0.2
        )
    elif mode == 'strong':
        pixel_level_augmentations = get_pixel_level_augmentations_for_greyscale_images(
            speckle_prob=0.5, 
            salt_and_pepper_prob=0.5,
            random_gamma_prob=0.5, 
            random_contrast_prob=0.5
        )
    else:
        pixel_level_augmentations = T.Identity()
    return pixel_level_augmentations


class ImageMaskTransform:
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

        if pixel_level_augmentations_mode == 'weak': 
            self.pixel_level_augmentations = get_pixel_level_augmentations_for_greyscale_images(
                speckle_prob=0.2, 
                salt_and_pepper_prob=0.2,
                random_gamma_prob=0.4, 
                random_contrast_prob=0.2
            )
        elif pixel_level_augmentations_mode == 'strong':
            self.pixel_level_augmentations = get_pixel_level_augmentations_for_greyscale_images(
                speckle_prob=0.5, 
                salt_and_pepper_prob=0.5,
                random_gamma_prob=0.5, 
                random_contrast_prob=0.5
            )
        else:
            self.pixel_level_augmentations = T.Identity()
        
    def __call__(self, image: Image.Image, mask: Image.Image):
        image = image.convert("RGB")
        image = tvt.Image(image) / 255.0
        mask = tvt.Mask(mask)

        if self.translate:
            image, mask = T.RandomAffine(degrees=0, translate=(0.1, 0.1))(image, mask)

        if self.random_crop_scale is not None:
            image, mask = T.RandomResizedCrop(self.size, self.random_crop_scale)(image, mask)
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
        return image, mask[0].long()

    def _random_border_erase(self, image: torch.Tensor): 
        H, W = image.shape[-2:]
        border_erase_width = np.random.uniform(*self.random_border_erase_range)
        border_erase_width = int(border_erase_width * W)
        image[:, :, :border_erase_width] = 0
        image[:, :, -border_erase_width:] = 0
        return image


@dataclass
class MultiCropDataAugmentationForSSL:
    global_crops_scale: tuple[float, float] = (0.14, 1)
    local_crops_scale: tuple[float, float] = (0.05, 0.4)
    global_crops_number: int = 2
    local_crops_number: int = 0
    global_crops_size: int = 224
    local_crops_size: int = 96
    jitter_prob: float = 0.0
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    blur_prob_1: float = 0.0
    blur_prob_2: float = 0.0
    solarization_prob: float = 0.0
    speckle_prob: float = 0.0
    salt_and_pepper_prob: float = 0.0
    random_gamma_prob: float = 0.0
    random_contrast_prob: float = 0.0
    initial_crop_size: int | None = None
    initial_crop_scale: tuple[float, float] | None = None
    initial_resize_size: int | None = None
    noise_before_crop: bool = False
    return_PIL: bool = False
    horizontal_flip_prob: float = 0.5

    def __post_init__(
        self,
    ):
        if self.initial_crop_size is not None:
            if self.initial_crop_scale is not None:
                self.initial_crop = T.RandomResizedCrop(
                    self.initial_crop_size, scale=self.initial_crop_scale
                )
            else:
                self.initial_crop = T.Compose(
                    [
                        T.RandomCrop(self.initial_crop_size),
                        T.Resize(self.initial_crop_size),
                    ]
                )
        else:
            self.initial_crop = lambda x: x

        strong_augs = T.Compose(
            [
                T.RandomApply(
                    [RandomNoise(apply_to_greyscale=True, type="speckle")],
                    p=self.speckle_prob,
                ),
                T.RandomApply(
                    [
                        RandomNoise(
                            (0.01, 0.1), apply_to_greyscale=True, type="salt_and_pepper"
                        )
                    ],
                    p=self.salt_and_pepper_prob,
                ),
                T.RandomApply([RandomGamma()], p=self.random_gamma_prob),
                T.RandomApply([RandomContrast()], p=self.random_contrast_prob),
            ]
        )

        weak_augs = T.Compose(
            [
                T.RandomHorizontalFlip(p=self.horizontal_flip_prob),
                T.RandomApply(
                    [
                        T.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=self.jitter_prob,
                ),
                T.RandomGrayscale(p=0.2),
                utils.GaussianBlur(p=self.blur_prob_1),
            ]
        )

        normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )

        self.global_crops_number = self.global_crops_number

        def make_global_transform(crop):
            if self.noise_before_crop:
                transforms_ = [strong_augs, crop, weak_augs]
            else:
                transforms_ = [crop, strong_augs, weak_augs]
            if self.return_PIL:
                transforms_.append(Lambda(lambda x: x.convert("RGB")))
            else:
                transforms_.append(normalize)
            return T.Compose(transforms_)

        # transformation for the first global crop
        self.global_transfo1 = make_global_transform(
            T.RandomResizedCrop(
                self.global_crops_size,
                scale=self.global_crops_scale,
                interpolation=Image.BICUBIC,
            ),
        )
        # transformation for the rest of global crops
        self.global_transfo2 = make_global_transform(
            T.RandomResizedCrop(
                self.global_crops_size,
                scale=self.global_crops_scale,
                interpolation=Image.BICUBIC,
            ),
        )
        # transformation for the local crops
        self.local_transfo = make_global_transform(
            T.RandomResizedCrop(
                self.local_crops_size,
                scale=self.local_crops_scale,
                interpolation=Image.BICUBIC,
            ),
        )

    def __call__(self, image):
        image = self.initial_crop(image)
        crops = []
        crops.append(self.global_transfo1(image))
        for _ in range(self.global_crops_number - 1):
            crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops

    def to_numpy(self, tensor):
        C, H, W = tensor.shape
        tensor *= torch.tensor(self.std)[..., None, None]
        tensor += torch.tensor(self.mean)[..., None, None]
        return tensor.permute(1, 2, 0).numpy()


class RFPatchTransformV0: 
    def __init__(
        self,
        initial_resize=224,
        instance_norm=True,
        crop_size=None,
        random_crop=False,
        crop_scale=None,
        horizontal_flip=False,
        vertical_flip=False, 
        aggregation: Literal[None, 'mean'] = None
    ):
        self.initial_resize = initial_resize
        self.instance_norm = instance_norm
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.horizontal_flip = horizontal_flip
        self.vertical_flip=vertical_flip
        self.aggregation = aggregation
        self.crop_scale = crop_scale

    def __call__(self, patch):
        if patch.ndim == 2: 
            patch = patch[:, :, None]
        patch = torch.tensor(patch).float().permute(2, 0, 1)
        if self.instance_norm:
            patch = (patch - patch.mean()) / patch.std()
        if self.aggregation == 'mean': 
            patch = patch.mean(dim=0, keepdim=True)

        patch = T.Resize((self.initial_resize, self.initial_resize))(patch)

        if self.crop_size is not None:
            if self.crop_size > self.initial_resize:
                raise ValueError(f"crop_size must be less than or equal to {self.initial_resize}")
            if self.random_crop:
                if self.crop_scale is not None: 
                    patch = T.RandomResizedCrop((self.crop_size, self.crop_size), self.crop_scale)
                else: 
                    patch = T.RandomCrop((self.crop_size, self.crop_size))(patch)
            else:
                patch = T.CenterCrop((self.crop_size, self.crop_size))(patch)

        if self.horizontal_flip:
            patch = T.RandomHorizontalFlip()(patch)
        if self.vertical_flip: 
            patch = T.RandomVerticalFlip()(patch)

        return patch


class BModeImageAndMaskTransformV0:
    def __init__(
        self,
        image_size=256,
        mean: tuple[float, float, float] = (0, 0, 0),
        std: tuple[float, float, float] = (1, 1, 1),
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

    def __call__(self, image, prostate_mask, needle_mask, boxes):
        image = T.ToImage()(image)
        image = (image - image.min()) / (image.max() - image.min())

        if self.pixel_level_augmentations_mode != "none":
            image = T.ToDtype(torch.uint8, scale=True)(image)
            image = T.ToPILImage()(image)
            image = self.pixel_level_augmentations(image)
            image = T.ToImage()(image)
            image = T.ToDtype(torch.float32, scale=True)(image)

        image = T.Normalize(mean=self.mean, std=self.std)(image)

        image = tvt.Image(image)
        T.ToImage()(prostate_mask)
        prostate_mask = tvt.Mask(prostate_mask)
        boxes = torch.tensor(boxes).long()
        boxes = tvt.BoundingBoxes(boxes, format="xyxy", canvas_size=image.shape[-2:])

        T.ToImage()(needle_mask)
        needle_mask = tvt.Mask(needle_mask)

        H, W = image.shape[-2:]

        if self.return_masks:
            # resize masks to image shape
            prostate_mask = T.Resize((H, W))(prostate_mask)
            needle_mask = T.Resize((H, W))(needle_mask)

            # geometric transformations
            image, prostate_mask, needle_mask, boxes = T.Resize(
                (self.image_size, self.image_size)
            )(image, prostate_mask, needle_mask, boxes)

            if self.random_resized_crop:
                image, prostate_mask, needle_mask, boxes = T.RandomResizedCrop(
                    self.image_size, scale=self.random_resized_crop
                )(image, prostate_mask, needle_mask, boxes)

            if self.random_translation:
                image, prostate_mask, needle_mask, boxes = T.RandomAffine(
                    degrees=0, translate=self.random_translation
                )(image, prostate_mask, needle_mask, boxes)

            return image, prostate_mask, needle_mask, boxes
        else: 
            image, boxes = T.Resize((self.image_size, self.image_size))(image, boxes)
            if self.random_resized_crop:
                image, boxes = T.RandomResizedCrop(
                    self.image_size, scale=self.random_resized_crop
                )(image, boxes)

            if self.random_translation:
                image, boxes = T.RandomAffine(
                    degrees=0, translate=self.random_translation
                )(image, boxes)

            return image, boxes