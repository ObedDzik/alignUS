import torch 
from torchvision.transforms import v2 as T


class BKPatchTransform:
    def __init__(
        self,
        size=256,
        instance_norm=True,
        crop_size=None,
        random_crop=False,
        horizontal_flip=False,
    ):
        self.size = size
        self.instance_norm = instance_norm
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.horizontal_flip = horizontal_flip

    def __call__(self, patch):
        patch = torch.tensor(patch).float().unsqueeze(0)
        if self.instance_norm:
            patch = (patch - patch.mean()) / patch.std()
        patch = T.Resize((self.size, self.size))(patch)

        if self.crop_size is not None:
            if self.crop_size > self.size:
                raise ValueError(f"crop_size must be less than or equal to {self.size}")
            if self.random_crop:
                patch = T.RandomCrop((self.crop_size, self.crop_size))(patch)
            else:
                patch = T.CenterCrop((self.crop_size, self.crop_size))(patch)

        if self.horizontal_flip:
            patch = T.RandomHorizontalFlip()(patch)

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
        self.image_transform.return_masks = False

    def __call__(self, item):
        img = item[NCT2013FullImageWithPatchesDataset.DATA_KEY_IMAGE]
        prostate_mask = item[NCT2013FullImageWithPatchesDataset.DATA_KEY_PROSTATE_MASK]
        needle_mask = item[NCT2013FullImageWithPatchesDataset.DATA_KEY_NEEDLE_MASK]

        metadata = item[NCT2013FullImageWithPatchesDataset.DATA_KEY_METADATA]
        patch = item[NCT2013FullImageWithPatchesDataset.DATA_KEY_PATCH]
        box = item[NCT2013FullImageWithPatchesDataset.DATA_KEY_PATCH_POSITION_XYXY]
        cancer_label = item[NCT2013FullImageWithPatchesDataset.DATA_KEY_CANCER_LABEL]

        img, box = self.image_transform(
            img, prostate_mask, needle_mask, box
        )

        patch = self.patch_transform(patch)
        cancer_label = torch.tensor(cancer_label).long()

        out = {
            NCT2013FullImageWithPatchesDataset.DATA_KEY_IMAGE: img,
            #NCT2013FullImageWithPatchesDataset.DATA_KEY_PROSTATE_MASK: prostate_mask,
            #NCT2013FullImageWithPatchesDataset.DATA_KEY_NEEDLE_MASK: needle_mask,
            NCT2013FullImageWithPatchesDataset.DATA_KEY_PATCH: patch,
            NCT2013FullImageWithPatchesDataset.DATA_KEY_PATCH_POSITION_XYXY: box,
            NCT2013FullImageWithPatchesDataset.DATA_KEY_CANCER_LABEL: cancer_label,
            NCT2013FullImageWithPatchesDataset.DATA_KEY_METADATA: metadata,
        }

        return out