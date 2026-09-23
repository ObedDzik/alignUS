from torchvision.transforms import v2 as T


class GeometricAugmentations:
    def __init__(
        self,
        horizontal_flip=False,
        vertical_flip=False,
        random_crop_scale=(0.4, 1.0),
        random_crop_ratio=(3 / 4, 4 / 3),
        size=(224, 224),
    ):
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.random_crop_scale = random_crop_scale
        self.random_crop_ratio = random_crop_ratio
        self.size = size

    def __call__(self, *args, **kwargs):
        augmentations = []
        if self.horizontal_flip:
            augmentations.append(T.RandomHorizontalFlip())
        if self.vertical_flip:
            augmentations.append(T.RandomVerticalFlip())
        if self.random_crop_scale is not None and self.random_crop_ratio is not None:
            augmentations.append(
                T.RandomResizedCrop(
                    size=self.size,
                    scale=self.random_crop_scale,
                    ratio=self.random_crop_ratio,
                )
            )
        return T.Compose(augmentations)(*args, **kwargs)


