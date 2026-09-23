import torch 
import random
import numpy as np


class RandomTranslation: 
    """
    A random translation that can be applied to multiple images of different sizes.
    (torchvision.transforms.RandomAffine does not support multiple images of different sizes.)

    Args:
        translation (tuple): maximum translation in x and y directions. Float between 0 and 1.
    """
    def __init__(self, translation=(0.2, 0.2)): 
        self.translation = translation

    def __call__(self, *images):
        """
        Apply a random translation to the input images.

        Args:
            *images (torch.Tensor): input images. Each image should have shape (C, H, W).
        """

        from torchvision.transforms.functional import affine
        from random import uniform

        h_factor, w_factor = uniform(-self.translation[0], self.translation[0]), uniform(-self.translation[1], self.translation[1])

        outputs = []
        for image in images:
            H, W = image.shape[-2:]
            translate_x = int(w_factor * W)
            translate_y = int(h_factor * H)
            outputs.append(affine(image, angle=0, translate=(translate_x, translate_y), scale=1, shear=0))

        return outputs[0] if len(outputs) == 1 else outputs


# lifted from https://github.com/LiheYoung/UniMatch/blob/main/dataset/transform.py
def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask