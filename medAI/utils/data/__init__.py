from .patch_extraction import *

from typing import Sequence
import numpy as np


def crop_to_mask(image, mask, threshold=0.5, padding=0):
    if not isinstance(padding, Sequence):
        padding = [padding, padding]

    mask = mask > threshold
    x, y = np.where(mask)
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_min = max(0, x_min - padding[0])
    x_max = min(image.shape[0], x_max + padding[0])
    y_min = max(0, y_min - padding[1])
    y_max = min(image.shape[1], y_max + padding[1])

    return image[x_min:x_max, y_min:y_max]


def crop_or_pad_from_top_right(image, shape, fill=0):
    if not isinstance(shape, Sequence): 
        shape = (shape, shape)
    output = np.ones(shape, dtype=image.dtype) * fill
    x, y = image.shape[:2]
    x_out, y_out = shape[:2]
    x_start = 0 
    y_start = 0
    x_end = min(x, x_out)
    y_end = min(y, y_out)
    output[x_start:x_end, y_start:y_end] = image[:x_end - x_start, :y_end - y_start]
    return output


from typing import Sequence
import numpy as np


def get_crop_to_mask_params(img_size, mask, threshold=0.5, padding=0):
    """Return the parameters to crop an image to the bounding box of a mask.

    Args:
        mask (np.ndarray): The mask to crop to.
        threshold (float): The threshold to apply to the mask.
        padding (int): The padding to add around the mask.

    Returns:
        Tuple[int, int, int, int]: The parameters to crop the image to the mask. (Top, Left, Height, Width)
    """
    if not isinstance(padding, Sequence):
        padding = [padding, padding]

    mask = mask > threshold
    x, y = np.where(mask)
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_min = max(0, x_min - padding[0])
    x_max = min(img_size[0], x_max + padding[0])
    y_min = max(0, y_min - padding[1])
    y_max = min(img_size[1], y_max + padding[1])

    return x_min, y_min, x_max - x_min, y_max - y_min