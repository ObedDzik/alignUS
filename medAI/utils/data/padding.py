import numpy as np


def strip_padding(reference_arr, *additional_arrs):
    """Removes rows and columns that are all zeros from an image.

    Args:
        arr - a numpy array of shape (H, W, T) where H is the height (axial direction), W is the width (lateral direction)
            and T is the number of frames.

    Returns:
        arr - a numpy array of shape (H, W, T) where H is the height (axial direction), W is the width (lateral direction)
            and T is the number of frames. The returned array has rows and columns that are all zeros removed.
    """
    H, W, *_ = reference_arr.shape
    column_is_dead = np.zeros(W, dtype=bool)
    row_is_dead = np.zeros(H, dtype=bool)
    for i in range(W):
        column_is_dead[i] = np.allclose(reference_arr[:, i, ...], 0)
    for i in range(H):
        row_is_dead[i] = np.allclose(reference_arr[i, :, ...], 0)

    reference_arr = reference_arr[~row_is_dead, :, ...]
    reference_arr = reference_arr[:, ~column_is_dead, ...]

    outputs = [reference_arr]
    for arr in additional_arrs:
        assert arr.shape == reference_arr.shape, f"Shape mismatch"
        arr = arr[~row_is_dead, :, ...]
        arr = arr[:, ~column_is_dead, ...]
        outputs.append(arr)

    if len(outputs) == 1:
        return outputs[0]
    else:
        return tuple(outputs)