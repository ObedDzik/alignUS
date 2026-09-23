import math 
import random
import numpy as np


class MaskSampler:
    """
    Wrapper for the dataset to apply a mask to the images.

    Args:
        dataset: The dataset to wrap.
        patch_size: The patch size.
        pred_ratio: The ratio of the masked area. For example, setting pred_ratio to 0.3 will mask 30% of the image.
            Setting to 0 will disable masking.
        pred_ratio_var: The variance of the masked area. 
        pred_aspect_ratio: The aspect ratio of the masked area. 
        pred_shape: The shape of the masked area. Can be "block" or "rand".
        pred_start_epoch: The epoch to start the masked area. Before this epoch, the masked area will be 0.
        meanfill_masked_pixel: Whether to meanfill the masked pixel.
    """

    def __init__(
        self,
        patch_size=16,
        mask_ratio=0.3,
        mask_ratio_var=0.1,
        mask_aspect_ratio=(0.3, 1 / 0.3),
        mask_shape="block",
        mask_start_epoch=0,
        image_key="image",
        meanfill_masked_pixel=False
    ):
        self.image_key = image_key
        self.psz = patch_size
        self.pred_ratio = (
            mask_ratio[0]
            if isinstance(mask_ratio, list) and len(mask_ratio) == 1
            else mask_ratio
        )
        self.pred_ratio_var = (
            mask_ratio_var[0]
            if isinstance(mask_ratio_var, list) and len(mask_ratio_var) == 1
            else mask_ratio_var
        )
        if isinstance(self.pred_ratio, list) and not isinstance(
            self.pred_ratio_var, list
        ):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), mask_aspect_ratio))
        self.pred_shape = mask_shape
        self.pred_start_epoch = mask_start_epoch
        self.meanfill_masked_pixel = meanfill_masked_pixel

    def get_pred_ratio(self):
        if hasattr(self, "epoch") and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = (
                random.uniform(
                    self.pred_ratio - self.pred_ratio_var,
                    self.pred_ratio + self.pred_ratio_var,
                )
                if self.pred_ratio_var > 0
                else self.pred_ratio
            )

        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __call__(self, sample):
        masks = []
        images = sample[self.image_key]
        if not isinstance(images, list):
            images = [images]

        for img in images:
            try:
                H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
            except:
                # skip non-image
                continue

            high = self.get_pred_ratio() * H * W

            if self.pred_shape == "block":
                # following BEiT (https://arxiv.org/abs/2106.08254), see at
                # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[top : top + h, left : left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if mask[i, j] == 0:
                                            mask[i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta

            elif self.pred_shape == "rand":
                mask = np.hstack(
                    [
                        np.zeros(H * W - int(high)),
                        np.ones(int(high)),
                    ]
                ).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)

            else:
                # no implementation
                assert False

            masks.append(mask)

        sample['masks'] = masks[0] if len(masks) == 1 else masks
        return sample
