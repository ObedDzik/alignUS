from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from medAI.utils.data.patch_extraction import PatchView


class PatchesDatasetWrapper(Dataset):
    def __init__(
        self,
        dataset,
        patch_size_mm=(5, 5),
        patch_stride_mm=(1, 1),
        image_key="image",
        mask_keys=["needle_mask"],
        physical_height_getter=lambda item: 28,
        physical_width_getter=lambda item: 46.06,
        mask_thresholds=[0.25],
        transform=None,
        yield_one_patch_per_item=True,
        include_images=False,
    ):
        self.dataset = dataset
        self.patch_size_mm = patch_size_mm
        self.patch_stride_mm = patch_stride_mm
        self.mask_keys = mask_keys
        self.mask_thresholds = mask_thresholds
        self.image_key = image_key
        self.transform = transform
        self.yield_one_patch_per_item = yield_one_patch_per_item
        self.include_images = include_images

        self.data = []
        self.patches = []
        self.positions = []
        self.indices = []

        i = 0
        for item in tqdm(self.dataset, desc="Extracting patches"):  # type: ignore
            im = item[self.image_key]
            masks = [item[k] for k in self.mask_keys]
            physical_height = physical_height_getter(item)
            physical_width = physical_width_getter(item)

            pv = PatchView.from_sliding_window_physical_coordinate(
                im,
                (physical_height, physical_width),
                self.patch_size_mm,
                self.patch_stride_mm,
                masks=masks,
                thresholds=self.mask_thresholds,
            )

            n = len(pv)
            if n == 0:
                continue

            datum = item.copy()
            if not self.include_images:
                datum.pop(self.image_key, None)
                for k in self.mask_keys:
                    datum.pop(k, None)

            pv.set_fmt("XYXY")
            positions = pv.positions
            patches = [pv[i] for i in range(n)]

            self.data.append(datum)
            self.patches.append(patches)
            self.positions.append(positions)

            if self.yield_one_patch_per_item:
                self.indices.extend([(i, [j]) for j in range(n)])
            else:
                self.indices.append((i, list(range(n))))

            i += 1

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item_idx, patch_idxs = self.indices[idx]
        out = self.data[item_idx].copy()

        patches = self.patches[item_idx]
        positions = self.positions[item_idx]
        patches = [patches[i] for i in patch_idxs]
        positions = [positions[i] for i in patch_idxs]

        out["patches"] = patches
        out["patch_positions_xyxy"] = positions

        if self.transform is not None:
            out = self.transform(out)
        return out
