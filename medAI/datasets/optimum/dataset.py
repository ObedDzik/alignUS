import json
import os
from pathlib import Path

import h5py
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset

from medAI.utils.data.padding import strip_padding
from tqdm import tqdm


class OptimumDataset(Dataset):
    def __init__(
        self,
        dir,
        mode="frames",
        strip_padding=True,
        transform=None,
        transforms=None,
        target_transform=None,
        format="h5",
        tqdm_kw={}
    ):
        self.dir = dir
        self.mode = mode
        self.strip_padding = strip_padding
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms
        self.format = format

        all_npy_paths = list(Path(self.dir).glob(f"**/*.{format}"))

        self.samples = []
        self.indices = []
        for case_idx, path in enumerate(tqdm(sorted(all_npy_paths), desc="OPTIMUM: Indexing data", **tqdm_kw)):
            path_rel = path.relative_to(self.dir)
            center, case, fname = str(path_rel).split(os.sep)

            json_path = path.parent / (str(path.name).split(".")[0] + ".json")
            with open(json_path) as f:
                vxml_info = json.load(f)
                length = int(vxml_info["length"])

            self.samples.append(
                dict(center=center, case=case, path=str(path), vxml_info=vxml_info)
            )

            if self.mode == "frames":
                self.indices.extend([(case_idx, j) for j in range(length)])
            else:
                self.indices.append((case_idx, None))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        case_idx, frame_idx = self.indices[idx]
        info = self.samples[case_idx]
        # mmap_arr = np.load(info["numpy_path"], mmap_mode="r")

        img_arr = self.load_raw_img_data(info["path"], frame=frame_idx)
        if self.strip_padding:
            img_arr = strip_padding(img_arr)
        if self.mode == "frames":
            img_data = Image.fromarray(img_arr).convert("RGB")
        else:
            img_data = img_arr

        if self.transform:
            img_data = self.transform(img_data)

        if self.target_transform:
            info = self.target_transform(info)

        if self.transforms:
            img_data, info = self.transforms(img_data, info)

        return img_data, info

    def load_raw_img_data(self, path, frame: int | None = None):
        if self.format == "h5":
            with h5py.File(path, "r") as f:
                data = f["img"]
                if frame is None:
                    return data[:]
                else:
                    return data[frame, ...]
        elif self.format == "zarr.zip":
            import zarr

            store = zarr.storage.ZipStore(path, read_only=True)
            z = zarr.open_array(store, mode="r")
            # read the data as a NumPy Array
            # z[:]
            if frame is None:
                return z[:]
            else:
                return z[frame]
        else:
            raise NotImplementedError(self.format)


if __name__ == "__main__":
    from argparse import ArgumentParser

    from torch.utils.data import DataLoader
    from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor
    from tqdm import tqdm

    parser = ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    loader = DataLoader(
        OptimumDataset(
            transform=Compose([Resize(256), CenterCrop(224), ToTensor()]),
            strip_padding=False,
            target_transform=lambda x: 0,
            format="h5",
        ),
        batch_size=64,
        shuffle=True,
        num_workers=args.num_workers,
    )

    for batch in tqdm(loader):
        ...
