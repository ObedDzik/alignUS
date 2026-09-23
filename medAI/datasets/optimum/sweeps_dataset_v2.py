from pathlib import Path
from medAI.datasets.registry import register_dataset
from torch.utils.data import Dataset
from glob import glob
import os
import json


class SweepsSingleFramePNGDataset(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        split_id=None,
        split=None,
        case_ids=None,
        out_fmt="png",
        limit_samples=None,
        torchvision_api=False,
        splits_file=None,
    ):
        self.root = root
        self.out_fmt = out_fmt
        self.torchvision_api = torchvision_api

        if split is not None:
            splits_file = splits_file or os.path.join(root, "splits.json")
            if not os.path.exists(splits_file):
                raise FileNotFoundError(f"Splits file not found: {splits_file}")
            with open(splits_file, "r") as f:
                splits_data = json.load(f)
            if split_id is None:
                split_id = list(splits_data.keys())[0]
            if split_id not in splits_data:
                raise ValueError(
                    f"Split ID '{split_id}' not found in splits file. Available IDs: {list(splits_data.keys())}"
                )
            if split not in splits_data[split_id]:
                raise ValueError(
                    f"Split '{split}' not found in split ID '{split_id}'. Available splits: {list(splits_data[split_id].keys())}"
                )
            case_ids = splits_data[split_id][split]

        self.case_ids = case_ids
        self.split = split
        self.split_id = split_id
        self.png_paths = []
        all_case_ids = os.listdir(root)
        for case_id in all_case_ids:
            if case_ids and case_id not in case_ids:
                continue
            self.png_paths.extend(
                glob(os.path.join(f"{root}/{case_id}/**/*.png"), recursive=True)
            )

        self.transform = transform
        self.limit_samples = limit_samples
        if self.limit_samples is not None:
            self.png_paths = self.png_paths[: self.limit_samples]

    def __len__(self):
        return len(self.png_paths)

    def __getitem__(self, idx):
        from PIL import Image
        import numpy as np

        path = self.png_paths[idx]
        image = Image.open(path)
        if self.out_fmt == "np":
            image = np.array(image)
        json_info_path = str(Path(path).parent.parent / "info.json")
        with open(json_info_path) as f:
            info = json.load(f)

        if self.torchvision_api:
            label = 0
            if self.transform:
                image = self.transform(image)
            return image, label

        else:
            data = {
                "image": image,
                "path": path,
                "info": info,
            }

            if self.transform:
                data = self.transform(data)

            return data

    def __repr__(self):
        return f"SweepsSingleFramePNGDataset(root={self.root}, split_id={self.split_id}, split={self.split}, num_samples={len(self)})"
