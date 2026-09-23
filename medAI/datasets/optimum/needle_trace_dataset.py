import os
import json
from typing import Literal
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class NeedleTraceImageFramesIndexDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split=None,
        split_id=None,
        case_ids=None,
        cine_ids=None,
        needle_mask_fname="needle_mask.png",
        splits_file=None
    ):
        self.root_dir = root_dir

        if split is not None:
            splits_file = splits_file or os.path.join(root_dir, "splits.json")
            if not os.path.exists(splits_file):
                raise FileNotFoundError(f"Splits file not found: {splits_file}")
            with open(splits_file, "r") as f:
                splits_data = json.load(f)
            if split_id is None:
                split_id = list(splits_data.keys())[0]
            if split_id not in splits_data:
                raise ValueError(f"Split ID '{split_id}' not found in splits file.")
            if split not in splits_data[split_id]:
                raise ValueError(f"Split '{split}' not found in split ID '{split_id}'.")
            case_ids = splits_data[split_id][split]

        self.case_ids = case_ids
        self.cine_ids = cine_ids
        self.data = []

        # Traverse the directory structure to collect all cases and cine IDs
        for case_dir in os.listdir(root_dir):
            case_path = os.path.join(root_dir, case_dir)
            if not os.path.isdir(case_path):
                continue

            # Filter by case_ids if provided
            if case_ids and case_dir not in case_ids:
                continue

            for cine_id_dir in os.listdir(case_path):
                cine_path = os.path.join(case_path, cine_id_dir)
                if not os.path.isdir(cine_path):
                    continue

                # Filter by cine_ids if provided
                if cine_ids and cine_id_dir not in cine_ids:
                    continue

                image_path = os.path.join(cine_path, "image.png")
                needle_mask_path = os.path.join(cine_path, needle_mask_fname)
                info_path = os.path.join(cine_path, "info.json")

                if (
                    os.path.exists(image_path)
                    and os.path.exists(needle_mask_path)
                    and os.path.exists(info_path)
                ):

                    # Load metadata
                    with open(info_path, "r") as f:
                        info = json.load(f)

                    self.data.append(
                        {
                            "image_path": image_path,
                            "needle_mask_path": needle_mask_path,
                            "info_path": info_path,
                            "microsegnet_prostate_mask_path": os.path.join(
                                cine_path, "micro_seg_net_prostate_mask.png"
                            ),
                            "info": info,
                            "core_id": info["cine_id"],
                        }
                    )

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Loads and returns a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.
        """
        return self.data[idx]


class NeedleTraceImageFramesDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split=None,
        split_id=None,
        case_ids=None,
        cine_ids=None,
        transform=None,
        needle_mask_fname="needle_mask.png",
        out_fmt: Literal["pil", "np"] = "pil",
    ):
        """
        PyTorch Dataset for loading needle annotation data.

        Args:
            root_dir (str): Path to the root directory containing exported data.
            case_ids (list, optional): List of case IDs to include. Defaults to None.
            cine_ids (list, optional): List of cine IDs to include. Defaults to None.
        """
        self.root_dir = root_dir

        if split is not None:
            splits_file = os.path.join(root_dir, "splits.json")
            if not os.path.exists(splits_file):
                raise FileNotFoundError(f"Splits file not found: {splits_file}")
            with open(splits_file, "r") as f:
                splits_data = json.load(f)
            if split_id is None:
                split_id = list(splits_data.keys())[0]
            if split_id not in splits_data:
                raise ValueError(f"Split ID '{split_id}' not found in splits file.")
            if split not in splits_data[split_id]:
                raise ValueError(f"Split '{split}' not found in split ID '{split_id}'.")
            case_ids = splits_data[split_id][split]

        self.case_ids = case_ids
        self.cine_ids = cine_ids
        self.transform = transform
        self.out_fmt = out_fmt
        self.data = []

        # Traverse the directory structure to collect all cases and cine IDs
        for case_dir in os.listdir(root_dir):
            case_path = os.path.join(root_dir, case_dir)
            if not os.path.isdir(case_path):
                continue

            # Filter by case_ids if provided
            if case_ids and case_dir not in case_ids:
                continue

            for cine_id_dir in os.listdir(case_path):
                cine_path = os.path.join(case_path, cine_id_dir)
                if not os.path.isdir(cine_path):
                    continue

                # Filter by cine_ids if provided
                if cine_ids and cine_id_dir not in cine_ids:
                    continue

                image_path = os.path.join(cine_path, "image.png")
                needle_mask_path = os.path.join(cine_path, needle_mask_fname)
                info_path = os.path.join(cine_path, "info.json")

                if (
                    os.path.exists(image_path)
                    and os.path.exists(needle_mask_path)
                    and os.path.exists(info_path)
                ):

                    # Load metadata
                    with open(info_path, "r") as f:
                        info = json.load(f)

                    self.data.append(
                        {
                            "image_path": image_path,
                            "needle_mask_path": needle_mask_path,
                            "info_path": info_path,
                            "microsegnet_prostate_mask_path": os.path.join(
                                cine_path, "micro_seg_net_prostate_mask.png"
                            ),
                            "info": info,
                        }
                    )

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Loads and returns a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the image, needle mask, and metadata.
        """
        sample = self.data[idx]

        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")
        if self.out_fmt == "np":
            image = np.array(image)

        # Load needle mask
        needle_mask = Image.open(sample["needle_mask_path"]).convert("L")
        if self.out_fmt == "np":
            needle_mask = np.array(needle_mask)

        # Load metadata
        info = sample["info"]

        out = {
            "image": image,
            "needle_mask": needle_mask,
            "path": sample["image_path"],
            "info": info,
        }
        if os.path.exists(sample["microsegnet_prostate_mask_path"]):
            microsegnet_prostate_mask = Image.open(
                sample["microsegnet_prostate_mask_path"]
            ).convert("L")
            if self.out_fmt == "np":
                microsegnet_prostate_mask = np.array(microsegnet_prostate_mask)
            out["microsegnet_prostate_mask"] = microsegnet_prostate_mask

        # Apply transformations if specified
        if self.transform:
            out = self.transform(out)

        return out

    def list_indices_by_patient_ids(self):
        # get all patient ids:

        outputs = {}

        for index, sample in enumerate(self.data):
            info = sample["info"]
            case_id = info["case"]
            core_id = info["cine_id"]

            outputs.setdefault(case_id, []).append((index, core_id))

        for case_id, index_info in outputs.items():
            indices_sorted = [
                index_info_i[0]
                for index_info_i in sorted(index_info, key=lambda item: item[1])
            ]
            outputs[case_id] = indices_sorted

        return outputs

if __name__ == "__main__":
    # Initialize the dataset with specific case_ids or cine_ids
    dataset = NeedleTraceImageFramesDataset(
        root_dir="/h/pwilson/projects/medAI/data/OPTIMUM/processed/UA_annotated_needles"
    )
    print(len(dataset))

    # Access a sample
    sample = dataset[0]
    print(sample["info"])  # Print metadata
    print(sample["image"].size)  # Print image size
    print(sample["needle_mask"].size)  # Print needle mask size
    # Display the sample
    sample["image"].show()  # Display the image
    sample["needle_mask"].show()  # Display the needle mask
