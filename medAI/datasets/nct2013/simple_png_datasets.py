import json
import os
from typing import Callable, Literal
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class NCT2013PatchesDataset(Dataset):
    """Dataset which returns patches and metadata for the NCT2013 dataset."""

    def __init__(
        self,
        root: str | None = None,
        core_ids: list[str] | Literal["all"] = "all",
        transform: Callable | None = None,
    ):
        """Builds the dataset

        Args:
            root (str): Path to the root directory of the dataset - here, the patches are saved as '.png' files.
            core_ids (list[str] | Literal['all']): List of core ids to include in the dataset. If 'all', all core ids are included.
            transform (Callable | None): Transform to apply to the image and metadata. Defaults to None. If not None, the transform should take in a
                tuple[image: PIL.Image, metadata: dict] and return the transformed image and metadata.
        """

        self.root = root or os.environ['NCT_BMODE_PATCHES_PNG']

        self.transform = transform

        self.metadata_table = pd.read_csv(os.path.join(self.root, "metadata.csv"))

        core_ids = (
            core_ids
            if core_ids != "all"
            else list(self.metadata_table["core_id"].unique())
        )

        self.core_ids = core_ids
        self._indices = []
        for core_id in core_ids:
            for frame in range(
                len(
                    [
                        path
                        for path in os.listdir(os.path.join(self.root, core_id))
                        if path.endswith(".png")
                    ]
                )
            ):
                self._indices.append((core_id, frame))

    def _lookup_image_path(self, core_id, frame):
        return os.path.join(self.root, core_id, f"{core_id}_{frame}.png")

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        core_id, frame = self._indices[idx]
        metadata = (
            self.metadata_table[self.metadata_table["core_id"] == core_id]
            .iloc[0]
            .to_dict()
        )
        image_path = self._lookup_image_path(core_id, frame)
        image = Image.open(image_path)
        if self.transform is not None:
            return self.transform(image, metadata)
        return image, metadata


class NCT2013ImagesAndCancerMasksDataset(Dataset):
    def __init__(
        self,
        root: str | None = None,
        core_ids: list[str] | Literal["all"] = "all",
        transform: Callable | None = None,
    ):
        self.root = root or os.environ['NCT_BMODE_FULL_IMAGES']
        self.transform = transform

        if core_ids == "all":
            core_ids = os.listdir(self.root)

        self.core_ids = core_ids

    def __len__(self):
        return len(self.core_ids)

    def __getitem__(self, idx):
        core_id = self.core_ids[idx]
        img_path = os.path.join(self.root, core_id, "bmode.png")
        mask_path = os.path.join(self.root, core_id, "cancer_mask.png")
        metadata_path = os.path.join(self.root, core_id, "metadata.json")

        img = Image.open(img_path)
        mask = Image.open(mask_path)
        metadata = json.load(open(metadata_path))

        output = dict(
            image=img, 
            cancer_mask=mask, 
            metadata=metadata
        )

        if self.transform: 
            output = self.transform(output)

        return output
    