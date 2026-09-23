from dataclasses import dataclass
import os
from typing_extensions import deprecated
from tqdm import tqdm
import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from medAI.utils.data.patch_extraction import PatchView
from .cohort_selection import (
    CohortSelectionOptions,
    select_cohort, 
)
from torchvision.transforms import v2 as T
from torchvision import tv_tensors
import numpy as np
from typing import Literal
import torch
from torchvision.transforms import v2 as T
from torchvision import tv_tensors
import matplotlib.pyplot as plt
from ...transforms import get_pixel_level_augmentations_by_mode
from abc import ABC, abstractmethod
import ctypes
import multiprocessing as mp


VECTOR_CLUSTER_PROCESSED_DATA_ROOT = (
    "/fs01/projects/exactvu_pca/nct2013/processed/h5_archive_2024-07-16_flipped"
)
VECTOR_CLUSTER_METADATA_PATH = "/ssd005/projects/exactvu_pca/nct2013/processed/h5_archive_2024-07-16_flipped/metadata.csv"


@dataclass
class NCT2013PatchesDatasetConf:
    patch_height_mm: float = 5
    patch_width_mm: float = 5
    stride_height_mm: float = 1
    stride_width_mm: float = 1
    patch_source: str = "rf"


@dataclass
class NCT2013FullImageWithPatchesDatasetConf(NCT2013PatchesDatasetConf):
    image_source: str = "bmode"


class NCT2013StorageBackend(ABC):
    @abstractmethod
    def rf(self, core_id): ...
    @abstractmethod
    def bmode(self, core_id): ...
    @abstractmethod
    def needle_mask(self, core_id): ...
    @abstractmethod
    def prostate_mask(self, core_id): ...


class HDF5StorageBackend(NCT2013StorageBackend):
    _RF_KEY = "rf_image"
    _BMODE_KEY = "bmode_image"
    _PROSTATE_MASK_KEY = "prostate_mask"
    _NEEDLE_MASK_KEY = "needle_mask"

    def __init__(self, filepath, keep_h5_file_open=True, **_):
        self.filepath = filepath
        self.keep_h5_file_open = keep_h5_file_open

        if self.keep_h5_file_open:
            self._data_file = h5py.File(self.filepath, "r")
        else:
            self._data_file = None

    def _get_data(self, core_id, data_key):
        if self.keep_h5_file_open:
            return self._data_file[f"{core_id}/{data_key}"]
        else:
            with h5py.File(self.filepath, "r") as f:
                return f[f"{core_id}/{data_key}"]

    def rf(self, core_id):
        return self._get_data(core_id, self._RF_KEY)[:]

    def bmode(self, core_id):
        return self._get_data(core_id, self._BMODE_KEY)[:]

    def needle_mask(self, core_id):
        return self._get_data(core_id, self._NEEDLE_MASK_KEY)[:]

    def prostate_mask(self, core_id):
        return self._get_data(core_id, self._PROSTATE_MASK_KEY)[:]

    def __del__(self):
        if self.keep_h5_file_open:
            self._data_file.close()


class ArrayStorageBackend(NCT2013StorageBackend):
    _RF_KEY = "rf_image"
    _BMODE_KEY = "bmode_image"
    _PROSTATE_MASK_KEY = "prostate_mask"
    _NEEDLE_MASK_KEY = "needle_mask"

    def __init__(self, filepath, core_ids, **_):
        print(f"Loading data for {len(core_ids)} cores from {filepath}")
        with h5py.File(filepath, "r") as _data_file:
            _data_arrays = {}
            for key in [
                self._RF_KEY,
                self._BMODE_KEY,
                self._PROSTATE_MASK_KEY,
                self._NEEDLE_MASK_KEY,
            ]:

                def _compute_most_common_shape(key):
                    sizes = {}
                    for core_id in core_ids:
                        im = _data_file[core_id][key]
                        sizes.setdefault(tuple(im.shape), 0)
                        sizes[tuple(im.shape)] += 1
                    most_common_size = sorted(
                        list(sizes.keys()), key=lambda k: sizes[k]
                    )[-1]
                    return most_common_size

                shape = _compute_most_common_shape(key)
                array_shape = [len(core_ids)] + list(shape)
                _data_arrays[key] = self._make_array(array_shape)

                print(
                    f"Loading {key} data - requires {_data_arrays[key].nbytes / 2 ** 20:.0f}MB"
                )
                for core_id in tqdm(
                    core_ids, desc="Loading data from h5 file into shared array"
                ):
                    im = _data_file[core_id][key][:]
                    if im.shape != shape: 
                        from skimage.transform import resize
                        im = resize(im, shape)
                    _data_arrays[key][core_ids.index(core_id)] = im
        
        self._data_arrays = _data_arrays
        self.core_ids = core_ids

    def _make_array(self, shape):
        array = np.zeros(shape, dtype=np.float16)
        return array

    def _read_view_from_array(self, core_id, key):
        return self._data_arrays[key][self.core_ids.index(core_id)]

    def bmode(self, core_id):
        return self._read_view_from_array(core_id, self._BMODE_KEY)

    def rf(self, core_id):
        return self._read_view_from_array(core_id, self._RF_KEY)

    def prostate_mask(self, core_id):
        return self._read_view_from_array(core_id, self._PROSTATE_MASK_KEY)

    def needle_mask(self, core_id):
        return self._read_view_from_array(core_id, self._NEEDLE_MASK_KEY)


class SharedArrayStorageBackend(ArrayStorageBackend):
    def _make_array(self, shape):
        shared_array_base = mp.Array(ctypes.c_float, int(np.prod(shape)), lock=False)
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array = shared_array.reshape(shape)
        return shared_array


class TensorStorageBackend(ArrayStorageBackend): 
    def _make_array(self, shape):
        return torch.zeros(shape, dtype=torch.float16)


def _get_backend(
    root: str, backend: str = "h5", **kwargs
) -> NCT2013StorageBackend:
    if backend == "h5":
        return HDF5StorageBackend(
            os.path.join(root, "data.h5"), **kwargs
        )
    elif backend == "shared_array":
        return SharedArrayStorageBackend(os.path.join(root, "data.h5"), **kwargs)
    elif backend == "array":
        return ArrayStorageBackend(os.path.join(root, "data.h5"), **kwargs)
    elif backend == "tensor":
        return TensorStorageBackend(os.path.join(root, "data.h5"), **kwargs)


class _NCT2013BaseDataset(Dataset):
    IMAGE_HEIGHT_MM = 28
    IMAGE_WIDTH_MM = 46.06

    def __init__(
        self,
        root: str,
        core_ids=None,
        cohort_selection_options=None,
        split=None,
        backend="h5",
        backend_kwargs={},
        debug=False, 
    ):
        self.root = root
        self.metadata = pd.read_csv(os.path.join(root, "metadata.csv"), index_col=0)
        if core_ids is None:
            if cohort_selection_options is None:
                cohort_selection_options = CohortSelectionOptions(fold=0, n_folds=5)
                print(
                    f"Using default cohort selection options. {cohort_selection_options}"
                )
            if split is None:
                split = "train"
                print(f"Using default split: {split}")

            self.cohort = select_cohort(cohort_selection_options)
            if split == "train":
                core_ids = self.cohort[0]
            elif split == "val":
                core_ids = self.cohort[1]
            elif split == "test":
                core_ids = self.cohort[2]
            elif split == "ssl_train":
                core_ids = self.cohort[3]
            else:
                raise ValueError(f"Unknown split: {split}")

        self.core_ids = sorted(core_ids)
        if debug:
            print("DEBUG MODE: Using only 10 cores.")
            import random 
            sample = random.sample(self.core_ids, 100)
            self.core_ids = sample

        print(f"Instantiating {type(self)} with {len(self.core_ids)} cores.")

        self.backend = _get_backend(root, backend, **{'core_ids': self.core_ids, **backend_kwargs})

        self._loaded_data = {}
        self._data_loaded = False

    def load_data(self, core_ids):
        rf_images = []
        bmode_images = []
        # core_ids = []
        prostate_masks = []
        needle_masks = []

        # loading images, masks, metadata
        for core_id in tqdm(core_ids, desc="Loading images"):

            rf_images.append(self.rf(core_id))
            bmode_images.append(self.bmode(core_id))
            needle_mask = self.needle_mask(core_id)
            prostate_mask = self.prostate_mask(core_id)
            prostate_masks.append(prostate_mask)
            needle_masks.append(needle_mask)

        print(f"Loaded {len(core_ids)} images, masks, and metadata.")
        # if self.keep_data_in_memory:
        #     print(f"Storing data in memory - this could take a while!")
        #     rfsize = rf_images[0].size * rf_images[0].itemsize * len(rf_images)
        #     bmsize = bmode_images[0].size * bmode_images[0].itemsize * len(bmode_images)
        #
        #     rfsizes = {}
        #     for rf in rf_images:
        #         rfsizes.setdefault(tuple(rf.shape), 0)
        #         rfsizes[tuple(rf.shape)] += 1
        #     most_common_size = sorted(list(rfsizes.keys()), key=lambda k: rfsizes[k])[-1]
        #     print(f"Most common rf size: {most_common_size}")
        #     print(f"Reshaping remaining images")
        #     reshaped_images = []
        #     for rf in tqdm(rf_images):
        #         if rf.shape != most_common_size:
        #             from skimage.transform import resize
        #             rf = resize(rf, most_common_size)
        #         reshaped_images.append(rf)
        #     rf_images = reshaped_images
        #
        #     print(f"RF requirement: {rfsize / 2 ** 20}MB")
        #     print(f"Bmode requirement: {bmsize / 2 ** 20}MB")
        #
        #     start = time.time()
        #     self._loaded_data['rf'] = np.stack(rf_images)
        #     self._loaded_data['bmode'] = np.stack(bmode_images)
        #     self._loaded_data['prostate_mask'] = np.stack(prostate_masks)
        #     self._loaded_data['needle_mask'] = np.stack(needle_masks)
        #     print(f"Data loading took {time.time() - start} seconds.")
        #     self._data_loaded = True
        #     self.data_file.close()
        #
        return core_ids, rf_images, bmode_images, prostate_masks, needle_masks

    def rf(self, core_id):
        return self.backend.rf(core_id)

    def bmode(self, core_id):
        return self.backend.bmode(core_id)

    def needle_mask(self, core_id):
        return self.backend.needle_mask(core_id)

    def prostate_mask(self, core_id):
        return self.backend.prostate_mask(core_id)

    def metadata_for_core_id(self, core_id):
        return self.metadata.loc[self.metadata.core_id == core_id].iloc[0].to_dict()


class NCT2013PatchesDataset(_NCT2013BaseDataset):
    DATA_KEY_PATCH = "patch"
    DATA_KEY_METADATA = "metadata"
    DATA_KEY_CANCER_LABEL = "cancer_label"

    def __init__(
        self,
        root: str,
        conf: NCT2013PatchesDatasetConf = NCT2013PatchesDatasetConf(),
        transform=None,
        **kwargs,
    ):

        super().__init__(root, **kwargs)
        self.conf = conf
        self.transform = transform
        self.patch_views = {}

        self.setup_patch_views(self.core_ids)
        self._indices = []
        for i, core_id in enumerate(self.core_ids):
            view = self.patch_views[core_id]
            for j in range(len(view)):
                self._indices.append((i, j))

    def setup_patch_views(self, core_ids):
        print(f"Setting up patch views for {len(core_ids)} cores from scratch.")
        patch_height_px, patch_width_px, stride_height_px, stride_width_px = (
            self.compute_patch_sizes()
        )
        core_ids, rf_images, bmode_images, prostate_masks, needle_masks = (
            self.load_data(core_ids)
        )
        patch_images = rf_images if self.conf.patch_source == "rf" else bmode_images

        patch_views = PatchView.build_collection_from_images_and_masks(
            patch_images,
            (patch_height_px, patch_width_px),
            (stride_height_px, stride_width_px),
            "topright",
            mask_lists=[needle_masks, prostate_masks],
            thresholds=[0.66, 0.9],
        )

        for core_id, view in zip(core_ids, patch_views):
            self.patch_views[core_id] = view

    def compute_patch_sizes(self):
        # Computing patch sizes in pixels
        core_id = self.core_ids[0]

        if self.conf.patch_source == "rf":
            patch_ref = self.rf(core_id)
        else:
            patch_ref = self.bmode(core_id)[..., -1]

        height_px = patch_ref.shape[0]
        width_px = patch_ref.shape[1]

        height_mm = self.IMAGE_HEIGHT_MM
        width_mm = self.IMAGE_WIDTH_MM

        patch_height_px = int(self.conf.patch_height_mm / height_mm * height_px)
        patch_width_px = int(self.conf.patch_width_mm / width_mm * width_px)
        stride_height_px = int(self.conf.stride_height_mm / height_mm * height_px)
        stride_width_px = int(self.conf.stride_width_mm / width_mm * width_px)

        print(f"stride (pixels): {stride_height_px}, {stride_width_px}")
        print(f"height (pixels): {patch_height_px}, width: {patch_width_px}")

        return patch_height_px, patch_width_px, stride_height_px, stride_width_px

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        i, j = self._indices[idx]
        core_id = self.core_ids[i]
        patch_view = self.patch_views[core_id]
        patch = patch_view[j]
        metadata_row = self.metadata_for_core_id(core_id)

        item = {
            self.DATA_KEY_PATCH: patch,
            self.DATA_KEY_METADATA: metadata_row,
            self.DATA_KEY_CANCER_LABEL: metadata_row["grade"] != "Benign",
        }

        if self.transform:
            item = self.transform(item)

        return item


class NCT2013FullImageWithPatchesDataset(NCT2013PatchesDataset):
    DATA_KEY_PATCH = "patch"
    DATA_KEY_PROSTATE_MASK = "prostate_mask"
    DATA_KEY_NEEDLE_MASK = "needle_mask"
    DATA_KEY_IMAGE = "image"
    DATA_KEY_METADATA = "metadata"
    DATA_KEY_CANCER_LABEL = "cancer_label"
    DATA_KEY_PATCH_POSITION_HWHW_RELATIVE = "patch_position"
    DATA_KEY_PATCH_POSITION_XYXY = "patch_position_xyxy"

    def __init__(
        self,
        root: str,
        conf: NCT2013FullImageWithPatchesDatasetConf = NCT2013FullImageWithPatchesDatasetConf(),
        transform=None,
        **kwargs,
    ):
        super().__init__(root, conf, **kwargs)
        self.conf = conf
        self.transform = transform

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index):
        i, j = self._indices[index]
        core_id = self.core_ids[i]
        patch_view = self.patch_views[core_id]
        patch = patch_view[j]
        position = patch_view.positions[j]
        prostate_mask = self.prostate_mask(core_id)
        needle_mask = self.needle_mask(core_id)
        metadata_row = self.metadata_for_core_id(core_id)
        if self.conf.image_source == "rf":
            image = self.rf(core_id)
        else:
            image = self.bmode(core_id)

        # make position be relative to the image size rather than absolute
        hmin, wmin, hmax, wmax = position
        h, w = patch_view.image.shape
        position = (hmin / h, wmin / w, hmax / h, wmax / w)
        position = np.array(position)

        # also add position in xyxy format relative to main image (bounding box style)
        hmin, wmin, hmax, wmax = position
        h, w = image.shape
        position_xyxy = (int(wmin * w), int(hmin * h), int(wmax * w), int(hmax * h))

        item = {
            self.DATA_KEY_PATCH: patch,
            self.DATA_KEY_PROSTATE_MASK: prostate_mask,
            self.DATA_KEY_NEEDLE_MASK: needle_mask,
            self.DATA_KEY_IMAGE: image,
            self.DATA_KEY_METADATA: metadata_row,
            self.DATA_KEY_CANCER_LABEL: metadata_row["grade"] != "Benign",
            self.DATA_KEY_PATCH_POSITION_HWHW_RELATIVE: position,
            self.DATA_KEY_PATCH_POSITION_XYXY: position_xyxy,
        }

        if self.transform:
            item = self.transform(item)

        return item

    def show_item(self, idx):
        _tmp_transform = self.transform
        self.transform = None
        item = self[idx]
        self.transform = _tmp_transform

        patch = item[self.DATA_KEY_PATCH]
        prostate_mask = item[self.DATA_KEY_PROSTATE_MASK]
        needle_mask = item[self.DATA_KEY_NEEDLE_MASK]
        image = item[self.DATA_KEY_IMAGE]
        metadata = item[self.DATA_KEY_METADATA]
        cancer_label = item[self.DATA_KEY_CANCER_LABEL]
        position = item[self.DATA_KEY_PATCH_POSITION_HWHW_RELATIVE]

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        IMG_EXTENT = (0, self.IMAGE_WIDTH_MM, self.IMAGE_HEIGHT_MM, 0)
        axs[0, 0].imshow(patch, cmap="gray", aspect="auto")
        axs[0, 0].set_title("Patch")
        axs[1, 1].imshow(prostate_mask, cmap="Purples", extent=IMG_EXTENT)
        axs[1, 1].imshow(needle_mask, cmap="Greens", extent=IMG_EXTENT, alpha=0.5)
        axs[1, 1].set_title("Masks")
        view = self.patch_views[metadata["core_id"]]
        view.show(axs[1, 0])
        axs[1, 0].set_title("Patch view")
        axs[0, 1].imshow(image, cmap="gray", extent=IMG_EXTENT)
        axs[0, 1].set_title("Image")

        # make box where patch came from
        hmin, wmin, hmax, wmax = position
        h, w = self.IMAGE_HEIGHT_MM, self.IMAGE_WIDTH_MM
        hmin, wmin, hmax, wmax = hmin * h, wmin * w, hmax * h, wmax * w
        rect = plt.Rectangle(
            (wmin, hmin), wmax - wmin, hmax - hmin, edgecolor="r", facecolor="none"
        )
        axs[1, 1].add_patch(rect)
        rect = plt.Rectangle(
            (wmin, hmin), wmax - wmin, hmax - hmin, edgecolor="r", facecolor="none"
        )
        axs[0, 1].add_patch(rect)

        plt.show()

    @staticmethod
    @deprecated("Use preprocess_raw_data_to_h5 script instead")
    def preprocess_raw_data_to_h5(
        raw_data_dir: str, prostate_mask_dir: str, output_dir: str
    ):
        raise DeprecationWarning("Use preprocess_raw_data_to_h5 script instead")
        os.makedirs(output_dir, exist_ok=True)

        accessor = RawDataAccessor(raw_data_dir, prostate_mask_dir)
        accessor.metadata.to_csv(os.path.join(output_dir, "metadata.csv"))
        core_ids = sorted(accessor.core_ids)

        with h5py.File(os.path.join(output_dir, "data.h5"), "w") as f:
            # loading images, masks, metadata
            for core_id in tqdm(core_ids, desc="Processing images"):
                rf_image = accessor.rf(core_id)[..., -1]
                bmode_image = accessor.bmode(core_id)[..., -1]
                needle_mask = accessor.needle_mask(core_id)[...]
                prostate_mask = accessor.prostate_mask(core_id)[...]

                f.create_dataset(f"{core_id}/rf_image", data=rf_image)
                f.create_dataset(f"{core_id}/bmode_image", data=bmode_image)
                f.create_dataset(f"{core_id}/needle_mask", data=needle_mask)
                f.create_dataset(f"{core_id}/prostate_mask", data=prostate_mask)


class NCT2013PatchTransform:
    def __init__(
        self,
        size=256,
        instance_norm=True,
        crop_size=None,
        random_crop=False,
        horizontal_flip=False,
    ):
        self.size = size
        self.instance_norm = instance_norm
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.horizontal_flip = horizontal_flip

    def __call__(self, patch):
        patch = torch.tensor(patch).float().unsqueeze(0)
        if self.instance_norm:
            patch = (patch - patch.mean()) / patch.std()
        patch = T.Resize((self.size, self.size))(patch)

        if self.crop_size is not None:
            if self.crop_size > self.size:
                raise ValueError(f"crop_size must be less than or equal to {self.size}")
            if self.random_crop:
                patch = T.RandomCrop((self.crop_size, self.crop_size))(patch)
            else:
                patch = T.CenterCrop((self.crop_size, self.crop_size))(patch)

        if self.horizontal_flip:
            patch = T.RandomHorizontalFlip()(patch)

        return patch


class NCT2013ImageAndMaskTransform:
    def __init__(
        self,
        image_size=256,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        random_translation: tuple[float, float] | None = None,
        random_resized_crop: tuple[int, int] | None = None,
        pixel_level_augmentations_mode: Literal["weak", "strong", "none"] = "none",
        return_masks: bool = False,
    ):
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.random_translation = random_translation
        self.random_resized_crop = random_resized_crop
        self.pixel_level_augmentations_mode = pixel_level_augmentations_mode
        self.pixel_level_augmentations = get_pixel_level_augmentations_by_mode(
            pixel_level_augmentations_mode
        )
        self.return_masks = return_masks

    def __call__(self, image, prostate_mask, needle_mask, box):
        image = torch.tensor(image).float().unsqueeze(0).repeat_interleave(3, dim=0)
        image = (image - image.min()) / (image.max() - image.min())

        if self.pixel_level_augmentations_mode != "none":
            image = T.ToDtype(torch.uint8, scale=True)(image)
            image = T.ToPILImage()(image)
            image = self.pixel_level_augmentations(image)
            image = T.ToImage()(image)
            image = T.ToDtype(torch.float32, scale=True)(image)

        image = T.Normalize(mean=self.mean, std=self.std)(image)

        image = tv_tensors.Image(image)
        prostate_mask = torch.tensor(prostate_mask).float().unsqueeze(0)
        prostate_mask = tv_tensors.Mask(prostate_mask)
        box = torch.tensor(box).unsqueeze(0).long()
        box = tv_tensors.BoundingBoxes(box, format="xyxy", canvas_size=image.shape[-2:])

        needle_mask = torch.tensor(needle_mask).float().unsqueeze(0)
        needle_mask = tv_tensors.Mask(needle_mask)

        H, W = image.shape[-2:]

        if self.return_masks:
            # resize masks to image shape
            prostate_mask = T.Resize((H, W))(prostate_mask)
            needle_mask = T.Resize((H, W))(needle_mask)

            # geometric transformations
            image, prostate_mask, needle_mask, box = T.Resize(
                (self.image_size, self.image_size)
            )(image, prostate_mask, needle_mask, box)

            if self.random_resized_crop:
                image, prostate_mask, needle_mask, box = T.RandomResizedCrop(
                    self.image_size, scale=self.random_resized_crop
                )(image, prostate_mask, needle_mask, box)

            if self.random_translation:
                image, prostate_mask, needle_mask, box = T.RandomAffine(
                    degrees=0, translate=self.random_translation
                )(image, prostate_mask, needle_mask, box)

            return image, prostate_mask, needle_mask, box
        else: 
            image, box = T.Resize((self.image_size, self.image_size))(image, box)
            if self.random_resized_crop:
                image, box = T.RandomResizedCrop(
                    self.image_size, scale=self.random_resized_crop
                )(image, box)

            if self.random_translation:
                image, box = T.RandomAffine(
                    degrees=0, translate=self.random_translation
                )(image, box)

            return image, box


class NCT2013FullTransform:
    def __init__(self, patch_transform, image_transform):
        self.patch_transform = patch_transform
        self.image_transform = image_transform
        self.image_transform.return_masks = False

    def __call__(self, item):
        img = item[NCT2013FullImageWithPatchesDataset.DATA_KEY_IMAGE]
        prostate_mask = item[NCT2013FullImageWithPatchesDataset.DATA_KEY_PROSTATE_MASK]
        needle_mask = item[NCT2013FullImageWithPatchesDataset.DATA_KEY_NEEDLE_MASK]

        metadata = item[NCT2013FullImageWithPatchesDataset.DATA_KEY_METADATA]
        patch = item[NCT2013FullImageWithPatchesDataset.DATA_KEY_PATCH]
        box = item[NCT2013FullImageWithPatchesDataset.DATA_KEY_PATCH_POSITION_XYXY]
        cancer_label = item[NCT2013FullImageWithPatchesDataset.DATA_KEY_CANCER_LABEL]

        img, box = self.image_transform(
            img, prostate_mask, needle_mask, box
        )

        patch = self.patch_transform(patch)
        cancer_label = torch.tensor(cancer_label).long()

        out = {
            NCT2013FullImageWithPatchesDataset.DATA_KEY_IMAGE: img,
            #NCT2013FullImageWithPatchesDataset.DATA_KEY_PROSTATE_MASK: prostate_mask,
            #NCT2013FullImageWithPatchesDataset.DATA_KEY_NEEDLE_MASK: needle_mask,
            NCT2013FullImageWithPatchesDataset.DATA_KEY_PATCH: patch,
            NCT2013FullImageWithPatchesDataset.DATA_KEY_PATCH_POSITION_XYXY: box,
            NCT2013FullImageWithPatchesDataset.DATA_KEY_CANCER_LABEL: cancer_label,
            NCT2013FullImageWithPatchesDataset.DATA_KEY_METADATA: metadata,
        }

        return out


def get_image_and_patch_transforms(
    image_size: int = 256, 
    patch_size: int = 256, 
    patch_crop_size: int = 224,
    instance_norm: bool = True,
    augmentations_mode: str = "none",
):
    patch_kw = {
        "size": patch_size,
        "random_crop": False,
        "horizontal_flip": False,
        "crop_size": patch_crop_size,
        "instance_norm": instance_norm,
    }
    image_kw = {
        "image_size": image_size,
    }
    patch_transform_val = NCT2013PatchTransform(**patch_kw)
    image_transform_val = NCT2013ImageAndMaskTransform(**image_kw)

    if augmentations_mode == "weak":
        patch_kw["random_crop"] = True
        patch_kw["horizontal_flip"] = True
        image_kw["random_translation"] = (0.1, 0.1)
        image_kw["random_resized_crop"] = (0.8, 1.0)
    elif augmentations_mode == "medium":
        patch_kw["random_crop"] = True
        patch_kw["horizontal_flip"] = True
        image_kw["random_translation"] = (0.1, 0.1)
        image_kw["random_resized_crop"] = (0.8, 1.0)
        image_kw["pixel_level_augmentations_mode"] = "weak"
    elif augmentations_mode == "strong":
        patch_kw["random_crop"] = True
        patch_kw["horizontal_flip"] = True
        image_kw["random_translation"] = (0.8, 0.1)
        image_kw["random_resized_crop"] = (0.8, 1.0)
        image_kw["pixel_level_augmentations_mode"] = "strong"

    patch_transform_train = NCT2013PatchTransform(**patch_kw)
    image_transform_train = NCT2013ImageAndMaskTransform(
        **image_kw
    )

    return NCT2013FullTransform(
        patch_transform_train, image_transform_train
    ), NCT2013FullTransform(patch_transform_val, image_transform_val)


if __name__ == "__main__":
    val_set = NCT2013FullImageWithPatchesDataset(
        VECTOR_CLUSTER_PROCESSED_DATA_ROOT,
        split='val', 
        backend='array', 
        backend_kwargs={'keep_h5_file_open': True}
    )
    for _ in tqdm(val_set): 
        pass

    val_set.transform = get_image_and_patch_transforms()[1]
    import cProfile
    cProfile.run('val_set[0]', sort='cumtime')
    # for _ in tqdm(val_set):
    #     pass

    breakpoint()
