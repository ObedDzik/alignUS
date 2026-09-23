from torch.utils.data import Dataset
from enum import StrEnum
import typing as tp
from logging import getLogger
from tqdm import tqdm 
from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np 
from skimage.transform import resize
from PIL import Image 
from torch.utils.data import Dataset


logger = getLogger(__name__)


class RawDataAccessor: 
    def __init__(self, raw_data_dir, prostate_mask_dir=None):
        self.dir = raw_data_dir 
        self.metadata = pd.read_csv(os.path.join(self.dir, 'metadata.csv'), index_col=0)
        self.prostate_mask_dir = prostate_mask_dir

    def grade(self, core_id): 
        return self.metadata[self.metadata['core_id'] == core_id].iloc[0]['grade']       
    
    @property
    def core_ids(self): 
        return self.metadata.core_id.to_list()

    def metadata_for_core_id(self, core_id):
        return self.metadata[self.metadata['core_id'] == core_id].iloc[0].to_dict()

    def rf(self, core_id): 
        return np.load(
            os.path.join(self.dir, 'rf', self._get_fname(core_id)),
            mmap_mode='r'
        )
    
    def bmode(self, core_id): 
        return np.load(
            os.path.join(self.dir, 'bmode', self._get_fname(core_id)),
            mmap_mode='r',
        )
    
    def needle_mask(self, core_id): 
        raw_mask = np.load(
            os.path.join(self.dir, 'needle_mask.npy')
        )
        return resize(raw_mask, (512, 512), order=0)

    def prostate_mask(self, core_id): 
        if self.prostate_mask_dir is None: 
            raise ValueError("Prostate mask directory not provided")

        path = os.path.join(self.prostate_mask_dir, f"{core_id}.png")
        if not os.path.exists(path): 
            raise ValueError(f"Prostate mask not available for for {core_id}")

        return (np.array(Image.open(path)) / 255).astype('uint8')

    def _get_fname(self, core_id): 
        return f"{core_id}_{self.grade(core_id)}.npy"


class DataKeys(StrEnum):
    PATH_ON_SERVER = "path_on_server"
    CENTER = "center"
    LOC = "loc"
    GRADE = "grade"
    AGE = "age"
    FAMILY_HISTORY = "family_history"
    PSA = "psa"
    PCT_CANCER = "pct_cancer"
    PRIMARY_GRADE = "primary_grade"
    SECONDARY_GRADE = "secondary_grade"
    PATIENT_ID = "patient_id"
    CORE_ID = "core_id"
    ALL_CORES_BENIGN = "all_cores_benign"
    GRADE_GROUP = "grade_group"
    CLINICALLY_SIGNIFICANT = "clinically_significant"
    APPROX_PSA_DENSITY = "approx_psa_density"
    BMODE = "bmode"
    RF = "rf"
    PROSTATE_MASK = "prostate_mask"
    NEEDLE_MASK = "needle_mask"
    FRAME_IDX = "frame_idx"

    def is_metadata(self):
        return self in {
            DataKeys.CENTER,
            DataKeys.LOC,
            DataKeys.GRADE,
            DataKeys.AGE,
            DataKeys.FAMILY_HISTORY,
            DataKeys.PSA,
            DataKeys.PCT_CANCER,
            DataKeys.PRIMARY_GRADE,
            DataKeys.SECONDARY_GRADE,
            DataKeys.PATIENT_ID,
            DataKeys.CORE_ID,
            DataKeys.ALL_CORES_BENIGN,
            DataKeys.GRADE_GROUP,
            DataKeys.CLINICALLY_SIGNIFICANT,
            DataKeys.APPROX_PSA_DENSITY,
        }


class NCT2013Dataset(Dataset):
    def __init__(
        self,
        data_path: str,
        prostate_mask_dir: tp.Optional[str] = None,
        core_ids=None,
        items: tp.List[DataKeys] = [DataKeys.BMODE, DataKeys.CORE_ID],
        transform=None,
        frames: tp.Literal["first", "all"] = "first",
        flip_ud: bool = False,
        apply_colormap: bool = False,
    ):
        self.flip_ud = flip_ud
        self.apply_colormap = apply_colormap

        self.data_accessor = RawDataAccessor(raw_data_dir=data_path, prostate_mask_dir=prostate_mask_dir)

        if core_ids is None:
            core_ids = self.data_accessor.metadata.core_id.unique().tolist()

        self.metadata = self.data_accessor.metadata.copy()
        self.metadata = self.metadata[self.metadata.core_id.isin(core_ids)]
        self.transform = transform
        self.items = [DataKeys(item) for item in items]

        self._indices = []
        self.core_ids = core_ids
        for i, core_id in enumerate(tqdm(self.core_ids, desc="Indexing image frames")):
            if frames == "first":
                self._indices.append((i, 0))
            else: 
                raise NotImplementedError()

    def __len__(self):
        return len(self._indices)
    
    def __getitem__(self, idx): 
        core_idx, frame_idx = self._indices[idx]
        core_id = self.core_ids[core_idx]

        out = {}
        if DataKeys.BMODE in self.items:
            out[DataKeys.BMODE] = self.data_accessor.bmode(core_id)[..., 0]
        if DataKeys.RF in self.items:
            out[DataKeys.RF] = self.data_accessor.rf(core_id)[..., 0]
        if DataKeys.PROSTATE_MASK in self.items:
            out[DataKeys.PROSTATE_MASK] = self.data_accessor.prostate_mask(core_id)
        if DataKeys.NEEDLE_MASK in self.items:
            out[DataKeys.NEEDLE_MASK] = self.data_accessor.needle_mask(core_id)
        if DataKeys.FRAME_IDX in self.items:  
            out[DataKeys.FRAME_IDX] = frame_idx

        metadata = self.metadata[self.metadata.core_id == core_id].iloc[0].to_dict()
        for item in self.items:
            if item.is_metadata():
                out[item] = metadata[item]

        if self.flip_ud: 
            out[DataKeys.BMODE] = np.flipud(out[DataKeys.BMODE]).copy()
            if DataKeys.NEEDLE_MASK in out: 
                out[DataKeys.NEEDLE_MASK] = np.flipud(out[DataKeys.NEEDLE_MASK]).copy()
            if DataKeys.PROSTATE_MASK in out: 
                out[DataKeys.PROSTATE_MASK] = np.flipud(out[DataKeys.PROSTATE_MASK]).copy()
        
        if self.apply_colormap: 
            from .utils import apply_colormap
            out[DataKeys.BMODE] = apply_colormap(out[DataKeys.BMODE])

        if self.transform:
            out = self.transform(out)

        return out
