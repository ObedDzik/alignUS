from torch.utils.data import Dataset
from enum import StrEnum
import typing as tp
from logging import getLogger
from .data_access import data_accessor
from tqdm import tqdm 
logger = getLogger(__name__)


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


class NCT2013Dataset:
    def __init__(
        self,
        core_ids=None,
        items: tp.List[DataKeys] = [DataKeys.BMODE, DataKeys.CORE_ID],
        transform=None,
        frames: tp.Literal["first", "all"] = "first",
    ):
        if core_ids is None:
            core_ids = data_accessor.get_metadata_table().core_id.unique().tolist()

        self.metadata = data_accessor.get_metadata_table().copy()
        self.metadata = self.metadata[self.metadata.core_id.isin(core_ids)]
        self.transform = transform
        self.items = [DataKeys(item) for item in items]

        self._indices = []
        self.core_ids = core_ids
        for i, core_id in enumerate(tqdm(self.core_ids, desc="Indexing image frames")):
            if frames == "first":
                self._indices.append((i, 0))
            elif frames == "all":
                n = data_accessor.get_num_frames(core_id)
                self._indices.extend([(i, j) for j in range(n)])

        

    def __len__(self):
        return len(self._indices)
    
    def __getitem__(self, idx): 
        core_idx, frame_idx = self._indices[idx]
        core_id = self.core_ids[core_idx]

        out = {}
        if DataKeys.BMODE in self.items:
            out[DataKeys.BMODE] = data_accessor.get_bmode_image(core_id, frame_idx)
        if DataKeys.RF in self.items:
            out[DataKeys.RF] = data_accessor.get_rf_image(core_id, frame_idx)
        if DataKeys.PROSTATE_MASK in self.items:
            out[DataKeys.PROSTATE_MASK] = data_accessor.get_prostate_mask(core_id)
        if DataKeys.NEEDLE_MASK in self.items:
            out[DataKeys.NEEDLE_MASK] = data_accessor.get_needle_mask(core_id)
        if DataKeys.FRAME_IDX in self.items:  
            out[DataKeys.FRAME_IDX] = frame_idx

        metadata = self.metadata[self.metadata.core_id == core_id].iloc[0].to_dict()
        for item in self.items:
            if item.is_metadata():
                out[item] = metadata[item]

        if self.transform:
            out = self.transform(out)

        return out



class BModeDatasetV1(Dataset):

    """Dataset for B-mode images.

    Samples are dictionaries with the following keys:
        bmode_frames (np.ndarray): Shape (H, W) uint8
        prostate_mask (np.ndarray): Shape (H, W)
        needle_mask (np.ndarray): Shape (H, W)
        core_id (str): Core id of the sample.
        frame_idx (int): Frame index of the sample.
        psa (float): PSA value of the sample.
        primary_grade (int): Primary grade of the sample.
        secondary_grade (int): Secondary grade of the sample.
        age (int): Age of the sample.
        family_history (int): Family history of the sample.
        ... etc (other metadata)

    Args:
        core_ids (list): List of core ids to include in the dataset.
        transform (callable): Transform to apply to each sample.
        frames (str): If 'first', only the first frame is returned. If 'all', all frames are returned.
    """

    def __init__(
        self,
        core_ids,
        transform=None,
        frames: tp.Literal["first", "all"] = "first",
        include_rf=False,
        rf_as_bmode=False,
    ):
        self.metadata = data_accessor.get_metadata_table().copy()
        self.metadata = self.metadata[self.metadata.core_id.isin(core_ids)]

        self._indices = []
        self.core_ids = core_ids
        for i, core_id in enumerate(tqdm(self.core_ids, desc="Loading dataset")):
            if frames == "first":
                self._indices.append((i, 0))
            elif frames == "all":
                n = data_accessor.get_num_frames(core_id)
                self._indices.extend([(i, j) for j in range(n)])

        self.transform = transform
        self.include_rf = include_rf
        self.rf_as_bmode = rf_as_bmode

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        core_idx, frame_idx = self._indices[idx]
        core_id = self.core_ids[core_idx]

        if not self.rf_as_bmode:
            bmode = data_accessor.get_bmode_image(core_id, frame_idx)
        else:
            bmode = data_accessor.get_rf_image(core_id, frame_idx)

        prostate_mask = data_accessor.get_prostate_mask(core_id)
        needle_mask = data_accessor.get_needle_mask(core_id)

        metadata = self.metadata[self.metadata.core_id == core_id].iloc[0].to_dict()
        metadata["frame_idx"] = frame_idx

        output = {
            "bmode": bmode,
            "prostate_mask": prostate_mask,
            "needle_mask": needle_mask,
            **metadata,
        }

        if self.include_rf:
            rf = data_accessor.get_rf_image(core_id, frame_idx)
            output["rf"] = rf

        if self.transform:
            output = self.transform(output)

        return output
