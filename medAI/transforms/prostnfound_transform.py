import numpy as np
import torch
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import InterpolationMode
from torchvision.tv_tensors import Image, Mask
from medAI.transforms.crop_to_mask import CropToMask
from medAI.transforms.pixel_augmentations import RandomContrast, RandomGamma

psa_min = 0.2
psa_max = 32.95
psa_avg = 6.821426488456866
age_min = 0
age_max = 79
age_avg = 62.5816
approx_psa_density_min = 4.615739672282483e-06
approx_psa_density_max = 0.000837278201784
approx_psa_density_avg = 0.000175347951594383


CORE_LOCATION_TO_IDX = {
    "LML": 0,
    "RBL": 1,
    "LMM": 2,
    "RMM": 2,
    "LBL": 1,
    "LAM": 3,
    "RAM": 3,
    "RML": 0,
    "LBM": 4,
    "RAL": 5,
    "RBM": 4,
    "LAL": 5,
}


class ProstNFoundTransform:
    def __init__(
        self,
        augment="none",
        image_size=1024,
        mask_size=256,
        mean=(0, 0, 0),
        std=(1, 1, 1),
        crop_to_prostate=False, 
        first_downsample_size=None,
        return_raw_images=False,
        grade_group_for_positive_label=1,
        flip_ud=False,
        pu_extra_flip=False,
    ):
        self.augmentations = augment
        self.image_size = image_size
        self.mask_size = mask_size
        self.mean = mean
        self.std = std
        self.crop_to_prostate = crop_to_prostate
        self.first_downsample_size = first_downsample_size
        self.return_raw_images = return_raw_images
        self.grade_group_for_positive_label = grade_group_for_positive_label
        self.flip_ud = flip_ud
        # Default False -- does not change behavior for any existing caller.
        # `_ProstNFoundDatasetAdapterOptimum`'s unconditional "probe_top" flip
        # below is correct for UA/OL (transrectal) but wrong for PU
        # (transperineal, opposite raw orientation on disk -- see
        # DATASET.md "Orientation/flip convention" and the same
        # `pu_extra_flip` mechanism already used by
        # projects/primus_vlm/dataset.py and projects/primus_joint). Set
        # True only when PU cores are actually included in the data (e.g.
        # projects/prostnfound/cfg/finetune_optimum_noprompt_pu.yaml).
        self.pu_extra_flip = pu_extra_flip

    def _coerce_input(self, item):
        if 'image' in item:
            # this is from the OPTMUM needle dataset format. We need to convert it to the standard format.
            item = _ProstNFoundDatasetAdapterOptimum()(item)
            if self.pu_extra_flip and item.get("center") == "PU":
                item = dict(item)
                item["bmode"] = np.flipud(item["bmode"]).copy()
                item["needle_mask"] = np.flipud(item["needle_mask"]).copy()
                item["prostate_mask"] = np.flipud(item["prostate_mask"]).copy()
        return item

    def __call__(self, item):
        item = self._coerce_input(item)
        out = item.copy()

        bmode = item["bmode"]
        needle_mask = item["needle_mask"] if "needle_mask" in item else np.zeros((224, 224), np.uint8)
        prostate_mask = item["prostate_mask"] if 'prostate_mask' in item else np.ones((224, 224), np.uint8)
        # print(np.unique(prostate_mask))

        if self.flip_ud:
            bmode = np.flipud(bmode).copy()
            needle_mask = np.flipud(needle_mask).copy()
            prostate_mask = np.flipud(prostate_mask).copy()

        if self.return_raw_images:
            out['bmode_raw'] = bmode.copy()
            out['needle_mask_raw'] = needle_mask.copy() 
            out['prostate_mask_raw'] = prostate_mask.copy()

        bmode = torch.from_numpy(bmode.copy()).float()
        bmode = bmode.unsqueeze(0)
        bmode = bmode.repeat(3, 1, 1)
        bmode = (bmode - bmode.min()) / (bmode.max() - bmode.min())
        bmode = T.Normalize(self.mean, self.std)(bmode)
        bmode = Image(bmode)

        needle_mask = needle_mask = torch.from_numpy(needle_mask.copy()).float()
        needle_mask = needle_mask.unsqueeze(0)
        needle_mask = Mask(needle_mask)

        prostate_mask = prostate_mask = torch.from_numpy(prostate_mask.copy()).float()
        prostate_mask = prostate_mask.unsqueeze(0)
        prostate_mask = Mask(prostate_mask)

        bmode, needle_mask, prostate_mask = T.Resize((1024, 1024), antialias=True)(bmode, needle_mask, prostate_mask)

        if self.crop_to_prostate: 
            bmode, needle_mask, prostate_mask = CropToMask('prostate_mask', 16)(
                dict(
                    bmode=bmode,
                    needle_mask=needle_mask, 
                    prostate_mask=prostate_mask, 
                )
            ).values()

        if "translate" in self.augmentations: 
            bmode, needle_mask, prostate_mask = T.RandomAffine([0, 0], [0.2, 0.2])(
                bmode, needle_mask, prostate_mask
            )

        if self.first_downsample_size is not None: 
            bmode, needle_mask, prostate_mask = T.Resize(
                (self.first_downsample_size, self.first_downsample_size), 
            )([bmode, needle_mask, prostate_mask])

        if "random_crop" in self.augmentations and (torch.rand((1,)).item() > 0.5):
            bmode, needle_mask, prostate_mask = T.RandomResizedCrop(
                (self.image_size, self.image_size),
                scale=(0.3, 1),
            )(bmode, needle_mask, prostate_mask)
        else: 
            bmode = T.Resize((self.image_size, self.image_size))(bmode)

        if "gamma" in self.augmentations: 
            bmode = T.RandomApply([RandomGamma((0.6, 3))])(bmode)

        if "contrast" in self.augmentations: 
            bmode = T.RandomApply([RandomContrast((0.7, 2))])(bmode)

        # interpolate the masks to the mask size
        needle_mask = T.Resize(
            (self.mask_size, self.mask_size),
            antialias=False,
            interpolation=InterpolationMode.NEAREST,
        )(needle_mask)
        prostate_mask = T.Resize(
            (self.mask_size, self.mask_size),
            antialias=False,
            interpolation=InterpolationMode.NEAREST,
        )(prostate_mask)

        # images
        out["bmode"] = bmode
        out["needle_mask"] = needle_mask
        out["prostate_mask"] = prostate_mask
        # print(np.unique(out['prostate_mask']))

        # labels
        # if "grade" in item:
        #     out["label"] = torch.tensor(item["grade"] != "Benign").long()
        
        if "grade_group" in item:
            out["label"] = torch.tensor(item['grade_group'] >= self.grade_group_for_positive_label).long()
        
        if "pct_cancer" in item: 
            pct_cancer = item["pct_cancer"]
            if np.isnan(pct_cancer):
                pct_cancer = 0

            out["involvement"] = torch.tensor(pct_cancer / 100).float()

        # handle prompts
        if "psa" in item:
            psa = item["psa"]
            if np.isnan(psa):
                psa = psa_avg
            psa = (psa - psa_min) / (psa_max - psa_min)
            out["psa"] = torch.tensor([psa]).float()

        if "age" in item:
            age = item["age"]
            if np.isnan(age):
                age = age_avg
            age = (age - age_min) / (age_max - age_min)
            out["age"] = torch.tensor([age]).float()

        if "approx_psa_density" in item:
            approx_psa_density = item["approx_psa_density"]
            if np.isnan(approx_psa_density):
                approx_psa_density = approx_psa_density_avg
            approx_psa_density = (approx_psa_density - approx_psa_density_min) / (
                approx_psa_density_max - approx_psa_density_min
            )
            out["approx_psa_density"] = torch.tensor([approx_psa_density]).float()

        if "family_history" in item: 
            if item["family_history"] is True:
                out["family_history"] = torch.tensor(1.)
            elif item["family_history"] is False:
                out["family_history"] = torch.tensor(-1.)
            elif np.isnan(item["family_history"]):
                out["family_history"] = torch.tensor(0.)

        # misc
        if "center" in item:
            out["center"] = item["center"]
        
        if "all_cores_benign" in item: 
            out["all_cores_benign"] = torch.tensor(item["all_cores_benign"]).bool()
        
        if "grade" in item:
            out["grade"] = item["grade"]
        
        if "core_id" in item:
            out["core_id"] = item["core_id"]

        return out


class _ProstNFoundDatasetAdapterOptimum:
    def __init__(self, image_orientation="probe_top"):
        self.image_orientation = image_orientation

    def __call__(self, item):
        """
        Adapter function to convert the Optimum dataset items to the format expected by prostnfound.
        """

        bmode = item["image"]
        bmode = np.array(bmode)[..., 0]
        needle_mask = np.array(item["needle_mask"])
        # prostate_mask = np.ones_like(needle_mask)
        try:
            prostate_mask = np.array(item['prostate_mask'])
        except:
            prostate_mask = np.ones_like(needle_mask)

        if self.image_orientation == "probe_top":
            bmode = np.flipud(bmode)  # Flip the image vertically
            needle_mask = np.flipud(needle_mask)  # Flip the mask vertically
            # if prostate_mask:
            prostate_mask = np.flipud(prostate_mask)  # Flip the mask vertically

        info = item["info"]

        # {'cine_id': 'UA-023-006',
        # 'center': 'UA',
        # 'case': 'UA-023',
        # 'isBiopsy': True,
        # 'isFusionVu': True,
        # 'hasFusionVuOverlay': True, 'bimg_path': '/Users/pwilson/Library/CloudStorage/Box-Box/OPTIMUM Blinded Database/UA/UA-023/UA-023/20220502075606670/20220502075606702/20220502081332512/336714191398189585739976459255897944.bimg', 'cine_number': 6.0, 'length': 60.0, 'heightMm': 28.0, 'heightPixel': 962.0, 'widthMm': 46.08, 'widthPixel': 1372.0, 'createdDate': '2022-05-02T14:39:35.0Z', 'Sample ID': 'RLM', 'Cine #': '6', 'PRI-MUS': '2', 'PI-RADS': 2.0, 'Notes': nan, 'Number of Samples': 1.0, 'Core Length': '14', 'Diagnosis': 'Benign', 'GG': nan, '% Cancer': nan, '% P4': nan, 'P Inv': False, 'IDC': False, 'Cribiform': False, 'Inflamm': False, 'HGPIN': False, 'Atypia': False, 'PATH Notes': nan, 'PATH ID': 'RLM', 'case_path': 'UA-023', 'age': 53.0, 'psa': 5.0, 'study_arm': 'Arm3-MicroUS-MRI'}

        return {
            "bmode": bmode,
            "needle_mask": needle_mask,
            "prostate_mask": prostate_mask,
            "grade": info.get("Diagnosis", "Unknown"),
            "pct_cancer": info.get("% Cancer", 0.0),
            "psa": info.get("psa", 0.0),
            "age": info.get("age", 0.0),
            "approx_psa_density": info.get("approx_psa_density", 0.0),
            "family_history": info.get("family_history", np.nan),
            "center": info.get("center", "Unknown"),
            "all_cores_benign": info.get("all_cores_benign", False),
            "core_id": info.get("cine_id", "Unknown"),
            "patient_id": info["case"],
            "loc": info["Sample ID"],
            "grade_group": info["GG"],
            "clinically_significant": info.get("clinically_significant", False),
        }