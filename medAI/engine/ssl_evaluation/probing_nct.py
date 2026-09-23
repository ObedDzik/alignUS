import json
import logging
import os
import torch
import numpy as np
from torch import nn
import torch.distributed
from tqdm import tqdm
from torchvision.transforms import v2
from torchvision.tv_tensors import Image, Mask
from torch import distributed as dist

from medAI.datasets.nct2013.cohort_selection import get_core_ids
from medAI.metrics import calculate_binary_classification_metrics
from medAI.datasets.nct2013.nct2013_legacy import NCT2013Dataset, DataKeys


_logger = logging.getLogger("NCT Probing")


class NCTProbing:
    def __init__(
        self,
        train_patients_fpath,
        val_patients_fpath,
        batch_size=8,
        image_size=512,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ):

        raw_data_dir = os.environ["NCT_RAW_DATA_DIR"]
        prostate_masks_dir = (
            "/ssd005/projects/exactvu_pca/nct_segmentations_medsam_finetuned_2023-11-10"
        )

        with open(train_patients_fpath) as f:
            train_patient_ids = json.load(f)
        with open(val_patients_fpath) as f:
            val_patient_ids = json.load(f)
        train_core_ids = get_core_ids(train_patient_ids)
        val_core_ids = get_core_ids(val_patient_ids)

        data_kw = dict(
            data_path=raw_data_dir,
            prostate_mask_dir=prostate_masks_dir,
            items=[
                DataKeys.BMODE,
                DataKeys.NEEDLE_MASK,
                DataKeys.PROSTATE_MASK,
                DataKeys.GRADE_GROUP,
                DataKeys.PCT_CANCER,
                DataKeys.PATIENT_ID,
                DataKeys.CORE_ID,
            ],
            transform=TransformNCT(
                size=image_size,
                mean=mean,
                std=std,
            ),
        )

        self.train_ds = NCT2013Dataset(core_ids=train_core_ids, **data_kw)
        self.val_ds = NCT2013Dataset(core_ids=val_core_ids, **data_kw)

        def _get_sampler(ds):
            if torch.distributed.is_initialized():
                sampler = torch.utils.data.DistributedSampler(ds)
            else:
                sampler = None
            return sampler

        self.train_loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=batch_size,
            num_workers=4,
            sampler=_get_sampler(self.train_ds),
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=batch_size,
            num_workers=4,
            sampler=_get_sampler(self.val_ds),
        )

    def run_probing(
        self,
        model: nn.Module,
        epoch,
        device,
        get_feature_map_fn=lambda model, x: model(x),
        is_main_process=True,
    ):
        return _run_nct_probing(
            model,
            self.train_loader,
            self.val_loader,
            epoch,
            device,
            get_feature_map_fn,
            is_main_process,
        )


@torch.no_grad()
def _run_nct_probing(
    model: nn.Module,
    train_loader,
    val_loader,
    epoch: int,
    device: str,
    get_feature_map_fn=lambda model, x: model(x),
    is_main_process=True,
):
    model.eval()

    train_outputs = _extract_all_data(model, train_loader, device, get_feature_map_fn)
    val_outputs = _extract_all_data(model, val_loader, device, get_feature_map_fn)

    if not is_main_process:
        return

    from sklearn.linear_model import LogisticRegression

    # ======== Linear probing at patch level ===================

    X_train = train_outputs["features"]
    y_train = train_outputs["labels"]
    inv_train = train_outputs["involvement"]

    # for training, exclude low involvement cancer cores
    mask = (y_train == 1) & (inv_train < 40)
    X_train = X_train[~mask]
    y_train = y_train[~mask]

    clf = LogisticRegression(
        random_state=0, max_iter=10000, class_weight="balanced"
    ).fit(X_train, y_train)
    y_hat_train = clf.predict_proba(X_train)[:, 1]

    X_val = val_outputs["features"]
    y_val = val_outputs["labels"]
    y_hat_val = clf.predict_proba(X_val)[:, 1]

    train_metrics = calculate_binary_classification_metrics(y_hat_train, y_train)
    val_metrics = calculate_binary_classification_metrics(y_hat_val, y_val)

    return train_metrics, val_metrics


@torch.no_grad()
def _extract_all_data(
    model: nn.Module, loader, device, output_adapter=None,
):

    outputs = {}

    model.eval()

    # Collect all the labels and masks from training and validation
    def extract_features(batch, model: nn.Module, device):
        bmode = batch["bmode"].to(device)

        image_features = model(bmode)
        if isinstance(image_features, dict):
            image_features = image_features["feature_map"]

        if output_adapter is not None:
            image_features = output_adapter(image_features)

        needle_mask = batch["needle_mask"].to(device)
        needle_mask = nn.functional.interpolate(
            needle_mask,
            size=(image_features.shape[2], image_features.shape[3]),
            mode="nearest",
        )
        prostate_mask = batch["prostate_mask"].to(device)
        prostate_mask = nn.functional.interpolate(
            prostate_mask,
            size=(image_features.shape[2], image_features.shape[3]),
            mode="nearest",
        )

        mask = (needle_mask > 0.5) & (prostate_mask > 0.5)
        mask = mask[:, 0, :, :]

        batch_idx = torch.arange(len(bmode), device=device)
        batch_idx = batch_idx[:, None, None, None].repeat(
            1, 1, mask.shape[1], mask.shape[2]
        )

        from einops import rearrange

        image_features = rearrange(image_features, "b c h w -> b h w c")
        batch_idx = rearrange(batch_idx, "b c h w -> b h w c")

        image_features = image_features[mask]
        batch_idx = batch_idx[mask].view(-1).cpu().numpy()

        def map_batch_level_feature_to_patch_level_feature(
            batch_level_features, batch_idx
        ):
            if isinstance(batch_level_features, torch.Tensor):
                batch_level_features = batch_level_features.view(-1).tolist()
            out = []
            for i in range(len(batch_idx)):
                out.append(batch_level_features[batch_idx[i]])
            return np.array(out)

        cancer = batch["cancer"]
        cancer = map_batch_level_feature_to_patch_level_feature(
            cancer.view(-1), batch_idx
        )
        involvement = batch["involvement"]
        involvement = map_batch_level_feature_to_patch_level_feature(
            involvement.view(-1), batch_idx
        )
        grade_group = batch["grade_group"]
        grade_group = map_batch_level_feature_to_patch_level_feature(
            grade_group, batch_idx
        )
        core_id = batch["core_id"]
        core_id = map_batch_level_feature_to_patch_level_feature(core_id, batch_idx)
        patient_id = batch["patient_id"]
        patient_id = map_batch_level_feature_to_patch_level_feature(
            patient_id, batch_idx
        )

        return image_features, cancer, involvement, grade_group, core_id, patient_id

    for batch in tqdm(loader, desc="Extracting features"):

        image_features, cancer, involvement, grade_group, core_id, patient_id = (
            extract_features(batch, model, device)
        )
        outputs.setdefault("features", []).append(image_features)
        # outputs.setdefault("patch_prediction", []).append(patch_prediction)
        outputs.setdefault("labels", []).append(cancer)
        outputs.setdefault("involvement", []).append(involvement)
        outputs.setdefault("grade_group", []).append(grade_group)
        outputs.setdefault("core_id", []).append(core_id)
        outputs.setdefault("patient_id", []).append(patient_id)

    outputs["features"] = torch.cat(outputs["features"], dim=0).cpu().numpy()
    # outputs["patch_prediction"] = torch.cat(outputs["patch_prediction"], dim=0).cpu().numpy()
    outputs["labels"] = np.concatenate(outputs["labels"]).reshape(-1)
    outputs["involvement"] = np.concatenate(outputs["involvement"]).reshape(-1)
    outputs["grade_group"] = np.concatenate(outputs["grade_group"]).reshape(-1)
    outputs["core_id"] = np.concatenate(outputs["core_id"]).reshape(-1)
    outputs["patient_id"] = np.concatenate(outputs["patient_id"]).reshape(-1)

    def gather_across_processes(array):
        if not dist.is_initialized():
            return array
        output_list = [np.zeros_like(array) for _ in range(dist.get_world_size())]
        dist.all_gather_object(output_list, array)
        return np.concatenate(output_list)

    for key in outputs:
        outputs[key] = gather_across_processes(outputs[key])

    return outputs


class TransformNCT:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=512):
        self.mean = mean
        self.std = std
        self.size = size

    def __call__(self, sample):
        bmode = sample["bmode"]
        needle_mask = sample["needle_mask"]
        prostate_mask = sample["prostate_mask"]
        pct_cancer = sample["pct_cancer"]
        grade_group = sample["grade_group"]

        bmode = torch.from_numpy(bmode).float() / 255
        bmode = bmode[None, ...].repeat_interleave(3, 0)
        prostate_mask = torch.from_numpy(prostate_mask)[None, ...].float()
        needle_mask = torch.from_numpy(needle_mask)[None, ...].float()

        bmode = Image(bmode)
        needle_mask = Mask(needle_mask)
        prostate_mask = Mask(prostate_mask)

        bmode, needle_mask, prostate_mask = v2.Resize(
            (self.size, self.size), interpolation=3
        )(bmode, needle_mask, prostate_mask)
        # bmode, needle_mask, prostate_mask = v2.CenterCrop((self.size))(bmode, needle_mask, prostate_mask)
        bmode = v2.Normalize(self.mean, self.std)(bmode)

        pct_cancer = torch.tensor(pct_cancer).float()
        grade_group = torch.tensor(grade_group).long()
        cancer = (grade_group > 0).long()

        output = {}
        output["bmode"] = bmode
        output["needle_mask"] = needle_mask
        output["prostate_mask"] = prostate_mask
        output["grade_group"] = grade_group
        output["cancer"] = cancer
        output["involvement"] = pct_cancer
        output["core_id"] = sample["core_id"]
        output["patient_id"] = sample["patient_id"]

        return output


def tokens_to_feature_map(tokens, cls_is_included=True):
    if cls_is_included:
        # discard cls token
        tokens = tokens[:, 1:, :]

    B, N, D = tokens.shape
    H = W = int(N**0.5)
    assert H * W == N

    tokens = tokens.view(B, H, W, D)
    tokens = tokens.permute(0, 3, 1, 2)

    return tokens


if __name__ == "__main__":
    probe = KFoldNCTProbing()
    item = probe.train_loader.dataset[0]

    import matplotlib.pyplot as plt

    plt.imshow(item["bmode"].permute(1, 2, 0))
    plt.savefig("test.png")
