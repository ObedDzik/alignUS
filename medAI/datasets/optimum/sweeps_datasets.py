import os
from medAI.datasets.generic.h5_dataset import H5SingleFrameDataset
from medAI.datasets.registry import register_dataset


OPTIMUM_28MM_SWEEPS_ROOT = os.getenv(
    "OPTIMUM_28MM_SWEEPS_ROOT",
)


@register_dataset
def optimum_28mm_sweeps_frames(split='train', splits_id=None, root_dir=None, transform=None, **kwargs):
    root_dir = root_dir or OPTIMUM_28MM_SWEEPS_ROOT
    if not root_dir:
        raise ValueError(
            "Please set the environment variable OPTIMUM_28MM_SWEEPS_ROOT to the path of the OPTIMUM 28mm sweeps dataset."
        )
    from .splits import get_splits, _SPLITS_FNS
    splits_id = splits_id or list(_SPLITS_FNS)[0]
    splits = get_splits(splits_id)
    if split not in splits:
        raise ValueError(f"Split '{split}' not found in splits '{splits_id}'")
    case_ids = splits[split]

    dataset = H5SingleFrameDataset(
        root_dir=root_dir,
        case_ids=case_ids,
        transform=transform,
        **kwargs
    )

    return dataset