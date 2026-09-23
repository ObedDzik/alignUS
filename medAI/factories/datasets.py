from medAI.registry import register
from medAI.datasets.optimum.sweeps_dataset_v2 import SweepsSingleFramePNGDataset
from medAI.datasets.optimum.needle_trace_dataset import NeedleTraceImageFramesDataset


@register("dataset", "optimum_sweeps_single_frame")
def build_optimum_sweeps_dataset(
    *, transform=None, root=None, split: str = "train", **kwargs
):
    """
    Build the Optimum Sweeps dataset.

    Args:
        root_dir (str): Root directory of the dataset.
        split (str): Dataset split, e.g., 'train', 'val', 'test'.
        **kwargs: Additional keyword arguments for the dataset.

    Returns:
        Dataset: An instance of the Optimum Sweeps dataset.
    """

    if root is None: 
        from medAI.global_config import OPTIMUM_SWEEPS_DATASET_PATH
        root = OPTIMUM_SWEEPS_DATASET_PATH

    return SweepsSingleFramePNGDataset(
        root=root, split=split, transform=transform, **kwargs
    )


@register("dataset", "optimum_biopsies_single_frame")
def build_optimum_biopsies_dataset(
    *, transform=None, root: str, split: str = "train", **kwargs
):
    """
    Build the Optimum Biopsies dataset.

    Args:
        root_dir (str): Root directory of the dataset.
        split (str): Dataset split, e.g., 'train', 'val', 'test'.
        **kwargs: Additional keyword arguments for the dataset.

    Returns:
        Dataset: An instance of the Optimum Biopsies dataset.
    """
    return NeedleTraceImageFramesDataset(
        root_dir=root, split=split, transform=transform, **kwargs
    )
