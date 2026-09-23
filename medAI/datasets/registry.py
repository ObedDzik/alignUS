import logging


_DATASETS = {}


def register_dataset(f):
    _DATASETS[f.__name__] = f
    return f


def create_dataset(*, name, split='train', transform=None, **kwargs):
    if name not in _DATASETS:
        raise ValueError(
            f"Unknown dataset: {name}. Available datasets: {list_datasets()}"
        )
    return _DATASETS[name](split=split, transform=transform, **kwargs)


def list_datasets():
    return list(_DATASETS.keys())


