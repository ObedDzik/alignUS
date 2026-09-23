
import os
import pandas as pd


def _get_sheet(): 
    return pd.read_csv(
        os.path.join(
            os.path.dirname(__file__),
            'optimum_case_sheet.csv'
        ),
    )
    

_SPLITS_FNS = {}


def register_split(fn):
    """Register a split function."""
    _SPLITS_FNS[fn.__name__] = fn


def get_splits(name, *args, **kwargs) -> dict[str, list[str]]:
    """Get a registered split function by name."""
    if name not in _SPLITS_FNS:
        raise ValueError(f"Split '{name}' is not registered.")
    return _SPLITS_FNS[name](*args, **kwargs)


@register_split
def ua_train_val_v0():
    """Get the UA train/val split for fold 0."""
    sheet = _get_sheet()
    sheet = sheet.loc[sheet['center'] == 'UA']

    all_cases = sheet['case'].unique().tolist()
    from sklearn.model_selection import train_test_split
    train_cases, val_cases = train_test_split(
        all_cases,
        test_size=0.2,
        random_state=42,
    )

    return {
        'train': train_cases,
        'val': val_cases
    }