from argparse import ArgumentParser, BooleanOptionalAction

import logging

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from typing import Literal

from .data_access import data_accessor


def get_patient_splits_nested_kfold(fold=0, n_folds=5, splits_file=None, val_size=0.2):
    """returns the list of patient ids for the train, val, and test splits."""
    if splits_file is not None:
        logging.info(f"Using splits file {splits_file}")
        # we manually override the this function and use the csv file
        # because the original code uses a random seed to split the data
        # and we want to be able to reproduce the splits
        table = pd.read_csv(splits_file)
        train_ids = table[table[f"fold_{fold}"] == "train"].patient_id.values.tolist()
        val_ids = table[table[f"fold_{fold}"] == "val"].patient_id.values.tolist()
        test_ids = table[table[f"fold_{fold}"] == "test"].patient_id.values.tolist()
        return train_ids, val_ids, test_ids

    if fold not in range(n_folds):
        raise ValueError(f"Fold must be in range {n_folds}, but got {fold}")

    metadata_table = data_accessor.get_metadata_table()
    patient_table = metadata_table.drop_duplicates(subset=["patient_id"])
    patient_table = patient_table[["patient_id", "center"]]

    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    for i, (train_idx, test_idx) in enumerate(
        kfold.split(patient_table["patient_id"], patient_table["center"])
    ):
        if i == fold:
            train = patient_table.iloc[train_idx]
            test = patient_table.iloc[test_idx]
            break

    if val_size > 0:
        train, val = train_test_split(
            train, test_size=val_size, random_state=0, stratify=train["center"]
        )
    else:
        val = pd.DataFrame(columns=train.columns)

    train = train.patient_id.values.tolist()
    val = val.patient_id.values.tolist()
    test = test.patient_id.values.tolist()

    return train, val, test


def get_patient_splits_kfold(fold=0, n_folds=5):
    if fold not in range(n_folds):
        raise ValueError(f"Fold must be in range {n_folds}, but got {fold}")

    metadata_table = data_accessor.get_metadata_table()
    patient_table = metadata_table.drop_duplicates(subset=["patient_id"])
    patient_table = patient_table[["patient_id", "center"]]

    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    for i, (train_idx, test_idx) in enumerate(
        kfold.split(patient_table["patient_id"], patient_table["center"])
    ):
        if i == fold:
            train = patient_table.iloc[train_idx]
            val = patient_table.iloc[test_idx]
            break

    test = pd.DataFrame(columns=train.columns)

    train = train.patient_id.values.tolist()
    val = val.patient_id.values.tolist()
    test = test.patient_id.values.tolist()

    return train, val, test


def get_patient_splits_by_center(leave_out="UVA", val_size=0.2, val_seed=0):
    """returns the list of patient ids for the train, val, and test splits."""
    if leave_out not in ["UVA", "CRCEO", "PCC", "PMCC", "JH"]:
        raise ValueError(
            f"leave_out must be one of 'UVA', 'CRCEO', 'PCC', 'PMCC', 'JH', but got {leave_out}"
        )

    metadata_table = data_accessor.get_metadata_table()
    patient_table = metadata_table.drop_duplicates(subset=["patient_id"])
    table = patient_table[["patient_id", "center"]]

    train = table[table.center != leave_out]
    train, val = train_test_split(
        train, test_size=val_size, random_state=val_seed, stratify=train["center"]
    )
    train = train.patient_id.values.tolist()
    val = val.patient_id.values.tolist()
    test = table[table.center == leave_out].patient_id.values.tolist()

    return train, val, test


def get_patient_splits(test_center="UVA", validation_fold=0, n_folds=5):
    """returns the list of patient ids for the train, val, and test splits."""
    if test_center not in ["UVA", "CRCEO", "PCC", "PMCC", "JH"]:
        raise ValueError(
            f"test_center must be one of 'UVA', 'CRCEO', 'PCC', 'PMCC', 'JH', but got {test_center}"
        )

    metadata_table = data_accessor.get_metadata_table()
    patient_table = metadata_table.drop_duplicates(subset=["patient_id"])
    table = patient_table[["patient_id", "center"]]

    test = table[table.center == test_center].patient_id.values.tolist()
    train_val = table[table.center != test_center]

    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    for i, (train_idx, val_idx) in enumerate(
        kfold.split(train_val["patient_id"], train_val["center"])
    ):
        if i == validation_fold:
            train = train_val.iloc[train_idx].patient_id.values.tolist()
            val = train_val.iloc[val_idx].patient_id.values.tolist()
            break

    return train, val, test


def get_patient_splits_by_mode(
    test_center=None,
    fold=None,
    n_folds=None,
    val_size=0.2,
    val_seed=0,
    mode: Literal["kfold", "nested_kfold", "center", "train_only", None] = None,
):
    if test_center is not None:
        logging.info(f"Using test center {test_center}")
        train, val, test = get_patient_splits_by_center(
            leave_out=test_center, val_size=val_size, val_seed=val_seed
        )
    elif mode == "train_only":
        train, val, test = get_patient_splits_kfold()
        train = train + val + test
        val = []
        test = []
    elif mode == "kfold":
        assert fold is not None, "Must specify fold if mode is 'kfold'."
        assert n_folds is not None, "Must specify n_folds if fold is specified."
        train, val, test = get_patient_splits_kfold(fold=fold, n_folds=n_folds)
    elif mode == "nested_kfold":
        assert fold is not None, "Must specify fold if mode is 'nested_kfold'."
        assert n_folds is not None, "Must specify n_folds if fold is specified."
        train, val, test = get_patient_splits_nested_kfold(
            fold=fold, n_folds=n_folds, val_size=val_size
        )
    elif fold is not None:
        assert n_folds is not None, "Must specify n_folds if fold is specified."
        train, val, test = get_patient_splits_nested_kfold(
            fold=fold, n_folds=n_folds, val_size=val_size
        )
    else:
        logging.info("Using default 5-fold split.")
        train, val, test = get_patient_splits_nested_kfold(fold=0, n_folds=5)

    return train, val, test


def get_core_ids(patient_ids):
    """returns the list of core ids for the given patient ids."""
    metadata_table = data_accessor.get_metadata_table()
    return metadata_table[
        metadata_table.patient_id.isin(patient_ids)
    ].core_id.values.tolist()


def remove_benign_cores_from_positive_patients(core_ids):
    """Returns the list of cores in the given list that are either malignant or from patients with no malignant cores."""
    table = data_accessor.get_metadata_table().copy()
    table["positive"] = table.grade.apply(lambda g: 0 if g == "Benign" else 1)
    num_positive_for_patient = table.groupby("patient_id").positive.sum()
    num_positive_for_patient.name = "patients_positive"
    table = table.join(num_positive_for_patient, on="patient_id")
    ALLOWED = table.query("positive == 1 or patients_positive == 0").core_id.to_list()

    return [core for core in core_ids if core in ALLOWED]


def remove_cores_below_threshold_involvement(core_ids, threshold_pct):
    """Returns the list of cores with at least the given percentage of cancer cells."""
    table = data_accessor.get_metadata_table().copy()
    ALLOWED = table.query(
        "grade == 'Benign' or pct_cancer >= @threshold_pct"
    ).core_id.to_list()
    return [core for core in core_ids if core in ALLOWED]


def undersample_benign(cores, seed=0, benign_to_cancer_ratio=1):
    """Returns the list of cores with the same cancer cores and the benign cores undersampled to the given ratio."""

    table = data_accessor.get_metadata_table().copy()
    benign = table.query('grade == "Benign"').core_id.to_list()
    cancer = table.query('grade != "Benign"').core_id.to_list()
    import random

    cores_benign = [core for core in cores if core in benign]
    cores_cancer = [core for core in cores if core in cancer]
    rng = random.Random(seed)
    cores_benign = rng.sample(
        cores_benign, int(len(cores_cancer) * benign_to_cancer_ratio)
    )

    return [core for core in cores if core in cores_benign or core in cores_cancer]


def apply_core_filters(
    core_ids,
    exclude_benign_cores_from_positive_patients=False,
    involvement_threshold_pct=None,
    undersample_benign_ratio=None,
):
    if exclude_benign_cores_from_positive_patients:
        core_ids = remove_benign_cores_from_positive_patients(core_ids)

    if involvement_threshold_pct is not None:
        if involvement_threshold_pct < 0 or involvement_threshold_pct > 100:
            raise ValueError(
                f"involvement_threshold_pct must be between 0 and 100, but got {involvement_threshold_pct}"
            )
        core_ids = remove_cores_below_threshold_involvement(
            core_ids, involvement_threshold_pct
        )

    if undersample_benign_ratio is not None:
        core_ids = undersample_benign(
            core_ids, seed=0, benign_to_cancer_ratio=undersample_benign_ratio
        )

    return core_ids


def select_cohort(
    fold=None,
    n_folds=5,
    test_center=None,
    exclude_benign_cores_from_positive_patients=False,
    involvement_threshold_pct=None,
    undersample_benign_ratio=None,
    splits_file=None,
    val_seed=0,
    val_size=0.2,
    mode: Literal[
        "kfold", "nested_kfold", "center", "train_only", None
    ] = None,  # Added mode for flexibility
    return_unfiltered_train_cores=False,
    run_test=False
):
    """Returns the list of core ids for the given cohort selection criteria.

    Default is to use the 5-fold split.

    Args:
        fold (int): If specified, the fold to use for the train/val/test split.
        n_folds (int): If specified, the number of folds to use for the train/val/test split.
        test_center (str): If specified, the center to use for the test set.

        The following arguments are used to filter the cores in the cohort, affecting
            only the train sets:
        remove_benign_cores_from_positive_patients (bool): If True, remove cores from patients with malignant cores that also have benign cores.
            Only applies to the training set.
        involvement_threshold_pct (float): If specified, remove cores with less than the given percentage of cancer cells.
            this should be a value between 0 and 100. Only applies to the training set.
        undersample_benign_ratio (float): If specified, undersample the benign cores to the given ratio. Only applies to the training set.
        seed (int): Random seed to use for the undersampling.
        splits_file: if specified, use the given csv file to load the train/val/test splits (kfold only)
    """

    if test_center is not None:
        logging.info(f"Using test center {test_center}")
        train, val, test = get_patient_splits_by_center(
            leave_out=test_center, val_size=val_size, val_seed=val_seed
        )
        if not run_test:
            train = train + val
            val = test
            test = []
    elif mode == "train_only":
        train, val, test = get_patient_splits_kfold()
        train = train + val + test
        val = []
        test = []
    elif mode == "kfold":
        assert fold is not None, "Must specify fold if mode is 'kfold'."
        assert n_folds is not None, "Must specify n_folds if fold is specified."
        train, val, test = get_patient_splits_kfold(fold=fold, n_folds=n_folds)
    elif mode == "nested_kfold":
        assert fold is not None, "Must specify fold if mode is 'nested_kfold'."
        assert n_folds is not None, "Must specify n_folds if fold is specified."
        train, val, test = get_patient_splits_nested_kfold(
            fold=fold, n_folds=n_folds, splits_file=splits_file, val_size=val_size
        )
    elif fold is not None:
        assert n_folds is not None, "Must specify n_folds if fold is specified."
        train, val, test = get_patient_splits_nested_kfold(
            fold=fold, n_folds=n_folds, splits_file=splits_file, val_size=val_size
        )
    else:
        logging.info("Using default 5-fold split.")
        train, val, test = get_patient_splits_nested_kfold(
            fold=0, n_folds=5, splits_file=splits_file
        )

    train_cores = get_core_ids(train)
    unfiltered_train_cores = train_cores.copy() 
    val_cores = get_core_ids(val)
    test_cores = get_core_ids(test)

    train_cores = apply_core_filters(
        train_cores,
        exclude_benign_cores_from_positive_patients=exclude_benign_cores_from_positive_patients,
        involvement_threshold_pct=involvement_threshold_pct,
        undersample_benign_ratio=undersample_benign_ratio,
    )

    print("Train cores:", len(train_cores))
    print("Validation cores:", len(val_cores))
    print("Test cores:", len(test_cores))

    if return_unfiltered_train_cores:
        return train_cores, val_cores, test_cores, unfiltered_train_cores
    else: 
        return train_cores, val_cores, test_cores


def get_parser():
    parser = ArgumentParser(add_help=False)
    group = parser.add_argument_group(
        "NCT2013 Cohort Selection", "Arguments for selecting the NCT2013 cohort."
    )

    group.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Fold number to use for the train/val/test split (default: None).",
    )
    group.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of folds to use for the train/val/test split (default: 5).",
    )
    group.add_argument(
        "--test_center",
        type=str,
        default=None,
        help="Center to use for the test set (default: None).",
    )
    group.add_argument(
        "--exclude_benign_cores_from_positive_patients",
        action=BooleanOptionalAction,
        default=True,
        help="If set, exclude benign cores from patients with malignant cores.",
    )
    group.add_argument(
        "--involvement_threshold_pct",
        type=float,
        default=None,
        help="If set, exclude cores with less than this percentage of cancer cells.",
    )
    group.add_argument(
        "--undersample_benign_ratio",
        type=float,
        default=None,
        help="If set, undersample benign cores to this ratio with respect to cancer cores.",
    )
    group.add_argument(
        "--cohort_selection_mode",
        type=str,
        choices=["kfold", "nested_kfold", "center", "train_only", None],
        default=None,
        help="Mode for cohort selection: 'kfold', 'nested_kfold', 'center', or None (default: None).",
    )
    return parser


def select_cohort_from_args(args, **kwargs):
    """Selects the cohort based on the provided command line arguments."""
    return select_cohort(
        fold=args.fold,
        n_folds=args.n_folds,
        test_center=args.test_center,
        exclude_benign_cores_from_positive_patients=args.exclude_benign_cores_from_positive_patients,
        involvement_threshold_pct=args.involvement_threshold_pct,
        undersample_benign_ratio=args.undersample_benign_ratio,
        mode=args.cohort_selection_mode,
        **kwargs
    )


if __name__ == "__main__":
    parser = ArgumentParser(parents=[get_parser()])
    args = parser.parse_args()

    train_cores, val_cores, test_cores = select_cohort_from_args(args)

    print("Train cores:", len(train_cores))
    print("Validation cores:", len(val_cores))
    print("Test cores:", len(test_cores))
