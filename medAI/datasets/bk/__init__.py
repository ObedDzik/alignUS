from enum import StrEnum
import os
import pandas as pd
from .utils import get_metadata_table
import numpy as np
import json
import logging


QUEENS_BK_RAW_DATA_DIR = os.environ.get("QUEENS_BK_RAW_DATA_DIR")
UBC_BK_RAW_DATA_DIR = os.environ.get("UBC_BK_RAW_DATA_DIR")


class DataKeys(StrEnum):
    RF = "rf"
    NEEDLE_BM = "needle_bm"
    NEEDLE = "needle"
    PROSTATE = "prostate"


class RawDataAccessor:
    def __init__(
        self,
        center_to_dir={"queens": QUEENS_BK_RAW_DATA_DIR, "ubc": UBC_BK_RAW_DATA_DIR},
    ):
        tables = []
        for center, dir in center_to_dir.items():
            if not os.path.exists(dir):
                raise ValueError(f"{dir} does not exist.")
            table = get_metadata_table(dir)

            if center == 'ubc': 
                table['Involvement'] = table["Involvement"] * 100
            table['Involvement'] = table['Involvement'].apply(lambda x: 0 if np.isnan(x) else x)

            table["center"] = center
            tables.append(table)

        self.metadata = pd.concat(tables)
        self.metadata["unique_core_id"] = self.metadata.apply(
            lambda x: f"{x['center']}_{str(x['PatientId']).zfill(4)}_{x['CoreName']}",
            axis=1,
        )
        self.metadata["unique_patient_id"] = self.metadata.apply(
            lambda x: f"{x['center']}_{str(x['PatientId']).zfill(4)}", axis=1
        )
        assert self.metadata["unique_core_id"].is_unique, "unique_core_id is not unique"
        self.metadata["positive"] = self.metadata.Pathology.apply(
            lambda g: 1 if g == "Adenocarcinoma" else 0
        )
   
    def get_ids(self):
        return self.metadata["unique_core_id"].tolist()

    def load_data(self, id, key: DataKeys, missing_strat='raise', **kwargs):
        row = self.metadata.loc[self.metadata["unique_core_id"] == id].iloc[0]
        path = row[f"{key}_path"]
        if path is None:
            if missing_strat == 'raise':
                raise ValueError(f"Path for {key} is None for core {id}")
            return None
        return np.load(path, **kwargs)

    def get_rf(self, id, **kwargs):
        return self.load_data(id, DataKeys.RF, **kwargs)

    def get_prostate(self, id, **kwargs):
        return self.load_data(id, DataKeys.PROSTATE, **kwargs)

    def get_needle(self, id, **kwargs):
        return self.load_data(id, DataKeys.NEEDLE, **kwargs)

    def get_metadata_for_id(self, id):
        return self.metadata.loc[self.metadata["unique_core_id"] == id].iloc[0].to_dict()


class CohortSelector: 
    def __init__(self, metadata_table):
        self.metadata = metadata_table

    def get_patient_splits_by_fold(self, fold, n_folds, seed=0):
        patient_table = self.metadata.groupby('unique_patient_id').first()

        from sklearn.model_selection import StratifiedKFold, train_test_split

        PATIENT_FOLDS = {}

        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for i, (train_idx, test_idx) in enumerate(
            kfold.split(patient_table.index, patient_table["center"])
        ):
            train = patient_table.iloc[train_idx]
            test = patient_table.iloc[test_idx]
            train, val = train_test_split(
                train, test_size=0.2, random_state=0, stratify=train["center"]
            )
            train = train.index.values.tolist()
            val = val.index.values.tolist()
            test = test.index.values.tolist()

            PATIENT_FOLDS[i] = [train, val, test]

        # double check that the splits are correct
        for i in range(n_folds):
            # print(f"Fold {i}")
            # print(f"Train: {len(PATIENT_FOLDS[i]['train'])}")
            # print(f"Val: {len(PATIENT_FOLDS[i]['val'])}")
            # print(f"Test: {len(PATIENT_FOLDS[i]['test'])}")
            assert len(set(PATIENT_FOLDS[i][0]) & set(PATIENT_FOLDS[i][1])) == 0
            assert len(set(PATIENT_FOLDS[i][0]) & set(PATIENT_FOLDS[i][2])) == 0
            assert len(set(PATIENT_FOLDS[i][1]) & set(PATIENT_FOLDS[i][2])) == 0

        test_sets = []
        for i in range(n_folds):
            test_sets.append(PATIENT_FOLDS[i][2])

        for i in range(n_folds):
            for j in range(i+1, n_folds):
                assert len(set(test_sets[i]) & set(test_sets[j])) == 0

        return PATIENT_FOLDS[fold]

    def get_core_ids(self, patient_ids):
        return self.metadata.loc[self.metadata['unique_patient_id'].isin(patient_ids)]['unique_core_id'].to_list()

    def remove_benign_cores_from_positive_patients(self, core_ids):
        """Returns the list of cores in the given list that are either malignant or from patients with no malignant cores."""
        table = self.metadata.copy()
        num_positive_for_patient = table.groupby("unique_patient_id").positive.sum()
        num_positive_for_patient.name = "patients_positive"
        table = table.join(num_positive_for_patient, on="unique_patient_id")
        ALLOWED = table.query("positive == 1 or patients_positive == 0").unique_core_id.to_list()

        return [core for core in core_ids if core in ALLOWED]

    def remove_cores_below_threshold_involvement(self, core_ids, threshold_pct):
        """Returns the list of cores with at least the given percentage of cancer cells."""
        table = self.metadata.copy()
        ALLOWED = table.query(
            "positive == 0 or Involvement >= @threshold_pct"
        ).unique_core_id.to_list()
        return [core for core in core_ids if core in ALLOWED]

    def undersample_benign(self, cores, seed=0, benign_to_cancer_ratio=1):
        """Returns the list of cores with the same cancer cores and the benign cores undersampled to the given ratio."""

        table = self.metadata.copy()
        benign = table.query('positive == 0').unique_core_id.to_list()
        cancer = table.query('positive == 1').unique_core_id.to_list()
        import random

        cores_benign = [core for core in cores if core in benign]
        cores_cancer = [core for core in cores if core in cancer]
        rng = random.Random(seed)
        cores_benign = rng.sample(
            cores_benign, int(len(cores_cancer) * benign_to_cancer_ratio)
        )

        return [core for core in cores if core in cores_benign or core in cores_cancer]

    def apply_core_filters(
        self,
        core_ids,
        exclude_benign_cores_from_positive_patients=False,
        involvement_threshold_pct=None,
        undersample_benign_ratio=None,
    ):
        if exclude_benign_cores_from_positive_patients:
            core_ids = self.remove_benign_cores_from_positive_patients(core_ids)

        if involvement_threshold_pct is not None:
            if involvement_threshold_pct < 0 or involvement_threshold_pct > 100:
                raise ValueError(
                    f"involvement_threshold_pct must be between 0 and 100, but got {involvement_threshold_pct}"
                )
            core_ids = self.remove_cores_below_threshold_involvement(
                core_ids, involvement_threshold_pct
            )

        if undersample_benign_ratio is not None:
            core_ids = self.undersample_benign(
                core_ids, seed=0, benign_to_cancer_ratio=undersample_benign_ratio
            )

        return core_ids

    def select_cohort(
        self, 
        fold=None,
        n_folds=None,
        test_center=None,
        exclude_benign_cores_from_positive_patients=False,
        involvement_threshold_pct=None,
        undersample_benign_ratio=None,
        val_seed=0,
        val_size=0.2,
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
            raise NotImplementedError("Test center selection is not implemented.")
            logging.info(f"Using test center {test_center}")
            train, val, test = self.get_patient_splits_by_center(
                leave_out=test_center, val_size=val_size, val_seed=val_seed
            )
        elif fold is not None:
            assert n_folds is not None, "Must specify n_folds if fold is specified."
            train, val, test = self.get_patient_splits_by_fold(
                fold=fold, n_folds=n_folds
            )
        else:
            logging.info("Using default 5-fold split.")
            train, val, test = self.get_patient_splits_by_fold(
                fold=0, n_folds=5
            )

        train_cores = self.get_core_ids(train)
        val_cores = self.get_core_ids(val)
        test_cores = self.get_core_ids(test)

        train_cores = self.apply_core_filters(
            train_cores,
            exclude_benign_cores_from_positive_patients=exclude_benign_cores_from_positive_patients,
            involvement_threshold_pct=involvement_threshold_pct,
            undersample_benign_ratio=undersample_benign_ratio,
        )

        # if exclude_benign_cores_from_positive_patients:
        #     train_cores = remove_benign_cores_from_positive_patients(train_cores)
        #
        # if involvement_threshold_pct is not None:
        #     if involvement_threshold_pct < 0 or involvement_threshold_pct > 100:
        #         raise ValueError(
        #             f"involvement_threshold_pct must be between 0 and 100, but got {involvement_threshold_pct}"
        #         )
        #     train_cores = remove_cores_below_threshold_involvement(
        #         train_cores, involvement_threshold_pct
        #     )
        #
        # if undersample_benign_ratio is not None:
        #     train_cores = undersample_benign(
        #         train_cores, seed=seed, benign_to_cancer_ratio=undersample_benign_ratio
        #     )

        return train_cores, val_cores, test_cores