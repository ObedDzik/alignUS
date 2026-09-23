import pandas as pd
import os
from sklearn.model_selection import KFold


class OptimumSplitsGenerator: 
    def __init__(self, cine_info_path: str):
        self.cine_info_path = cine_info_path
        self.cine_info_table = pd.read_csv(cine_info_path)
        self._patient_splits = None 
        self._cine_id_splits = None

    def _get_cine_ids(self, patient_ids): 
        cine_ids = self.cine_info_table[self.cine_info_table['case'].isin(patient_ids)]['cine_id'].tolist()
        return cine_ids

    def split_patients_kfold(self, k: int, fold=0, seed=42):
        patient_ids = self.cine_info_table['case'].unique().tolist()
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        splits = list(kf.split(patient_ids))
        train_patient_ids = [patient_ids[i] for i in splits[fold][0]]
        val_patient_ids = [patient_ids[i] for i in splits[fold][1]]
        self._patient_splits = dict(
            train=train_patient_ids,
            val=val_patient_ids,
        )
        return self

    def get_cine_id_splits(self): 
        if self._cine_id_splits is None:
            if self._patient_splits is None:
                raise ValueError("Patient splits not generated yet.")
            self._cine_id_splits = {
                split: self._get_cine_ids(patient_ids)
                for split, patient_ids in self._patient_splits.items()
            }
        return self._cine_id_splits
    
    def get_patient_splits(self): 
        if self._patient_splits is None:
            raise ValueError("Patient splits not generated yet.")
        return self._patient_splits