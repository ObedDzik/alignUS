import pandas as pd
import os


def get_cleaned_ua_metadata_table(
    filter_has_pirads=False,
    filter_has_primus=False,
    filter_pct_cancer_threshold=None,
    low_grade_exclusion_threshold=None,
    path=None,
):

    path = path or os.path.join(
        os.environ["EXACTVU_PCA_DATA_ROOT"],
        "OPTIMUM/processed/UA_annotated_needles/mined_path_reports.csv",
    )

    table = pd.read_csv(path)
    table = table.loc[table["Diagnosis"].isin(["Benign", "Carcinoma"])]

    table["PI-RADS"] = table["PI-RADS"].apply(
        lambda val: val if pd.isna(val) else int(val)
    )
    table["PRI-MUS"] = table["PRI-MUS"].apply(
        lambda val: int(val) if str(val) in "12345" else float("nan")
    )
    table.loc[table["Diagnosis"] == "Benign", "GG"] = 0

    table.rename(
        columns={
            "GG": "grade_group",
            "% Cancer": "pct_cancer",
            "cine_id": "core_id",
            "case": "patient_id",
        },
        inplace=True,
    )

    table.loc[table["pct_cancer"].isna(), "pct_cancer"] = 0

    if filter_has_pirads:
        table = table.loc[~table["PI-RADS"].isna()]
    if filter_has_primus:
        table = table.loc[~table["PRI-MUS"].isna()]
    if filter_pct_cancer_threshold:
        table = table.loc[
            (table["Diagnosis"] == "Benign")
            | (table["pct_cancer"] > filter_pct_cancer_threshold)
        ]
    if low_grade_exclusion_threshold:
        table = table.loc[
            (table["Diagnosis"] == "Benign")
            | (table["grade_group"] > low_grade_exclusion_threshold)
        ]

    return table.set_index("core_id")
