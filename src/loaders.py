"""
Trimmed from medAI/projects/ggnus_align/loaders.py — this repo only ever
needs check_grade_distribution() from that file, not its dataloader
machinery (which pulls in an unrelated ProstNFoundTransform variant from
projects/prostnfound/src/transform.py that this repo doesn't otherwise
depend on). The real dataloader used here is src/nct_optimum_loader.py.
"""

from collections import Counter

import numpy as np


def check_grade_distribution(dataset, split_name="dataset"):
    """
    Check grade distribution in a dataset.
    Assumes dataset has _indices and core_ids attributes,
    and df with core_id and grade_group columns.
    """
    df = dataset.df
    core_ids = dataset.core_ids

    # Build lookup
    core_id_to_grade = (
        df[["core_id", "grade_group"]]
        .drop_duplicates("core_id")
        .set_index("core_id")["grade_group"]
        .to_dict()
    )

    # Collect all grades
    grades = []
    for core_idx, _ in dataset._indices:
        core_id = str(core_ids[core_idx]).strip()
        grade = core_id_to_grade.get(core_id, None)
        if grade is not None:
            grades.append(int(grade))

    total = len(grades)
    counts = Counter(grades)

    print(f"\n{'='*40}")
    print(f"Grade distribution — {split_name} ({total} samples)")
    print(f"{'='*40}")
    for grade in sorted(counts.keys()):
        count = counts[grade]
        pct = count / total * 100
        bar = '█' * int(pct / 2)
        print(f"  Grade {grade}: {count:5d} ({pct:5.1f}%)  {bar}")

    print(f"\nInverse frequency weights (normalized to max=1.0):")
    inv_freq = {g: total / counts[g] for g in sorted(counts.keys())}
    max_weight = max(inv_freq.values())
    normalized = {g: round(v / max_weight, 4) for g, v in inv_freq.items()}
    print(f"  {normalized}")

    print(f"\nSquare root inverse frequency weights (softer):")
    sqrt_inv = {g: np.sqrt(total / counts[g]) for g in sorted(counts.keys())}
    max_sqrt = max(sqrt_inv.values())
    normalized_sqrt = {g: round(v / max_sqrt, 4) for g, v in sqrt_inv.items()}
    print(f"  {normalized_sqrt}")
