from collections import Counter
import torch
from torch.utils.data import WeightedRandomSampler
from collections import defaultdict
import random

class GradeStratifiedSampler:
    """
    WeightedRandomSampler that balances sampling across ISUP grade groups.
    Reads grade_group directly from the dataset's core_ids and df.
    Each index in the dataset corresponds to a (core_idx, frame_idx) pair —
    we assign the weight of the core's grade to each frame index.
    """
    def __init__(self, dataset, num_samples: int = None):
        self.dataset = dataset
        self.num_samples = num_samples or len(dataset)

    def build(self) -> WeightedRandomSampler:
        grade_groups = self._get_grade_groups()
        counts = Counter(grade_groups)
        weights = torch.tensor(
            [1.0 / counts[g] for g in grade_groups],
            dtype=torch.float
        )
        return WeightedRandomSampler(
            weights=weights,
            num_samples=self.num_samples,
            replacement=True
        )
    def _get_grade_groups(self):
        df = self.dataset.df
        core_ids = self.dataset.core_ids

        core_id_to_grade = (
            df[["core_id", "grade_group"]]
            .drop_duplicates("core_id")
            .set_index("core_id")["grade_group"]
            .to_dict()
        )

        # Sanity check: warn if many cores are missing
        grade_groups = []
        missing = []
        for core_idx, _ in self.dataset._indices:
            core_id = str(core_ids[core_idx]).strip()
            grade = core_id_to_grade.get(core_id)
            if grade is None:
                missing.append(core_id)
                grade = 0
            grade_groups.append(int(grade))

        if missing:
            print(f"GradeStratifiedSampler: {len(missing)} indices missing from df, defaulting to grade 0. Sample missing: {missing[:5]}")

        return grade_groups

class GradeBalancedBatchSampler(torch.utils.data.Sampler):
    """
    Constructs batches with exactly samples_per_grade samples from each grade.
    Total batch size = samples_per_grade * num_grades.
    """
    def __init__(self, dataset, samples_per_grade: int = 4, num_grades: int = 6):
        self.samples_per_grade = samples_per_grade
        self.num_grades = num_grades
        self.batch_size = samples_per_grade * num_grades

        # Build index lists per grade
        df = dataset.df
        core_ids = dataset.core_ids
        core_id_to_grade = (
            df[["core_id", "grade_group"]]
            .drop_duplicates("core_id")
            .set_index("core_id")["grade_group"]
            .to_dict()
        )

        self.grade_indices = defaultdict(list)
        for i, (core_idx, _) in enumerate(dataset._indices):
            core_id = str(core_ids[core_idx]).strip()
            grade = int(core_id_to_grade.get(core_id, 0))
            self.grade_indices[grade].append(i)

        # Verify all grades have enough samples
        for grade, indices in self.grade_indices.items():
            print(f"Grade {grade}: {len(indices)} samples available")

        self.num_batches = max(
            len(indices) // samples_per_grade
            for indices in self.grade_indices.values()
            if len(indices) >= samples_per_grade
        )

    def __iter__(self):
        # Shuffle indices within each grade at the start of each epoch
        shuffled = {
            grade: torch.randperm(len(indices)).tolist()
            for grade, indices in self.grade_indices.items()
        }
        pointers = {grade: 0 for grade in self.grade_indices}

        for _ in range(self.num_batches):
            batch = []
            for grade in sorted(self.grade_indices.keys()):
                indices = self.grade_indices[grade]
                ptr = pointers[grade]
                # Wrap around if exhausted
                selected = []
                while len(selected) < self.samples_per_grade:
                    if ptr >= len(shuffled[grade]):
                        shuffled[grade] = torch.randperm(len(indices)).tolist()
                        ptr = 0
                    selected.append(indices[shuffled[grade][ptr]])
                    ptr += 1
                pointers[grade] = ptr
                batch.extend(selected)
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches