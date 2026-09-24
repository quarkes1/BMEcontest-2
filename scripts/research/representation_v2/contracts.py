"""Subject partition contracts for representation research."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Partition:
    outer_fold: int
    inner_fold: int
    subfold: int | None
    train_subjects: tuple[str, ...]
    score_subjects: tuple[str, ...]
    parent_ids: tuple[str, ...]


def validate_partition(partition: Partition) -> Partition:
    if partition.outer_fold not in range(5) or partition.inner_fold not in range(4):
        raise ValueError("outer/inner fold is outside the registered range")
    if partition.subfold is not None and partition.subfold not in range(3):
        raise ValueError("subfold is outside the registered range")
    if not partition.train_subjects or not partition.score_subjects:
        raise ValueError("partition needs both train and score subjects")
    for role, subjects in (("train", partition.train_subjects), ("score", partition.score_subjects)):
        if any(not isinstance(s, str) or not s for s in subjects):
            raise ValueError(f"{role} subjects must be nonempty strings")
        if len(subjects) != len(set(subjects)):
            raise ValueError(f"duplicate {role} subject")
    overlap = set(partition.train_subjects) & set(partition.score_subjects)
    if overlap:
        raise ValueError(f"subject overlap: {sorted(overlap)}")
    return partition
