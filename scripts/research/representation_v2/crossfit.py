"""Subject-isolated slices of the incumbent outer-train payload."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.model_selection import GroupKFold

from src.pipeline.runner import OuterTrainOnlyData, WindowBatch


def _slice_batch(batch: WindowBatch | None, keep_sids: set[str]) -> WindowBatch | None:
    if batch is None:
        return None
    rows = np.asarray([window.sid in keep_sids for window in batch.windows], dtype=bool)
    return WindowBatch(batch.features[rows], batch.labels[rows], tuple(window for window, keep in zip(batch.windows, rows) if keep))


def slice_outer_train(data: OuterTrainOnlyData, subjects: Iterable[str]) -> OuterTrainOnlyData:
    """Remove every other subject from windows, metadata and truth together."""
    selected = set(subjects)
    known = set(data.subject_by_session.values())
    if not selected or selected - known:
        raise ValueError(f"unknown or empty subject selection: {sorted(selected - known)}")
    sids = {sid for sid, subject in data.subject_by_session.items() if subject in selected}
    return OuterTrainOnlyData(
        window_train=_slice_batch(data.window_train, sids),
        candidate_train=_slice_batch(data.candidate_train, sids),
        train_truths=tuple(truth for truth in data.train_truths if truth.sid in sids),
        subject_by_session={sid: subject for sid, subject in data.subject_by_session.items() if sid in sids},
        session_bounds_by_sid=None if data.session_bounds_by_sid is None else {
            sid: bounds for sid, bounds in data.session_bounds_by_sid.items() if sid in sids
        },
        micro_window_train=_slice_batch(data.micro_window_train, sids),
        micro_candidate_train=_slice_batch(data.micro_candidate_train, sids),
        micro_cache_extraction_seconds=data.micro_cache_extraction_seconds,
    )


def inner_subject_partitions(data: OuterTrainOnlyData, *, n_splits: int = 4) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
    """Deterministic GroupKFold partition over actual training-window subjects."""
    subjects = sorted(set(data.subject_by_session.values()))
    if n_splits < 2 or len(subjects) < n_splits:
        raise ValueError("not enough subjects for inner folds")
    groups = np.asarray([data.subject_by_session[w.sid] for w in data.window_train.windows])
    if set(groups) != set(subjects):
        raise ValueError("some mapped subjects have no training windows")
    folds = []
    for train_rows, score_rows in GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups):
        train = tuple(sorted(set(groups[train_rows])))
        score = tuple(sorted(set(groups[score_rows])))
        if set(train) & set(score):
            raise RuntimeError("inner partition subject overlap")
        folds.append((train, score))
    return tuple(folds)
