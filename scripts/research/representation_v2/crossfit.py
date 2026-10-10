"""Subject-isolated slices of the incumbent outer-train payload."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupKFold

from src.pipeline import runner as _runner
from src.pipeline.crossfit import crossfit_predict_proba
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


@dataclass(frozen=True)
class CandidateLedger:
    candidates: tuple[object, ...]
    features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    micro_threshold: float
    fit_subjects: tuple[str, ...]
    score_subjects: tuple[str, ...]


def build_candidate_ledgers(data: OuterTrainOnlyData, config: object, train_subjects: Iterable[str],
                            score_subjects: Iterable[str], *, fit_audit: dict | None = None) -> tuple[CandidateLedger, CandidateLedger | None]:
    """Build T OOF and optional V candidates with T-only learned ancestors.

    The micro threshold is selected from T truth only. V truth is consulted only
    after candidate generation, for probe metrics/labels.
    Empty score_subjects builds only T OOF, without V reads or final refits.
    """
    train = tuple(sorted(set(train_subjects)))
    score = tuple(sorted(set(score_subjects)))
    if set(train) & set(score):
        raise ValueError("candidate producer train/score subject overlap")
    t = slice_outer_train(data, train)
    v = slice_outer_train(data, score) if score else None
    if t.micro_window_train is None or t.micro_candidate_train is None or (v is not None and v.micro_candidate_train is None):
        raise ValueError("registered candidate producer requires micro path")

    def _fit_and_score(train_batch: WindowBatch, t_candidate: WindowBatch, v_candidate: WindowBatch | None,
                       keep: np.ndarray, estimator_factory, *, time_prior: bool):
        x = lambda batch: _runner._with_time_prior(batch.features, batch.windows) if time_prior else batch.features
        labels = np.asarray(train_batch.labels[keep], dtype=np.int8)
        _runner._validate_binary(labels, "candidate producer training")
        groups = _runner._groups_for(train_batch.windows, t.subject_by_session)[keep]
        t_groups = _runner._groups_for(t_candidate.windows, t.subject_by_session)
        oof = crossfit_predict_proba(x(train_batch)[keep], labels, groups, x(t_candidate), t_groups,
                                    config.inner_splits, estimator_factory=estimator_factory)
        t_scores = oof.probabilities
        if fit_audit is not None:
            name = 'macro' if time_prior else 'micro'
            for fold in oof.folds:
                fit_audit[f'{name}_oof{fold.index}'] = {'fit_subjects': list(map(str, fold.train_groups)), 'parents': []}
        if v_candidate is None:
            return _runner._windows_by_session(t_candidate.windows, t_scores), None
        model = estimator_factory()
        model.fit(x(train_batch)[keep], labels)
        if fit_audit is not None:
            fit_audit[f'{name}_refit'] = {'fit_subjects': sorted(map(str, set(groups))), 'parents': []}
        v_scores = _runner._positive_probability(model, x(v_candidate))
        return (_runner._windows_by_session(t_candidate.windows, t_scores),
                _runner._windows_by_session(v_candidate.windows, v_scores))

    macro_t, macro_v = _fit_and_score(
        t.window_train, t.candidate_train, v.candidate_train if v is not None else None,
        np.asarray(t.window_train.labels) >= 0,
        lambda: _runner._window_estimator(config.seed), time_prior=True)
    micro_t, micro_v = _fit_and_score(
        t.micro_window_train, t.micro_candidate_train, v.micro_candidate_train if v is not None else None,
        _runner._micro_training_keep(t.micro_window_train, t.train_truths, config.micro_positive_middle_fraction),
        lambda: _runner._micro_window_estimator(config.seed + 10), time_prior=False)
    selected = _runner.select_micro_candidate_threshold(
        micro_t, t.train_truths, config.micro_threshold_grid, config.micro_candidate)
    if fit_audit is not None:
        fit_audit['threshold'] = {
            'fit_subjects': sorted({str(t.subject_by_session[truth.sid]) for truth in t.train_truths}),
            'parents': [key for key in fit_audit if key.startswith('micro_oof')]}
        fit_audit['candidates'] = {'fit_subjects': [], 'parents': list(fit_audit)}

    def _ledger(partition: OuterTrainOnlyData, macro: dict, micro: dict, scored_subjects: tuple[str, ...]) -> CandidateLedger:
        candidates = tuple(_runner.union_candidates(
            _runner.density_candidates(macro, config.density),
            _runner.micro_candidates(micro, selected.threshold, config.micro_candidate)))
        features = _runner.multiscale_verifier_features(
            candidates, macro, micro, context_features_version="v1",
            session_bounds_by_sid=_runner._context_session_bounds(
                partition, candidates, partition.candidate_train.windows, partition.micro_candidate_train.windows))
        if features.shape != (len(candidates), 116):
            raise RuntimeError("candidate ledger must use incumbent 116D Context-v1")
        labels = _runner._candidate_labels(candidates, partition.train_truths)
        groups = _runner._groups_for([candidate.event for candidate in candidates], partition.subject_by_session)
        return CandidateLedger(candidates, features, labels, groups, float(selected.threshold), train, scored_subjects)

    return _ledger(t, macro_t, micro_t, train), _ledger(v, macro_v, micro_v, score) if v is not None else None
