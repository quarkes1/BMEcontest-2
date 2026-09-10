"""Leakage-safe nested runner for the event verification stack."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Mapping, Sequence

def _physical_cpu_count() -> int:
    """Prefer physical cores, with a conservative limit when detection is absent."""
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        if physical is not None and physical > 0:
            return int(physical)
    except (ImportError, OSError, NotImplementedError):
        pass
    return max(1, (os.cpu_count() or 1) // 2)


# Set the limit before sklearn initializes joblib. Loky skips WMIC detection
# only below the logical count, including machines without hyperthreading.
# Keep a positive floor and preserve a caller's explicit worker limit.
os.environ.setdefault(
    "LOKY_MAX_CPU_COUNT",
    str(max(1, min(_physical_cpu_count(), (os.cpu_count() or 1) - 1))),
)

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.eval.metrics import event_iou
from src.pipeline.crossfit import crossfit_predict_proba
from src.pipeline.event_stack import (
    _GLOBAL_PRIOR,
    CandidateEvent,
    MultiScaleCandidate,
    DensityConfig,
    MicroCandidateConfig,
    EventSelectionPolicy,
    EventMetrics,
    EventRef,
    aggregate_candidate_features,
    apply_event_policy,
    compute_event_metrics,
    density_candidates,
    micro_candidates,
    multiscale_verifier_features,
    select_micro_candidate_threshold,
    select_event_policy,
    select_event_threshold,
    verifier_features,
    union_candidates,
)


WINDOW_MODEL_PARAMETERS = {
    "learning_rate": 0.05,
    "max_iter": 150,
    "max_leaf_nodes": 15,
    "max_depth": 4,
    "min_samples_leaf": 100,
    "l2_regularization": 1.0,
    "early_stopping": False,
}
VERIFIER_MODEL_PARAMETERS = {
    "C": 0.1,
    "class_weight": "balanced",
    "max_iter": 3000,
}
MICRO_WINDOW_MODEL_PARAMETERS = {
    "n_estimators": 300,
    "num_leaves": 31,
    "min_child_samples": 100,
    "learning_rate": 0.05,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "class_weight": "balanced",
    "n_jobs": 1,
    "verbosity": -1,
}
RUNNER_SCHEMA_VERSION = 4


@dataclass(frozen=True)
class RunConfig:
    outer_fold: int
    inner_splits: int = 4
    seed: int = 20260908
    no_tcn: bool = True
    workers: int = 1
    device: str = "auto"
    subject_cap_grid: tuple[int, ...] = ()
    verifier_feature_mode: str = "probability"
    verifier_c_grid: tuple[float, ...] = (0.1,)
    density: DensityConfig = field(default_factory=DensityConfig)
    micro_enabled: bool = False
    micro_gravity_align: bool = True
    micro_threshold_grid: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50)
    micro_candidate: MicroCandidateConfig = field(default_factory=MicroCandidateConfig)
    micro_positive_middle_fraction: float | None = None
    external_fd_weight_grid: tuple[float, ...] = (0.0,)


@dataclass(frozen=True)
class FoldResult:
    config_hash: str
    threshold: float
    max_events_per_subject: int | None
    verifier_c: float
    verifier_feature_count: int
    inner_metrics: EventMetrics
    outer_metrics: EventMetrics
    candidate_count: int
    candidate_match_recall: float
    slices: Mapping[str, EventMetrics]
    timings_seconds: Mapping[str, float]
    cache_hits: Mapping[str, bool]
    outer_subjects: frozenset[str]
    window_fit_subjects: frozenset[str]
    verifier_fit_subjects: frozenset[str]
    micro_threshold: float | None = None
    micro_candidate_count: int = 0
    micro_candidate_match_recall: float = 0.0
    short_meal_candidate_recall: float = 0.0
    external_weight: float = 0.0
    macro_window_feature_count: int = 0
    micro_window_feature_count: int = 0
    micro_window_fit_subjects: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class WindowBatch:
    """One in-memory NPZ-equivalent window batch."""

    features: np.ndarray
    labels: np.ndarray
    windows: tuple[EventRef, ...]


@dataclass(frozen=True)
class FoldDataset:
    """All arrays and truth metadata required for one untouched outer fold."""

    window_train: WindowBatch
    candidate_train: WindowBatch
    validation: WindowBatch
    train_truths: tuple[EventRef, ...]
    validation_truths: tuple[EventRef, ...]
    subject_by_session: Mapping[str, str]
    outer_subjects: frozenset[str]
    validation_truth_slices: Mapping[str, tuple[EventRef, ...]] = field(
        default_factory=dict
    )
    micro_window_train: WindowBatch | None = None
    micro_candidate_train: WindowBatch | None = None
    micro_validation: WindowBatch | None = None
    micro_cache_extraction_seconds: float = 0.0


def _file_signature(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def cache_key(
    config: RunConfig,
    feature_dimensions: Sequence[int] = (),
    input_files: Sequence[Path] = (),
    model_parameters: Mapping[str, object] | None = None,
) -> str:
    """Hash every setting and input identity that can change fold predictions."""

    models = model_parameters or {
        "window": WINDOW_MODEL_PARAMETERS,
        "micro_window": MICRO_WINDOW_MODEL_PARAMETERS,
        "verifier": VERIFIER_MODEL_PARAMETERS,
    }
    payload = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "config": asdict(config),
        "feature_dimensions": list(feature_dimensions),
        "model_parameters": models,
        "input_files": [_file_signature(Path(path)) for path in input_files],
    }
    canonical = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def validate_outer_isolation(
    fit_subjects: set[str] | frozenset[str],
    outer_subjects: set[str] | frozenset[str],
) -> None:
    overlap = sorted(set(fit_subjects) & set(outer_subjects))
    if overlap:
        raise ValueError("outer subject leakage: " + ", ".join(overlap))


def _validate_batch(batch: WindowBatch, name: str) -> None:
    if batch.features.ndim != 2:
        raise ValueError(f"{name}.features must be two-dimensional")
    if batch.labels.ndim != 1:
        raise ValueError(f"{name}.labels must be one-dimensional")
    if len(batch.features) != len(batch.labels) or len(batch.features) != len(
        batch.windows
    ):
        raise ValueError(f"{name} arrays must have equal row counts")


def _groups_for(
    windows: Sequence[EventRef], subject_by_session: Mapping[str, str]
) -> np.ndarray:
    missing = sorted({window.sid for window in windows} - set(subject_by_session))
    if missing:
        raise ValueError("sessions absent from subject mapping: " + ", ".join(missing))
    return np.asarray([subject_by_session[window.sid] for window in windows])


def _with_time_prior(features: np.ndarray, windows: Sequence[EventRef]) -> np.ndarray:
    prior = np.asarray(
        [
            _GLOBAL_PRIOR[int((window.start_ms / 3.6e6) % 24)]
            for window in windows
        ],
        dtype=np.float32,
    ).reshape((-1, 1))
    return np.concatenate((np.asarray(features), prior), axis=1)


def _window_estimator(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    **WINDOW_MODEL_PARAMETERS,
                    random_state=seed,
                ),
            ),
        ]
    )


def _micro_window_estimator(seed: int) -> Pipeline:
    from lightgbm import LGBMClassifier

    parameters = dict(MICRO_WINDOW_MODEL_PARAMETERS)
    parameters["random_state"] = seed
    return Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("model", LGBMClassifier(**parameters))]
    )


def _verifier_estimator(seed: int, regularization_c: float) -> Pipeline:
    parameters = dict(VERIFIER_MODEL_PARAMETERS)
    parameters["C"] = regularization_c
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    **parameters,
                    random_state=seed,
                ),
            ),
        ]
    )


def _windows_by_session(
    windows: Sequence[EventRef], probabilities: np.ndarray
) -> dict[str, list[tuple[int, int, float]]]:
    if len(windows) != len(probabilities):
        raise ValueError("window probabilities must align with windows")
    grouped: defaultdict[str, list[tuple[int, int, float]]] = defaultdict(list)
    for window, probability in zip(windows, probabilities):
        grouped[window.sid].append(
            (window.start_ms, window.end_ms, float(probability))
        )
    return {sid: sorted(rows) for sid, rows in grouped.items()}


def _candidate_labels(
    candidates: Sequence[CandidateEvent | MultiScaleCandidate], truths: Sequence[EventRef]
) -> np.ndarray:
    labels = []
    for candidate in candidates:
        best_iou = max(
            (
                event_iou(candidate.event.interval, truth.interval)
                for truth in truths
                if truth.sid == candidate.event.sid
            ),
            default=0.0,
        )
        labels.append(int(best_iou >= 0.25))
    return np.asarray(labels, dtype=np.int8)


def _candidate_matrix(
    candidates: Sequence[CandidateEvent],
    windows_by_sid: Mapping[str, Sequence[tuple[int, int, float]]],
    include_coverage: bool,
    window_batch: WindowBatch,
    feature_mode: str,
) -> tuple[tuple[CandidateEvent, ...], np.ndarray]:
    usable = tuple(candidate for candidate in candidates if len(candidate.probabilities) >= 2)
    features = verifier_features(
        usable,
        windows_by_sid,
        include_coverage=include_coverage,
    )
    if len(features) != len(usable):
        raise RuntimeError("verifier feature rows do not align with candidates")
    if feature_mode == "raw_summary":
        raw_features = aggregate_candidate_features(
            usable,
            window_batch.windows,
            window_batch.features,
        )
        features = np.concatenate((features, raw_features), axis=1)
    elif feature_mode != "probability":
        raise ValueError("verifier_feature_mode must be probability or raw_summary")
    return usable, features


def _positive_probability(estimator: Pipeline, features: np.ndarray) -> np.ndarray:
    values = np.asarray(estimator.predict_proba(features), dtype=np.float64)
    classes = np.asarray(estimator.classes_)
    columns = np.flatnonzero(classes == 1)
    if values.ndim != 2 or len(columns) != 1:
        raise ValueError("estimator must expose binary positive class 1")
    scores = values[:, int(columns[0])]
    if not np.isfinite(scores).all():
        raise ValueError("estimator produced non-finite probabilities")
    return scores


def _validate_binary(labels: np.ndarray, stage: str) -> None:
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"{stage} requires both binary classes")


def _micro_training_keep(
    batch: WindowBatch,
    truths: Sequence[EventRef],
    middle_fraction: float | None,
) -> np.ndarray:
    """Exclude ambiguous labels and optionally retain pure positive centers."""
    keep = np.asarray(batch.labels) >= 0
    if middle_fraction is None:
        return keep
    if not 0 < middle_fraction <= 1:
        raise ValueError("micro_positive_middle_fraction must be in (0, 1]")
    truths_by_sid: defaultdict[str, list[EventRef]] = defaultdict(list)
    for truth in truths:
        truths_by_sid[truth.sid].append(truth)
    for index in np.flatnonzero(np.asarray(batch.labels) == 1):
        window = batch.windows[index]
        center = (window.start_ms + window.end_ms) // 2
        keep[index] = any(
            truth.start_ms + (truth.end_ms - truth.start_ms) * (1 - middle_fraction) / 2
            <= center <=
            truth.end_ms - (truth.end_ms - truth.start_ms) * (1 - middle_fraction) / 2
            for truth in truths_by_sid[window.sid]
        )
    return keep


def _run_outer_dataset(
    config: RunConfig,
    data_source: FoldDataset,
) -> FoldResult:
    """Run nested OOF threshold selection and one untouched outer evaluation."""

    if config.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if config.device == "cuda":
        raise ValueError("CUDA is not available in the no-TCN CPU foundation runner")
    if not config.no_tcn:
        raise ValueError("TCN scoring is not implemented by the CPU foundation runner")
    if config.verifier_feature_mode not in {"probability", "raw_summary"}:
        raise ValueError("verifier_feature_mode must be probability or raw_summary")
    if not config.verifier_c_grid or any(value <= 0 for value in config.verifier_c_grid):
        raise ValueError("verifier_c_grid must contain positive values")
    if len(set(config.verifier_c_grid)) != len(config.verifier_c_grid):
        raise ValueError("verifier_c_grid values must be unique")
    if not isinstance(data_source, FoldDataset):
        raise TypeError("data_source must be a FoldDataset")
    if config.external_fd_weight_grid != (0.0,):
        raise ValueError("external_fd_weight_grid must be (0.0,) in the target-domain runner")
    started = time.perf_counter()
    for name, batch in (
        ("window_train", data_source.window_train),
        ("candidate_train", data_source.candidate_train),
        ("validation", data_source.validation),
    ):
        _validate_batch(batch, name)

    window_groups = _groups_for(
        data_source.window_train.windows, data_source.subject_by_session
    )
    candidate_window_groups = _groups_for(
        data_source.candidate_train.windows, data_source.subject_by_session
    )
    validation_groups = _groups_for(
        data_source.validation.windows, data_source.subject_by_session
    )
    window_fit_subjects = frozenset(window_groups)
    observed_outer_subjects = frozenset(validation_groups)
    if observed_outer_subjects != data_source.outer_subjects:
        raise ValueError("outer subject manifest does not match validation windows")
    validate_outer_isolation(window_fit_subjects, data_source.outer_subjects)
    validate_outer_isolation(
        frozenset(candidate_window_groups), data_source.outer_subjects
    )

    micro_fit_subjects: frozenset[str] = frozenset()
    micro_selection = None
    micro_oof_seconds = 0.0
    if config.micro_enabled:
        for name in ("micro_window_train", "micro_candidate_train", "micro_validation"):
            batch = getattr(data_source, name)
            if batch is None:
                raise ValueError(f"{name} is required when micro_enabled=True")
            _validate_batch(batch, name)
            if batch.features.shape[1] != 47:
                raise ValueError(f"{name}.features must have 47 columns")
        micro_train = data_source.micro_window_train
        micro_candidates_batch = data_source.micro_candidate_train
        micro_validation = data_source.micro_validation
        micro_groups = _groups_for(micro_train.windows, data_source.subject_by_session)
        micro_candidate_groups = _groups_for(
            micro_candidates_batch.windows, data_source.subject_by_session
        )
        micro_validation_groups = _groups_for(
            micro_validation.windows, data_source.subject_by_session
        )
        validate_outer_isolation(frozenset(micro_groups), data_source.outer_subjects)
        validate_outer_isolation(frozenset(micro_candidate_groups), data_source.outer_subjects)
        if frozenset(micro_validation_groups) != data_source.outer_subjects:
            raise ValueError("micro validation does not match outer subject manifest")
        micro_keep = _micro_training_keep(
            micro_train, data_source.train_truths, config.micro_positive_middle_fraction
        )
        micro_train_labels = np.asarray(micro_train.labels[micro_keep], dtype=np.int8)
        _validate_binary(micro_train_labels, "micro window training")
        micro_train_features = micro_train.features[micro_keep]
        micro_train_groups = micro_groups[micro_keep]
        micro_fit_subjects = frozenset(micro_train_groups)

    keep = np.asarray(data_source.window_train.labels) >= 0
    train_labels = np.asarray(data_source.window_train.labels[keep], dtype=np.int8)
    _validate_binary(train_labels, "window training")
    train_features = _with_time_prior(
        data_source.window_train.features,
        data_source.window_train.windows,
    )[keep]
    train_groups = window_groups[keep]
    candidate_features = _with_time_prior(
        data_source.candidate_train.features,
        data_source.candidate_train.windows,
    )

    stage_started = time.perf_counter()
    window_oof = crossfit_predict_proba(
        train_features,
        train_labels,
        train_groups,
        candidate_features,
        candidate_window_groups,
        config.inner_splits,
        estimator_factory=lambda: _window_estimator(config.seed),
    )
    oof_windows = _windows_by_session(
        data_source.candidate_train.windows, window_oof.probabilities
    )
    raw_train_candidates = density_candidates(oof_windows, config.density)
    if config.micro_enabled:
        window_oof_seconds = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        micro_oof = crossfit_predict_proba(
            micro_train_features, micro_train_labels, micro_train_groups,
            micro_candidates_batch.features, micro_candidate_groups,
            config.inner_splits,
            estimator_factory=lambda: _micro_window_estimator(config.seed + 10),
        )
        micro_oof_windows = _windows_by_session(
            micro_candidates_batch.windows, micro_oof.probabilities
        )
        micro_selection = select_micro_candidate_threshold(
            micro_oof_windows, data_source.train_truths,
            config.micro_threshold_grid, config.micro_candidate,
        )
        raw_micro_train_candidates = micro_candidates(
            micro_oof_windows, micro_selection.threshold, config.micro_candidate
        )
        train_candidates = union_candidates(raw_train_candidates, raw_micro_train_candidates)
        train_candidate_features = multiscale_verifier_features(
            train_candidates, oof_windows, micro_oof_windows
        )
        micro_oof_seconds = time.perf_counter() - stage_started
    else:
        train_candidates, train_candidate_features = _candidate_matrix(
            raw_train_candidates,
            oof_windows,
            include_coverage=config.density.coverage_fix,
            window_batch=data_source.candidate_train,
            feature_mode=config.verifier_feature_mode,
        )
    if not train_candidates:
        raise ValueError("inner window OOF produced no verifier candidates")
    train_candidate_labels = _candidate_labels(
        train_candidates, data_source.train_truths
    )
    _validate_binary(train_candidate_labels, "verifier training")
    train_candidate_groups = _groups_for(
        [candidate.event for candidate in train_candidates],
        data_source.subject_by_session,
    )
    verifier_fit_subjects = frozenset(train_candidate_groups)
    validate_outer_isolation(verifier_fit_subjects, data_source.outer_subjects)
    if not config.micro_enabled:
        window_oof_seconds = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    train_candidate_events = [candidate.event for candidate in train_candidates]
    best_verifier_rank: tuple[float, float, float] | None = None
    selected_c: float | None = None
    policy: EventSelectionPolicy | None = None
    for regularization_c in config.verifier_c_grid:
        verifier_oof = crossfit_predict_proba(
            train_candidate_features,
            train_candidate_labels,
            train_candidate_groups,
            train_candidate_features,
            train_candidate_groups,
            config.inner_splits,
            estimator_factory=lambda c=regularization_c: _verifier_estimator(
                config.seed + 1, c
            ),
        )
        if config.subject_cap_grid:
            candidate_policy = select_event_policy(
                train_candidate_events,
                verifier_oof.probabilities,
                data_source.train_truths,
                train_candidate_groups,
                max_events_options=config.subject_cap_grid,
            )
        else:
            threshold_selection = select_event_threshold(
                train_candidate_events,
                verifier_oof.probabilities,
                data_source.train_truths,
            )
            candidate_policy = EventSelectionPolicy(
                threshold_selection.threshold,
                None,
                threshold_selection.metrics,
            )
        rank = (
            candidate_policy.metrics.f1,
            candidate_policy.metrics.ppv,
            -float(regularization_c),
        )
        if best_verifier_rank is None or rank > best_verifier_rank:
            best_verifier_rank = rank
            selected_c = float(regularization_c)
            policy = candidate_policy
    assert selected_c is not None and policy is not None
    verifier_oof_seconds = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    window_model = _window_estimator(config.seed + 2)
    window_model.fit(train_features, train_labels)
    if config.micro_enabled:
        micro_model = _micro_window_estimator(config.seed + 12)
        micro_model.fit(micro_train_features, micro_train_labels)
    verifier_model = _verifier_estimator(config.seed + 3, selected_c)
    verifier_model.fit(train_candidate_features, train_candidate_labels)
    fit_seconds = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    validation_features = _with_time_prior(
        data_source.validation.features, data_source.validation.windows
    )
    validation_window_scores = _positive_probability(
        window_model, validation_features
    )
    validation_windows = _windows_by_session(
        data_source.validation.windows, validation_window_scores
    )
    raw_validation_candidates = density_candidates(
        validation_windows, config.density
    )
    raw_micro_validation_candidates = []
    if config.micro_enabled:
        micro_validation_scores = _positive_probability(micro_model, micro_validation.features)
        micro_validation_windows = _windows_by_session(
            micro_validation.windows, micro_validation_scores
        )
        raw_micro_validation_candidates = micro_candidates(
            micro_validation_windows, micro_selection.threshold, config.micro_candidate
        )
        validation_candidates = union_candidates(
            raw_validation_candidates, raw_micro_validation_candidates
        )
        validation_candidate_features = multiscale_verifier_features(
            validation_candidates, validation_windows, micro_validation_windows
        )
    else:
        validation_candidates, validation_candidate_features = _candidate_matrix(
            raw_validation_candidates,
            validation_windows,
            include_coverage=config.density.coverage_fix,
            window_batch=data_source.validation,
            feature_mode=config.verifier_feature_mode,
        )
    if validation_candidates:
        validation_scores = _positive_probability(
            verifier_model, validation_candidate_features
        )
    else:
        validation_scores = np.empty(0, dtype=np.float64)
    validation_candidate_events = [
        candidate.event for candidate in validation_candidates
    ]
    validation_candidate_groups = _groups_for(
        validation_candidate_events,
        data_source.subject_by_session,
    )
    selected_predictions = apply_event_policy(
        validation_candidate_events,
        validation_scores,
        validation_candidate_groups,
        threshold=policy.threshold,
        max_events_per_group=policy.max_events_per_group,
    )
    outer_metrics = compute_event_metrics(
        selected_predictions, data_source.validation_truths
    )
    candidate_metrics = compute_event_metrics(
        [candidate.event for candidate in validation_candidates],
        data_source.validation_truths,
    )
    slices = {}
    for name, truths in data_source.validation_truth_slices.items():
        session_ids = {truth.sid for truth in truths}
        slice_predictions = [
            prediction
            for prediction in selected_predictions
            if prediction.sid in session_ids
        ]
        slices[name] = compute_event_metrics(slice_predictions, truths)
    inference_seconds = time.perf_counter() - stage_started

    micro_metrics = compute_event_metrics(
        [candidate.event for candidate in raw_micro_validation_candidates],
        data_source.validation_truths,
    ) if config.micro_enabled else None
    short_candidate_metrics = compute_event_metrics(
        validation_candidate_events,
        data_source.validation_truth_slices.get("duration_lt10", ()),
    ) if config.micro_enabled else None
    feature_dimensions = (train_features.shape[1], train_candidate_features.shape[1])
    if config.micro_enabled:
        feature_dimensions += (micro_train_features.shape[1],)
    config_hash = cache_key(
        config,
        feature_dimensions=feature_dimensions,
    )
    timings = {
        "feature_extraction": 0.0,
        "window_oof": window_oof_seconds,
        "verifier_oof": verifier_oof_seconds,
        "final_fit": fit_seconds,
        "outer_inference": inference_seconds,
        "total": time.perf_counter() - started,
    }
    if config.micro_enabled:
        timings["feature_extraction"] = data_source.micro_cache_extraction_seconds
        timings["macro_window_oof"] = timings.pop("window_oof")
        timings["micro_window_oof"] = micro_oof_seconds
    return FoldResult(
        config_hash=config_hash,
        threshold=policy.threshold,
        max_events_per_subject=policy.max_events_per_group,
        verifier_c=selected_c,
        verifier_feature_count=train_candidate_features.shape[1],
        inner_metrics=policy.metrics,
        outer_metrics=outer_metrics,
        candidate_count=len(validation_candidates),
        candidate_match_recall=candidate_metrics.sensitivity,
        slices=slices,
        timings_seconds=timings,
        cache_hits={"fold_result": False},
        outer_subjects=data_source.outer_subjects,
        window_fit_subjects=window_fit_subjects,
        verifier_fit_subjects=verifier_fit_subjects,
        macro_window_feature_count=train_features.shape[1],
        micro_threshold=micro_selection.threshold if micro_selection is not None else None,
        micro_candidate_count=len(raw_micro_validation_candidates),
        micro_candidate_match_recall=micro_metrics.sensitivity if micro_metrics else 0.0,
        short_meal_candidate_recall=short_candidate_metrics.sensitivity if short_candidate_metrics else 0.0,
        micro_window_feature_count=micro_train_features.shape[1] if config.micro_enabled else 0,
        micro_window_fit_subjects=micro_fit_subjects,
    )


class FilesystemDataSource:
    """Load the existing slide NPZ artifacts once for one outer fold."""

    def __init__(self, root: Path | None = None) -> None:
        if root is None:
            from src import config as project_config

            root = project_config.ROOT_DIR
        self.root = Path(root)
        self.slide_dir = self.root / "cache" / "slide"
        self.micro_dir = self.root / "cache" / "micro15"
        self.session_dir = self.root / "cache" / "sessions"
        self.cache_directory = self.root / "cache" / "crossfit"

    def _split_path(self, fold: int, split: str) -> Path:
        return self.slide_dir / f"fold{fold}_{split}.npz"

    def _micro_split_path(self, fold: int, split: str) -> Path:
        return self.micro_dir / f"fold{fold}_{split}.npz"

    def input_files(self, config: RunConfig) -> tuple[Path, ...]:
        from src.data import manifests

        fold = config.outer_fold
        paths = [
            self._split_path(fold, split)
            for split in ("train", "meal_train", "no_meal_train", "val")
        ]
        if config.micro_enabled:
            paths.extend(
                self._micro_split_path(fold, split)
                for split in ("train", "meal_train", "no_meal_train", "val")
            )
        paths.extend((manifests.INDEX_CSV, manifests.MEALS_CSV))
        split_manifest = self.root / "cache" / "splits" / f"fold{fold}.json"
        if split_manifest.exists():
            paths.append(split_manifest)
            payload = json.loads(split_manifest.read_text(encoding="utf-8"))
            session_ids = list(payload.get("train_sessions", ())) + list(
                payload.get("val_sessions", ())
            )
            for sid in session_ids:
                session_path = self.session_dir / f"{sid}.npz"
                if session_path.exists():
                    paths.append(session_path)
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError("missing runner inputs: " + ", ".join(missing))
        return tuple(sorted(set(paths), key=lambda path: str(path)))

    @staticmethod
    def _load_batch(path: Path) -> WindowBatch:
        with np.load(path, allow_pickle=True) as data:
            features = np.asarray(data["feat"]).copy()
            labels = np.asarray(data["label"]).copy()
            windows = tuple(
                EventRef(str(sid), int(start), int(end))
                for sid, start, end in (json.loads(str(value)) for value in data["wid"])
            )
        return WindowBatch(features, labels, windows)

    @staticmethod
    def _load_micro_batch(path: Path, expected_metadata: Mapping[str, object]) -> tuple[WindowBatch, float]:
        from src.pipeline.micro_cache import read_micro_cache, read_micro_metadata

        arrays = read_micro_cache(path, expected_metadata)
        windows = tuple(
            EventRef(str(sid), int(start), int(end))
            for sid, start, end in (json.loads(str(value)) for value in arrays.wid)
        )
        return (
            WindowBatch(arrays.feat, arrays.label, windows),
            float(read_micro_metadata(path)["extraction_seconds"]),
        )

    def _micro_expected_metadata(self, config: RunConfig, split: str) -> dict:
        from src.pipeline.imu_features import MicroFeatureConfig
        from src.pipeline.micro_cache import cache_metadata, split_sessions

        sessions = split_sessions(self.root, config.outer_fold, split)
        source_files = [self.session_dir / f"{sid}.npz" for sid in sessions]
        source_files.append(self.root / "cache" / "splits" / f"fold{config.outer_fold}.json")
        return cache_metadata(
            MicroFeatureConfig(gravity_align=config.micro_gravity_align), source_files
        )

    @staticmethod
    def _combine_batches(*batches: WindowBatch) -> WindowBatch:
        return WindowBatch(
            features=np.concatenate([batch.features for batch in batches]),
            labels=np.concatenate([batch.labels for batch in batches]),
            windows=tuple(
                window for batch in batches for window in batch.windows
            ),
        )

    def _eligible_truths(
        self,
        session_ids: set[str],
        window_batch: WindowBatch,
        subject_by_session: Mapping[str, str],
    ) -> tuple[tuple[EventRef, ...], dict[str, tuple[EventRef, ...]]]:
        from src.data import manifests

        index = manifests.load_sensor_index()
        meal_meta, _ = manifests.load_meal_meta()
        index_by_sid = {
            str(row["session_id"]): row for _, row in index.iterrows()
        }
        windows_by_sid: defaultdict[str, list[EventRef]] = defaultdict(list)
        for window in window_batch.windows:
            windows_by_sid[window.sid].append(window)

        truths: list[EventRef] = []
        slices: defaultdict[str, list[EventRef]] = defaultdict(list)
        for sid in sorted(session_ids):
            row = index_by_sid.get(sid)
            session_path = self.session_dir / f"{sid}.npz"
            if row is None or not session_path.exists():
                continue
            with np.load(session_path) as session:
                valid_times = session["t_acc"][session["imu_valid"]]
            subject = subject_by_session[sid]
            session_start = int(row["timeStamp.startTime"])
            session_end = int(row["timeStamp.endTime"])
            session_windows = windows_by_sid.get(sid, ())
            for meal in meal_meta.get(subject, ()):
                meal_start = int(meal["before"])
                meal_end = int(meal["after"])
                if meal_start < session_start or meal_end > session_end:
                    continue
                left = int(np.searchsorted(valid_times, meal_start))
                right = int(np.searchsorted(valid_times, meal_end))
                if right <= left:
                    continue
                covered_span = int(
                    valid_times[min(right, len(valid_times) - 1)]
                    - valid_times[max(left, 0)]
                )
                if covered_span < 0.5 * (meal_end - meal_start):
                    continue
                max_window_overlap = max(
                    (
                        min(window.end_ms, meal_end)
                        - max(window.start_ms, meal_start)
                        for window in session_windows
                    ),
                    default=0,
                )
                if max_window_overlap < 120_000:
                    continue
                truth = EventRef(sid, meal_start, meal_end)
                truths.append(truth)
                scene = str(meal["scene"])
                slices[scene].append(truth)
                duration_minutes = (meal_end - meal_start) / 60_000.0
                if duration_minutes < 10:
                    slices["duration_lt10"].append(truth)
                elif duration_minutes < 20:
                    slices["duration_10_20"].append(truth)
                else:
                    slices["duration_ge20"].append(truth)
        return tuple(truths), {
            name: tuple(events) for name, events in sorted(slices.items())
        }

    def load_outer_fold(self, config: RunConfig) -> FoldDataset:
        from src.data import manifests

        fold = config.outer_fold
        window_train = self._load_batch(self._split_path(fold, "train"))
        meal_train = self._load_batch(self._split_path(fold, "meal_train"))
        no_meal_train = self._load_batch(
            self._split_path(fold, "no_meal_train")
        )
        validation = self._load_batch(self._split_path(fold, "val"))
        candidate_train = self._combine_batches(meal_train, no_meal_train)

        micro_batches: dict[str, WindowBatch] = {}
        extraction_seconds = 0.0
        if config.micro_enabled:
            for split in ("train", "meal_train", "no_meal_train", "val"):
                micro_batches[split], seconds = self._load_micro_batch(
                    self._micro_split_path(fold, split),
                    self._micro_expected_metadata(config, split),
                )
                extraction_seconds += seconds
            micro_batches["candidate_train"] = self._combine_batches(
                micro_batches["meal_train"], micro_batches["no_meal_train"]
            )
            # Macro coverage defines the frozen evaluation session universe.
            # Micro extraction can retain sessions with no eligible macro rows.
            for split, macro_batch in (
                ("train", window_train),
                ("candidate_train", candidate_train),
                ("val", validation),
            ):
                allowed_sids = {window.sid for window in macro_batch.windows}
                batch = micro_batches[split]
                keep = np.asarray(
                    [window.sid in allowed_sids for window in batch.windows], dtype=bool
                )
                micro_batches[split] = WindowBatch(
                    batch.features[keep],
                    batch.labels[keep],
                    tuple(window for window, retained in zip(batch.windows, keep) if retained),
                )

        index = manifests.load_sensor_index()
        subject_by_session = {
            str(row["session_id"]): str(row["externalid"])
            for _, row in index.iterrows()
        }
        train_sids = {window.sid for window in candidate_train.windows}
        validation_sids = {window.sid for window in validation.windows}
        train_truths, _ = self._eligible_truths(
            train_sids, candidate_train, subject_by_session
        )
        validation_truths, validation_slices = self._eligible_truths(
            validation_sids, validation, subject_by_session
        )
        outer_subjects = frozenset(
            subject_by_session[sid] for sid in validation_sids
        )
        return FoldDataset(
            window_train=window_train,
            candidate_train=candidate_train,
            validation=validation,
            train_truths=train_truths,
            validation_truths=validation_truths,
            subject_by_session=subject_by_session,
            outer_subjects=outer_subjects,
            validation_truth_slices=validation_slices,
            micro_window_train=micro_batches.get("train"),
            micro_candidate_train=micro_batches.get("candidate_train"),
            micro_validation=micro_batches.get("val"),
            micro_cache_extraction_seconds=extraction_seconds,
        )


def fold_result_to_dict(result: FoldResult) -> dict[str, object]:
    def metrics_dict(metrics: EventMetrics) -> dict[str, int | float]:
        return asdict(metrics)

    return {
        "config_hash": result.config_hash,
        "threshold": result.threshold,
        "max_events_per_subject": result.max_events_per_subject,
        "verifier_c": result.verifier_c,
        "verifier_feature_count": result.verifier_feature_count,
        "inner_metrics": metrics_dict(result.inner_metrics),
        "outer_metrics": metrics_dict(result.outer_metrics),
        "candidate_count": result.candidate_count,
        "candidate_match_recall": result.candidate_match_recall,
        "slices": {
            name: metrics_dict(metrics)
            for name, metrics in sorted(result.slices.items())
        },
        "timings_seconds": dict(result.timings_seconds),
        "cache_hits": dict(result.cache_hits),
        "outer_subjects": sorted(result.outer_subjects),
        "window_fit_subjects": sorted(result.window_fit_subjects),
        "verifier_fit_subjects": sorted(result.verifier_fit_subjects),
        "micro_threshold": result.micro_threshold,
        "micro_candidate_count": result.micro_candidate_count,
        "micro_candidate_match_recall": result.micro_candidate_match_recall,
        "short_meal_candidate_recall": result.short_meal_candidate_recall,
        "external_weight": result.external_weight,
        "macro_window_feature_count": result.macro_window_feature_count,
        "micro_window_feature_count": result.micro_window_feature_count,
        "micro_window_fit_subjects": sorted(result.micro_window_fit_subjects),
    }


def _fold_result_from_dict(payload: Mapping[str, object]) -> FoldResult:
    def metrics(value: object) -> EventMetrics:
        if not isinstance(value, Mapping):
            raise ValueError("invalid cached metrics")
        return EventMetrics(**value)

    slice_payload = payload.get("slices", {})
    if not isinstance(slice_payload, Mapping):
        raise ValueError("invalid cached slices")
    return FoldResult(
        config_hash=str(payload["config_hash"]),
        threshold=float(payload["threshold"]),
        max_events_per_subject=(
            int(payload["max_events_per_subject"])
            if payload.get("max_events_per_subject") is not None
            else None
        ),
        verifier_c=float(payload.get("verifier_c", 0.1)),
        verifier_feature_count=int(payload.get("verifier_feature_count", 37)),
        inner_metrics=metrics(payload["inner_metrics"]),
        outer_metrics=metrics(payload["outer_metrics"]),
        candidate_count=int(payload["candidate_count"]),
        candidate_match_recall=float(payload["candidate_match_recall"]),
        slices={name: metrics(value) for name, value in slice_payload.items()},
        timings_seconds={
            str(name): float(value)
            for name, value in dict(payload.get("timings_seconds", {})).items()
        },
        cache_hits={
            str(name): bool(value)
            for name, value in dict(payload.get("cache_hits", {})).items()
        },
        outer_subjects=frozenset(payload.get("outer_subjects", ())),
        window_fit_subjects=frozenset(payload.get("window_fit_subjects", ())),
        verifier_fit_subjects=frozenset(payload.get("verifier_fit_subjects", ())),
        micro_threshold=(float(payload["micro_threshold"]) if payload.get("micro_threshold") is not None else None),
        micro_candidate_count=int(payload.get("micro_candidate_count", 0)),
        micro_candidate_match_recall=float(payload.get("micro_candidate_match_recall", 0.0)),
        short_meal_candidate_recall=float(payload.get("short_meal_candidate_recall", 0.0)),
        external_weight=float(payload.get("external_weight", 0.0)),
        macro_window_feature_count=int(payload.get("macro_window_feature_count", 63)),
        micro_window_feature_count=int(payload.get("micro_window_feature_count", 0)),
        micro_window_fit_subjects=frozenset(payload.get("micro_window_fit_subjects", ())),
    )


def experiment_key(configs: Sequence[RunConfig]) -> str:
    """Identify registered settings independently of fold numbers and outcomes."""
    normalized = sorted(
        json.dumps(asdict(replace(config, outer_fold=-1)), sort_keys=True, separators=(",", ":"))
        for config in configs
    )
    canonical = json.dumps(normalized, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def aggregate_fold_results(
    configs: Sequence[RunConfig], results: Sequence[FoldResult]
) -> dict[str, object]:
    """Pool event counts and truth-weight candidate recall across outer folds."""
    if not configs or len(configs) != len(results):
        raise ValueError("configs and results must have equal nonzero lengths")
    if len({config.outer_fold for config in configs}) != len(configs):
        raise ValueError("duplicate outer folds are not allowed")

    def pooled_metrics(metrics: Sequence[EventMetrics]) -> dict[str, int | float]:
        n_tp = sum(item.n_tp for item in metrics)
        n_pred = sum(item.n_pred for item in metrics)
        n_true = sum(item.n_true for item in metrics)
        return asdict(EventMetrics(
            n_tp=n_tp, n_pred=n_pred, n_true=n_true,
            sensitivity=n_tp / n_true if n_true else 0.0,
            ppv=n_tp / n_pred if n_pred else 0.0,
            f1=2 * n_tp / (n_pred + n_true) if n_pred + n_true else 0.0,
        ))

    truth_counts = [result.outer_metrics.n_true for result in results]
    short_counts = [
        result.slices["duration_lt10"].n_true if "duration_lt10" in result.slices else 0
        for result in results
    ]

    def weighted_recall(field_name: str, counts: Sequence[int]) -> float:
        denominator = sum(counts)
        return (
            sum(getattr(result, field_name) * count for result, count in zip(results, counts)) / denominator
            if denominator else 0.0
        )

    slice_names = sorted({name for result in results for name in result.slices})
    timing_names = sorted({name for result in results for name in result.timings_seconds})
    return {
        "inner_metrics": pooled_metrics([result.inner_metrics for result in results]),
        "outer_metrics": pooled_metrics([result.outer_metrics for result in results]),
        "candidate_count": sum(result.candidate_count for result in results),
        "candidate_match_recall": weighted_recall("candidate_match_recall", truth_counts),
        "micro_candidate_count": sum(result.micro_candidate_count for result in results),
        "micro_candidate_match_recall": weighted_recall("micro_candidate_match_recall", truth_counts),
        "short_meal_candidate_recall": weighted_recall("short_meal_candidate_recall", short_counts),
        "slices": {
            name: pooled_metrics([result.slices[name] for result in results if name in result.slices])
            for name in slice_names
        },
        "timings_seconds": {
            name: sum(result.timings_seconds.get(name, 0.0) for result in results)
            for name in timing_names
        },
        "folds": [
            {"outer_fold": config.outer_fold, "config_hash": result.config_hash,
             "micro_threshold": result.micro_threshold}
            for config, result in sorted(zip(configs, results), key=lambda pair: pair[0].outer_fold)
        ],
    }


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_outer_fold(
    config: RunConfig,
    data_source: FoldDataset | FilesystemDataSource | None = None,
    force: bool = False,
) -> FoldResult:
    """Resolve a data source, reuse a valid cache, and run one nested fold."""

    if config.external_fd_weight_grid != (0.0,):
        raise ValueError("external_fd_weight_grid must be (0.0,) in the target-domain runner")
    if isinstance(data_source, FoldDataset):
        return _run_outer_dataset(config, data_source)
    source = data_source or FilesystemDataSource()
    if not isinstance(source, FilesystemDataSource):
        raise TypeError("data_source must be FoldDataset or FilesystemDataSource")
    input_files = source.input_files(config)
    verifier_dimensions = 42 if config.density.coverage_fix else 37
    if config.verifier_feature_mode == "raw_summary":
        verifier_dimensions += 312
    dimensions = (63, verifier_dimensions)
    if config.micro_enabled:
        dimensions = (63, 56, 47)
    key = cache_key(config, dimensions, input_files)
    cached_path = source.cache_directory / f"fold{config.outer_fold}_{key}.json"
    if cached_path.exists() and not force:
        cached = _fold_result_from_dict(
            json.loads(cached_path.read_text(encoding="utf-8"))
        )
        return replace(cached, cache_hits={"fold_result": True})

    dataset = source.load_outer_fold(config)
    result = replace(
        _run_outer_dataset(config, dataset),
        config_hash=key,
        cache_hits={"fold_result": False},
    )
    write_json_atomic(cached_path, fold_result_to_dict(result))
    return result


def _run_outer_fold_limited(config: RunConfig, force: bool) -> FoldResult:
    with threadpool_limits(limits=1):
        return run_outer_fold(config, force=force)


def run_folds(
    configs: Sequence[RunConfig], workers: int = 1, force: bool = False
) -> list[FoldResult]:
    """Run folds inline or in bounded processes without BLAS oversubscription."""

    configs = tuple(configs)
    if not configs:
        return []
    if workers < 0:
        raise ValueError("workers must be non-negative")
    if workers == 0:
        workers = min(_physical_cpu_count(), len(configs))
    workers = min(workers, len(configs))
    if workers == 1:
        return [run_outer_fold(config, force=force) for config in configs]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(
            executor.map(
                _run_outer_fold_limited,
                configs,
                [force] * len(configs),
            )
        )
