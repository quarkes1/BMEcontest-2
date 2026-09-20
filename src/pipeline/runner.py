"""Leakage-safe nested runner for the event verification stack."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from numbers import Integral, Real
from pathlib import Path
from typing import Callable, Mapping, Sequence

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
from src.pipeline.candidate_control import (
    CandidateAdmissionConfig,
    admit_candidates,
)
from src.pipeline.crossfit import crossfit_predict_proba
from src.pipeline.event_stack import (
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
from src.pipeline.context_features import CONTEXT_V1_COLUMNS
from src.pipeline.context_features_v2 import CONTEXT_V2_COLUMNS
from src.pipeline.diagnostics import aggregate_subject_diagnostics, subject_diagnostics


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
VERIFIER_LGBM_PARAMETERS = {
    "n_estimators": 200,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 40,
    "learning_rate": 0.03,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "class_weight": "balanced",
    "n_jobs": 1,
    "verbosity": -1,
}
RUNNER_SCHEMA_VERSION = 5


def _validate_registered_probability_grid(
    values: tuple[float, ...], name: str, *, lower_exclusive: bool = False
) -> tuple[float, ...]:
    """Reject noncanonical stacked-control probability grids at the boundary."""
    if not values:
        raise ValueError(f"{name} must be a nonempty grid")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        for value in values
    ):
        raise ValueError(f"{name} must contain real non-boolean values")
    normalized = tuple(float(value) for value in values)
    if any(
        not math.isfinite(value)
        or value > 1.0
        or (value <= 0.0 if lower_exclusive else value < 0.0)
        for value in normalized
    ):
        interval = "(0, 1]" if lower_exclusive else "[0, 1]"
        raise ValueError(f"{name} must contain finite values in {interval}")
    if len(set(normalized)) != len(normalized) or tuple(sorted(normalized)) != normalized:
        raise ValueError(f"{name} must be unique and sorted")
    return normalized


def _validate_registered_subject_cap_grid(values: tuple[int, ...]) -> tuple[int, ...]:
    if not values:
        raise ValueError("admission_subject_cap_grid must be a nonempty grid")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < 1
        for value in values
    ):
        raise ValueError("admission_subject_cap_grid must contain positive integers")
    if len(set(values)) != len(values) or tuple(sorted(values)) != values:
        raise ValueError("admission_subject_cap_grid must be unique and sorted")
    return tuple(int(value) for value in values)


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
    context_features_version: str | None = None
    micro_gravity_align: bool = True
    micro_threshold_grid: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50)
    micro_candidate: MicroCandidateConfig = field(default_factory=MicroCandidateConfig)
    micro_positive_middle_fraction: float | None = None
    external_fd_weight_grid: tuple[float, ...] = (0.0,)
    candidate_control_enabled: bool = False
    admission_nms_iou_grid: tuple[float, ...] = (0.3, 0.5, 0.7)
    admission_threshold_grid: tuple[float, ...] = (0.2, 0.35, 0.5, 0.65)
    admission_subject_cap_grid: tuple[int, ...] = (3, 4, 5, 6, 8)
    verifier_blend_weight_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    admission_minimum_recall: float = 0.88

    def __post_init__(self) -> None:
        if self.context_features_version not in (None, "v1", "v2"):
            raise ValueError("context_features_version must be None, 'v1', or 'v2'")
        if self.context_features_version == "v2":
            if not self.micro_enabled or not self.candidate_control_enabled:
                raise ValueError("context_features_version='v2' requires micro and candidate control")
            if self.admission_minimum_recall != 0.80:
                raise ValueError("v2 admission_minimum_recall must be exactly 0.80")
        if not isinstance(self.candidate_control_enabled, bool):
            raise ValueError("candidate_control_enabled must be boolean")
        if self.candidate_control_enabled and not self.micro_enabled:
            raise ValueError("candidate_control_enabled requires micro_enabled")
        object.__setattr__(
            self,
            "admission_nms_iou_grid",
            _validate_registered_probability_grid(
                self.admission_nms_iou_grid,
                "admission_nms_iou_grid",
                lower_exclusive=True,
            ),
        )
        object.__setattr__(
            self,
            "admission_threshold_grid",
            _validate_registered_probability_grid(
                self.admission_threshold_grid, "admission_threshold_grid"
            ),
        )
        object.__setattr__(
            self,
            "admission_subject_cap_grid",
            _validate_registered_subject_cap_grid(self.admission_subject_cap_grid),
        )
        object.__setattr__(
            self,
            "verifier_blend_weight_grid",
            _validate_registered_probability_grid(
                self.verifier_blend_weight_grid, "verifier_blend_weight_grid"
            ),
        )
        if (
            isinstance(self.admission_minimum_recall, (bool, np.bool_))
            or not isinstance(self.admission_minimum_recall, Real)
            or not math.isfinite(float(self.admission_minimum_recall))
            or not 0.0 <= float(self.admission_minimum_recall) <= 1.0
        ):
            raise ValueError("admission_minimum_recall must be finite in [0, 1]")
        object.__setattr__(
            self, "admission_minimum_recall", float(self.admission_minimum_recall)
        )


def expected_feature_dimensions(
    config: RunConfig, *, macro_width: int = 63
) -> tuple[int, ...]:
    """Return fitted-model widths for one run configuration.

    Filesystem runs use the registered 63-column macro surface. In-memory
    fixtures can supply their already-derived macro width.
    """

    if config.micro_enabled:
        verifier = 56 + (
            (len(CONTEXT_V1_COLUMNS) if config.context_features_version in ("v1", "v2") else 0)
            + (len(CONTEXT_V2_COLUMNS) if config.context_features_version == "v2" else 0)
        )
        return macro_width, verifier, 47
    verifier = 42 if config.density.coverage_fix else 37
    if config.verifier_feature_mode == "raw_summary":
        # ``aggregate_candidate_features`` emits five statistics per raw
        # input column plus two sample-count fields. ``macro_width`` includes
        # the runner's one-column time prior.
        verifier += (macro_width - 1) * 5 + 2
    return macro_width, verifier


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
    selected_blend_weight: float | None = None
    selected_admission_nms_iou: float | None = None
    selected_admission_threshold: float | None = None
    selected_admission_subject_cap: int | None = None
    raw_union_candidate_count: int = 0
    admitted_candidate_count: int = 0
    verifier_logistic_oof_seconds: float = 0.0
    verifier_lgbm_oof_seconds: float = 0.0
    verifier_logistic_outer_seconds: float = 0.0
    verifier_lgbm_outer_seconds: float = 0.0
    subject_diagnostics: Mapping[str, object] = field(default_factory=dict)
    runtime_diagnostics: Mapping[str, object] = field(default_factory=dict)


def peak_working_set_bytes() -> int | None:
    """Return the Windows process peak working set, or ``None`` if unavailable."""

    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        # Without explicit ctypes signatures, the pseudo-handle returned by
        # GetCurrentProcess is truncated to a signed 32-bit integer on 64-bit
        # Python, so Psapi rejects it.  Bind the documented HANDLE/pointer/DWORD
        # signature before making the call.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        get_process_memory_info = psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
            wintypes.DWORD,
        )
        get_process_memory_info.restype = wintypes.BOOL
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(counters)
        ok = get_process_memory_info(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        return int(counters.PeakWorkingSetSize) if ok else None
    except (AttributeError, OSError):
        return None


def _runtime_diagnostics(
    peak_provider: Callable[[], int | None] | None = None,
) -> dict[str, object]:
    """Record runtime evidence without inventing unavailable measurements."""

    peak = (peak_provider or peak_working_set_bytes)()
    return {
        "peak_working_set_bytes": peak,
        "unavailable_reason": (
            None
            if peak is not None
            else (
                "Windows GetProcessMemoryInfo is unavailable"
                if os.name == "nt"
                else "Windows GetProcessMemoryInfo is unavailable on this platform"
            )
        ),
        "cuda_peak_bytes": None,
        "ssl_runtime": None,
    }


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
    # Filesystem datasets carry index.csv's authoritative acquisition bounds.
    # ``None`` is only the in-memory test-fixture compatibility path.
    session_bounds_by_sid: Mapping[str, tuple[int, int]] | None = None
    validation_truth_slices: Mapping[str, tuple[EventRef, ...]] = field(
        default_factory=dict
    )
    micro_window_train: WindowBatch | None = None
    micro_candidate_train: WindowBatch | None = None
    micro_validation: WindowBatch | None = None
    micro_cache_extraction_seconds: float = 0.0


@dataclass(frozen=True)
class OuterTrainOnlyData:
    """The strictly outer-train-only payload used for E02 inner selection.

    Deliberately has no validation windows, labels, truth, or subject fields.
    """

    window_train: WindowBatch
    candidate_train: WindowBatch
    train_truths: tuple[EventRef, ...]
    subject_by_session: Mapping[str, str]
    session_bounds_by_sid: Mapping[str, tuple[int, int]] | None = None
    micro_window_train: WindowBatch | None = None
    micro_candidate_train: WindowBatch | None = None
    micro_cache_extraction_seconds: float = 0.0


@dataclass(frozen=True)
class InnerOOFResult:
    """Train-only policy-selection evidence; no outer score is representable."""

    inner_metrics: EventMetrics
    verifier_feature_count: int
    window_fit_subjects: frozenset[str]
    verifier_fit_subjects: frozenset[str]
    micro_window_fit_subjects: frozenset[str]


def _concatenate_window_batches(batches: Sequence[WindowBatch]) -> WindowBatch:
    """Join batches only after callers have established their partition invariants."""

    if not batches:
        raise ValueError("at least one window batch is required")
    widths = {batch.features.shape[1] for batch in batches}
    if len(widths) != 1:
        raise ValueError("full-target window batches must share one feature width")
    return WindowBatch(
        features=np.concatenate([batch.features for batch in batches]),
        labels=np.concatenate([batch.labels for batch in batches]),
        windows=tuple(window for batch in batches for window in batch.windows),
    )


def _event_key(event: EventRef) -> tuple[str, int, int]:
    return event.sid, event.start_ms, event.end_ms


def _validate_imputable_features(
    values: np.ndarray, name: str, *, allow_all_missing_columns: bool = False
) -> None:
    """Accept NaN only where the registered ``SimpleImputer`` can consume it."""

    try:
        numeric = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values or NaN") from exc
    if np.isinf(numeric).any():
        raise ValueError(f"{name} must not contain +/-Inf values")
    if not allow_all_missing_columns and np.isnan(numeric).all(axis=0).any():
        raise ValueError(
            f"{name} must not contain all-missing columns; the registered "
            "SimpleImputer would drop them"
        )


def _validate_tristate_labels(values: np.ndarray, name: str) -> None:
    """Enforce the documented -1/0/1 label contract before filtering -1 rows."""

    try:
        numeric = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite tri-state labels") from exc
    if not np.isfinite(numeric).all() or not np.isin(numeric, (-1.0, 0.0, 1.0)).all():
        raise ValueError(f"{name} must contain finite tri-state labels (-1, 0, 1)")


def _validate_full_target_batch(batch: WindowBatch, name: str, width: int) -> set[str]:
    _validate_batch(batch, name)
    if batch.features.shape[1] != width:
        raise ValueError(f"{name}.features must have {width} columns")
    _validate_imputable_features(batch.features, f"{name}.features")
    _validate_tristate_labels(batch.labels, f"{name}.labels")
    if any(not isinstance(window, EventRef) or window.end_ms <= window.start_ms for window in batch.windows):
        raise ValueError(f"{name}.windows must contain valid EventRef intervals")
    return {window.sid for window in batch.windows}


def _validate_full_target_partition(partition: FoldDataset) -> tuple[set[str], set[str]]:
    """Validate one raw held-out partition before it joins the deployment union."""

    macro_sessions = _validate_full_target_batch(partition.validation, "validation", 62)
    missing_sessions = macro_sessions - set(partition.subject_by_session)
    if missing_sessions:
        raise ValueError("validation sessions absent from subject mapping: " + ", ".join(sorted(missing_sessions)))
    macro_subjects = {partition.subject_by_session[sid] for sid in macro_sessions}
    if macro_subjects != set(partition.outer_subjects):
        raise ValueError("held-out subject mapping does not match validation windows")

    truths = tuple(partition.validation_truths)
    truth_keys = [_event_key(truth) for truth in truths]
    if len(set(truth_keys)) != len(truth_keys):
        raise ValueError("duplicate validation truth")
    if any(truth.sid not in macro_sessions for truth in truths):
        raise ValueError("held-out truth is not in its validation session universe")
    truth_set = set(truth_keys)
    for name, slice_truths in partition.validation_truth_slices.items():
        keys = [_event_key(truth) for truth in slice_truths]
        if len(set(keys)) != len(keys):
            raise ValueError(f"duplicate truth in slice {name}")
        if not set(keys).issubset(truth_set):
            raise ValueError(f"slice truth is not a member of validation truths: {name}")

    micro = partition.micro_validation
    if micro is not None:
        micro_sessions = _validate_full_target_batch(micro, "micro_validation", 47)
        if micro_sessions != macro_sessions:
            raise ValueError("micro validation session universe does not match macro")
        if {partition.subject_by_session[sid] for sid in micro_sessions} != macro_subjects:
            raise ValueError("micro validation subject mapping does not match macro")
    return macro_sessions, macro_subjects


def _validate_full_target_dataset(data: FoldDataset) -> None:
    """Revalidate the deployment union before its models are fit."""

    for name in ("window_train", "candidate_train", "validation"):
        sessions = _validate_full_target_batch(getattr(data, name), name, 62)
        if sessions != {window.sid for window in data.validation.windows}:
            raise ValueError(f"{name} session universe does not match full-target validation")
    micro_names = ("micro_window_train", "micro_candidate_train", "micro_validation")
    micro_batches = [getattr(data, name) for name in micro_names]
    if any(batch is None for batch in micro_batches) and not all(batch is None for batch in micro_batches):
        raise ValueError("full-target data must be either entirely micro-enabled or entirely macro-only")
    for name, batch in zip(micro_names, micro_batches):
        if batch is None:
            continue
        sessions = _validate_full_target_batch(batch, name, 47)
        if sessions != {window.sid for window in data.validation.windows}:
            raise ValueError(f"{name} session universe does not match full-target validation")
    _validate_full_target_partition(data)


def build_full_target_dataset(
    configs: Sequence[RunConfig],
    source: object,
    *,
    expected_truth_count: int = 153,
) -> FoldDataset:
    """Build the deployment universe from the five disjoint held-out partitions.

    The source is intentionally used only for ``validation``/``micro_validation``
    fields.  It may load an outer fold internally, but no outer-train batch is
    copied into the returned all-target data set.  This is the boundary that
    prevents accidental five-fold train-set concatenation during promotion.
    """

    ordered = tuple(sorted(configs, key=lambda item: item.outer_fold))
    if not ordered or len({item.outer_fold for item in ordered}) != len(ordered):
        raise ValueError("full-target construction requires unique outer folds")
    loader = getattr(source, "load_outer_fold", None)
    if not callable(loader):
        raise TypeError("full-target source must expose load_outer_fold")
    partitions = tuple(loader(config) for config in ordered)
    if not all(isinstance(item, FoldDataset) for item in partitions):
        raise TypeError("full-target source must return FoldDataset partitions")

    seen_subjects: set[str] = set()
    seen_sessions: set[str] = set()
    seen_truths: set[tuple[str, int, int]] = set()
    subject_by_session: dict[str, str] = {}
    session_bounds_by_sid: dict[str, tuple[int, int]] | None = {}
    for partition in partitions:
        validation_sids, validation_subjects = _validate_full_target_partition(partition)
        overlap = seen_subjects & validation_subjects
        if overlap:
            raise ValueError("held-out subject overlap: " + ", ".join(sorted(overlap)))
        session_overlap = seen_sessions & validation_sids
        if session_overlap:
            raise ValueError("held-out session overlap: " + ", ".join(sorted(session_overlap)))
        for sid in validation_sids:
            subject = partition.subject_by_session[sid]
            existing = subject_by_session.setdefault(sid, subject)
            if existing != subject:
                raise ValueError("session-to-subject mapping changed across held-out partitions")
        if session_bounds_by_sid is not None:
            if partition.session_bounds_by_sid is None:
                session_bounds_by_sid = None
            else:
                missing_bounds = validation_sids - set(partition.session_bounds_by_sid)
                if missing_bounds:
                    raise ValueError(
                        "held-out validation sessions are missing authoritative bounds: "
                        + ", ".join(sorted(missing_bounds))
                    )
                session_bounds_by_sid.update(
                    {
                        sid: partition.session_bounds_by_sid[sid]
                        for sid in validation_sids
                    }
                )
        for truth in partition.validation_truths:
            key = (truth.sid, truth.start_ms, truth.end_ms)
            if key in seen_truths:
                raise ValueError("held-out truth overlap")
            seen_truths.add(key)
        seen_subjects.update(validation_subjects)
        seen_sessions.update(validation_sids)

    if len(seen_truths) != expected_truth_count:
        raise ValueError(
            f"held-out truth count must be exactly {expected_truth_count}; got {len(seen_truths)}"
        )
    validation = _concatenate_window_batches([item.validation for item in partitions])
    micro_batches = [item.micro_validation for item in partitions]
    if any(batch is None for batch in micro_batches):
        micro_validation = None
    elif all(batch is not None for batch in micro_batches):
        micro_validation = _concatenate_window_batches(
            [batch for batch in micro_batches if batch is not None]
        )
        if {window.sid for window in micro_validation.windows} != seen_sessions:
            raise ValueError("full-target micro validation session universe does not match macro")
    else:
        raise ValueError("full-target partitions mix micro-enabled and macro-only data")
    truths = tuple(
        truth for partition in partitions for truth in partition.validation_truths
    )
    slices: defaultdict[str, list[EventRef]] = defaultdict(list)
    for partition in partitions:
        for name, values in partition.validation_truth_slices.items():
            slices[name].extend(values)
    full_target = FoldDataset(
        window_train=validation,
        candidate_train=validation,
        validation=validation,
        train_truths=truths,
        validation_truths=truths,
        subject_by_session=subject_by_session,
        outer_subjects=frozenset(seen_subjects),
        session_bounds_by_sid=session_bounds_by_sid,
        validation_truth_slices={name: tuple(values) for name, values in sorted(slices.items())},
        micro_window_train=micro_validation,
        micro_candidate_train=micro_validation,
        micro_validation=micro_validation,
    )
    _validate_full_target_dataset(full_target)
    return full_target


@dataclass(frozen=True)
class _FinalModelFit:
    """Private fitted state retained during one outer-fold evaluation.

    Keeping model objects with their train-only selected decoding policy makes
    artifact extraction possible without widening ``run_outer_fold``'s public
    ``FoldResult`` contract.  A legal full-target trainer is deliberately not
    constructed here.
    """

    macro: Pipeline
    micro: Pipeline | None
    verifier_logistic: Pipeline
    verifier_lgbm: Pipeline | None
    policy: EventSelectionPolicy
    admission: CandidateAdmissionConfig | None
    blend_weight: float | None


@dataclass(frozen=True)
class _OuterRun:
    """One evaluated fold plus the fitted train-only state used for evidence."""

    result: FoldResult
    fit: _FinalModelFit


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
        "verifier_lgbm": VERIFIER_LGBM_PARAMETERS,
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
    """Legacy model-boundary name retained for frozen caller compatibility."""
    matrix = np.asarray(features)
    if matrix.ndim != 2 or len(matrix) != len(windows):
        raise ValueError("time-prior adapter requires aligned two-dimensional features and windows")

    # The canonical primitive intentionally accepts only the release-frozen
    # 62-D macro matrix.  Historical runner callers also pass synthetic or
    # already-selected in-memory matrices, for which the old boundary helper
    # appended the same prior regardless of width.
    if matrix.shape[1] != 62:
        from src.pipeline.event_stack import _GLOBAL_PRIOR

        prior = np.asarray(
            [_GLOBAL_PRIOR[int((window.start_ms / 3.6e6) % 24)] for window in windows],
            dtype=np.float32,
        ).reshape((-1, 1))
        return np.concatenate((matrix, prior), axis=1)

    from src.pipeline.features.macro import add_time_prior

    return add_time_prior(features, windows)


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


def _verifier_lgbm_estimator(seed: int) -> Pipeline:
    """Return the registered single-threaded nonlinear candidate verifier."""
    from lightgbm import LGBMClassifier

    parameters = dict(VERIFIER_LGBM_PARAMETERS)
    parameters["random_state"] = seed
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMClassifier(**parameters)),
        ]
    )


def _fit_final_models(
    config: RunConfig,
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    micro_train_features: np.ndarray | None,
    micro_train_labels: np.ndarray | None,
    train_candidate_features: np.ndarray,
    train_candidate_labels: np.ndarray,
    verifier_c: float,
    policy: EventSelectionPolicy,
    admission: CandidateAdmissionConfig | None,
    blend_weight: float | None,
) -> _FinalModelFit:
    """Fit the frozen outer-train models after all OOF choices are final."""

    macro = _window_estimator(config.seed + 2)
    macro.fit(train_features, train_labels)
    micro = None
    if config.micro_enabled:
        if micro_train_features is None or micro_train_labels is None:
            raise ValueError("micro model inputs are required when micro_enabled=True")
        micro = _micro_window_estimator(config.seed + 12)
        micro.fit(micro_train_features, micro_train_labels)
    verifier_logistic = _verifier_estimator(config.seed + 3, verifier_c)
    verifier_logistic.fit(train_candidate_features, train_candidate_labels)
    verifier_lgbm = None
    if config.candidate_control_enabled:
        verifier_lgbm = _verifier_lgbm_estimator(config.seed + 23)
        verifier_lgbm.fit(train_candidate_features, train_candidate_labels)
    return _FinalModelFit(
        macro=macro,
        micro=micro,
        verifier_logistic=verifier_logistic,
        verifier_lgbm=verifier_lgbm,
        policy=policy,
        admission=admission,
        blend_weight=blend_weight,
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


def _test_only_session_bounds_by_sid(
    *window_groups: Sequence[EventRef],
) -> dict[str, tuple[int, int]]:
    """Compatibility bounds for synthetic in-memory fixtures only."""

    bounds: dict[str, tuple[int, int]] = {}
    for windows in window_groups:
        for window in windows:
            previous = bounds.get(window.sid)
            if previous is None:
                bounds[window.sid] = (window.start_ms, window.end_ms)
            else:
                bounds[window.sid] = (
                    min(previous[0], window.start_ms),
                    max(previous[1], window.end_ms),
                )
    return bounds


def _context_session_bounds(
    data_source: FoldDataset | OuterTrainOnlyData,
    candidates: Sequence[MultiScaleCandidate],
    *window_groups: Sequence[EventRef],
) -> Mapping[str, tuple[int, int]]:
    """Return authoritative bounds, with a clearly isolated fixture fallback."""

    if data_source.session_bounds_by_sid is None:
        return _test_only_session_bounds_by_sid(*window_groups)
    bounds = data_source.session_bounds_by_sid
    candidate_sids = {candidate.event.sid for candidate in candidates}
    missing = candidate_sids - set(bounds)
    if missing:
        raise ValueError(
            "authoritative session bounds are missing candidate sessions: "
            + ", ".join(sorted(missing))
        )
    return bounds


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


def run_outer_train_inner_oof(
    data: OuterTrainOnlyData,
    config: RunConfig,
) -> InnerOOFResult:
    """Select E02 policy from outer-train data without an outer-data handle.

    The function deliberately accepts the narrow immutable payload instead of
    ``FoldDataset``.  This makes a validation array, label, or truth
    inaccessible to all inner OOF fitting and policy selection code.
    """

    if not isinstance(data, OuterTrainOnlyData):
        raise TypeError("inner OOF requires OuterTrainOnlyData")
    if config.external_fd_weight_grid != (0.0,):
        raise ValueError("external_fd_weight_grid must be (0.0,) in the target-domain runner")
    _validate_batch(data.window_train, "window_train")
    _validate_batch(data.candidate_train, "candidate_train")
    _validate_imputable_features(data.window_train.features, "window_train.features")
    _validate_imputable_features(data.candidate_train.features, "candidate_train.features")
    _validate_tristate_labels(data.window_train.labels, "window_train.labels")
    _validate_tristate_labels(data.candidate_train.labels, "candidate_train.labels")
    window_groups = _groups_for(data.window_train.windows, data.subject_by_session)
    candidate_groups = _groups_for(data.candidate_train.windows, data.subject_by_session)
    keep = np.asarray(data.window_train.labels) >= 0
    train_labels = np.asarray(data.window_train.labels[keep], dtype=np.int8)
    _validate_binary(train_labels, "window training")
    train_features = _with_time_prior(data.window_train.features, data.window_train.windows)[keep]
    train_groups = window_groups[keep]
    candidate_features = _with_time_prior(data.candidate_train.features, data.candidate_train.windows)
    window_oof = crossfit_predict_proba(
        train_features, train_labels, train_groups, candidate_features, candidate_groups,
        config.inner_splits, estimator_factory=lambda: _window_estimator(config.seed),
    )
    macro_windows = _windows_by_session(data.candidate_train.windows, window_oof.probabilities)
    raw_macro = density_candidates(macro_windows, config.density)
    micro_fit_subjects: frozenset[str] = frozenset()
    if config.micro_enabled:
        if data.micro_window_train is None or data.micro_candidate_train is None:
            raise ValueError("outer-train micro batches are required when micro_enabled=True")
        _validate_batch(data.micro_window_train, "micro_window_train")
        _validate_batch(data.micro_candidate_train, "micro_candidate_train")
        micro_groups = _groups_for(data.micro_window_train.windows, data.subject_by_session)
        micro_candidate_groups = _groups_for(data.micro_candidate_train.windows, data.subject_by_session)
        micro_keep = _micro_training_keep(data.micro_window_train, data.train_truths, config.micro_positive_middle_fraction)
        micro_labels = np.asarray(data.micro_window_train.labels[micro_keep], dtype=np.int8)
        _validate_binary(micro_labels, "micro window training")
        micro_train_features = data.micro_window_train.features[micro_keep]
        micro_train_groups = micro_groups[micro_keep]
        micro_fit_subjects = frozenset(micro_train_groups)
        micro_oof = crossfit_predict_proba(
            micro_train_features, micro_labels, micro_train_groups,
            data.micro_candidate_train.features, micro_candidate_groups,
            config.inner_splits, estimator_factory=lambda: _micro_window_estimator(config.seed + 10),
        )
        micro_windows = _windows_by_session(data.micro_candidate_train.windows, micro_oof.probabilities)
        micro_selection = select_micro_candidate_threshold(
            micro_windows, data.train_truths, config.micro_threshold_grid, config.micro_candidate,
        )
        candidates = union_candidates(raw_macro, micro_candidates(micro_windows, micro_selection.threshold, config.micro_candidate))
        features = multiscale_verifier_features(
            candidates, macro_windows, micro_windows,
            context_features_version=config.context_features_version,
            session_bounds_by_sid=_context_session_bounds(
                data, candidates, data.candidate_train.windows, data.micro_candidate_train.windows,
            ),
        )
    else:
        candidates, features = _candidate_matrix(raw_macro, macro_windows, include_coverage=config.density.coverage_fix,
                                                  window_batch=data.candidate_train, feature_mode=config.verifier_feature_mode)
    if not candidates:
        raise ValueError("inner window OOF produced no verifier candidates")
    _validate_imputable_features(features, "verifier training features", allow_all_missing_columns=not config.candidate_control_enabled)
    labels = _candidate_labels(candidates, data.train_truths)
    _validate_binary(labels, "verifier training")
    groups = _groups_for([candidate.event for candidate in candidates], data.subject_by_session)
    candidate_events = [candidate.event for candidate in candidates]
    best: tuple[tuple[object, ...], EventSelectionPolicy] | None = None
    if config.candidate_control_enabled:
        logistic_by_c = {
            float(c): crossfit_predict_proba(features, labels, groups, features, groups, config.inner_splits,
                estimator_factory=lambda c=c: _verifier_estimator(config.seed + 20, c)).probabilities
            for c in config.verifier_c_grid
        }
        lgbm = crossfit_predict_proba(features, labels, groups, features, groups, config.inner_splits,
            estimator_factory=lambda: _verifier_lgbm_estimator(config.seed + 21)).probabilities
        choices: list[tuple[float, tuple[object, ...], EventSelectionPolicy]] = []
        for c_rank, c in enumerate(config.verifier_c_grid):
            for blend_rank, weight in enumerate(config.verifier_blend_weight_grid):
                scores = weight * logistic_by_c[float(c)] + (1.0 - weight) * lgbm
                for admission_rank, admission in enumerate(
                    CandidateAdmissionConfig(nms, threshold, cap)
                    for nms in config.admission_nms_iou_grid
                    for threshold in config.admission_threshold_grid
                    for cap in config.admission_subject_cap_grid
                ):
                    rows = np.asarray(admit_candidates(candidates, scores, groups, admission), dtype=np.int64)
                    events, admitted_scores, admitted_groups = ([candidate_events[i] for i in rows], scores[rows], groups[rows])
                    admission_metrics = compute_event_metrics(events, data.train_truths)
                    if config.subject_cap_grid:
                        policy = select_event_policy(events, admitted_scores, data.train_truths, admitted_groups,
                                                     max_events_options=config.subject_cap_grid)
                    else:
                        selected = select_event_threshold(events, admitted_scores, data.train_truths)
                        policy = EventSelectionPolicy(selected.threshold, None, selected.metrics)
                    rank = (policy.metrics.f1, policy.metrics.ppv, -len(rows), -c_rank, -blend_rank,
                            -admission_rank, -policy.threshold, -float(policy.max_events_per_group or math.inf))
                    choices.append((admission_metrics.sensitivity, rank, policy))
        feasible = [choice for choice in choices if choice[0] >= config.admission_minimum_recall]
        choice = max(feasible or choices, key=lambda value: value[1] if feasible else (value[0], *value[1]))
        policy = choice[2]
    else:
        for c in config.verifier_c_grid:
            scores = crossfit_predict_proba(features, labels, groups, features, groups, config.inner_splits,
                estimator_factory=lambda c=c: _verifier_estimator(config.seed + 1, c)).probabilities
            selected = select_event_threshold(candidate_events, scores, data.train_truths)
            policy = EventSelectionPolicy(selected.threshold, None, selected.metrics)
            rank = (policy.metrics.f1, policy.metrics.ppv, -float(c))
            if best is None or rank > best[0]:
                best = (rank, policy)
        assert best is not None
        policy = best[1]
    expected = expected_feature_dimensions(config, macro_width=train_features.shape[1])
    observed = (train_features.shape[1], features.shape[1]) + ((micro_train_features.shape[1],) if config.micro_enabled else ())
    if observed != expected:
        raise ValueError("fitted feature dimensions do not match the run configuration")
    return InnerOOFResult(
        inner_metrics=policy.metrics,
        verifier_feature_count=features.shape[1],
        window_fit_subjects=frozenset(window_groups),
        verifier_fit_subjects=frozenset(groups),
        micro_window_fit_subjects=micro_fit_subjects,
    )


def _execute_outer_dataset(
    config: RunConfig,
    data_source: FoldDataset,
    *,
    peak_provider: Callable[[], int | None] | None = None,
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
        _validate_imputable_features(batch.features, f"{name}.features")
        _validate_tristate_labels(batch.labels, f"{name}.labels")

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
            _validate_imputable_features(batch.features, f"{name}.features")
            _validate_tristate_labels(batch.labels, f"{name}.labels")
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
            train_candidates,
            oof_windows,
            micro_oof_windows,
            context_features_version=config.context_features_version,
            session_bounds_by_sid=_context_session_bounds(
                data_source,
                train_candidates,
                data_source.candidate_train.windows,
                micro_candidates_batch.windows,
            ),
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
    _validate_imputable_features(
        train_candidate_features,
        "verifier training features",
        allow_all_missing_columns=not config.candidate_control_enabled,
    )
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
    selected_blend_weight: float | None = None
    selected_admission: CandidateAdmissionConfig | None = None
    verifier_logistic_oof_seconds = 0.0
    verifier_lgbm_oof_seconds = 0.0
    admission_selection_seconds = 0.0

    if config.candidate_control_enabled:
        stage_started = time.perf_counter()
        logistic_oof_by_c = {}
        for regularization_c in config.verifier_c_grid:
            logistic_oof_by_c[float(regularization_c)] = crossfit_predict_proba(
                train_candidate_features,
                train_candidate_labels,
                train_candidate_groups,
                train_candidate_features,
                train_candidate_groups,
                config.inner_splits,
                estimator_factory=lambda c=regularization_c: _verifier_estimator(
                    config.seed + 20, c
                ),
            ).probabilities
        verifier_logistic_oof_seconds = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        lgbm_oof = crossfit_predict_proba(
            train_candidate_features,
            train_candidate_labels,
            train_candidate_groups,
            train_candidate_features,
            train_candidate_groups,
            config.inner_splits,
            estimator_factory=lambda: _verifier_lgbm_estimator(config.seed + 21),
        ).probabilities
        verifier_lgbm_oof_seconds = time.perf_counter() - stage_started
        if not np.isfinite(lgbm_oof).all():
            raise RuntimeError("LightGBM verifier OOF produced non-finite scores")

        admission_configs = tuple(
            CandidateAdmissionConfig(nms_iou, threshold, cap)
            for nms_iou in config.admission_nms_iou_grid
            for threshold in config.admission_threshold_grid
            for cap in config.admission_subject_cap_grid
        )
        stage_started = time.perf_counter()
        stacked_choices: list[
            tuple[
                float,
                tuple[object, ...],
                float,
                float,
                CandidateAdmissionConfig,
                EventSelectionPolicy,
            ]
        ] = []
        for c_rank, regularization_c in enumerate(config.verifier_c_grid):
            logistic_oof = logistic_oof_by_c[float(regularization_c)]
            if (
                logistic_oof.shape != lgbm_oof.shape
                or not np.isfinite(logistic_oof).all()
            ):
                raise RuntimeError("verifier OOF scores are not finite and aligned")
            for blend_rank, blend_weight in enumerate(config.verifier_blend_weight_grid):
                blended_scores = (
                    blend_weight * logistic_oof
                    + (1.0 - blend_weight) * lgbm_oof
                )
                for admission_rank, admission in enumerate(admission_configs):
                    admitted = admit_candidates(
                        train_candidates,
                        blended_scores,
                        train_candidate_groups,
                        admission,
                    )
                    admitted_rows = np.asarray(admitted, dtype=np.int64)
                    admitted_events = [
                        train_candidate_events[index] for index in admitted
                    ]
                    admitted_scores = blended_scores[admitted_rows]
                    admitted_groups = train_candidate_groups[admitted_rows]
                    admission_metrics = compute_event_metrics(
                        admitted_events, data_source.train_truths
                    )
                    if config.subject_cap_grid:
                        candidate_policy = select_event_policy(
                            admitted_events,
                            admitted_scores,
                            data_source.train_truths,
                            admitted_groups,
                            max_events_options=config.subject_cap_grid,
                        )
                    else:
                        threshold_selection = select_event_threshold(
                            admitted_events,
                            admitted_scores,
                            data_source.train_truths,
                        )
                        candidate_policy = EventSelectionPolicy(
                            threshold_selection.threshold,
                            None,
                            threshold_selection.metrics,
                        )
                    event_rank = (
                        candidate_policy.metrics.f1,
                        candidate_policy.metrics.ppv,
                        -len(admitted),
                        -c_rank,
                        -blend_rank,
                        -admission_rank,
                        -candidate_policy.threshold,
                        -float(candidate_policy.max_events_per_group or math.inf),
                    )
                    stacked_choices.append(
                        (
                            admission_metrics.sensitivity,
                            event_rank,
                            float(regularization_c),
                            float(blend_weight),
                            admission,
                            candidate_policy,
                        )
                    )
        admission_selection_seconds = time.perf_counter() - stage_started
        feasible_choices = [
            choice
            for choice in stacked_choices
            if choice[0] >= config.admission_minimum_recall
        ]
        if feasible_choices:
            selected_choice = max(feasible_choices, key=lambda choice: choice[1])
        else:
            selected_choice = max(
                stacked_choices, key=lambda choice: (choice[0], *choice[1])
            )
        (
            _,
            _,
            selected_c,
            selected_blend_weight,
            selected_admission,
            policy,
        ) = selected_choice
    else:
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
    verifier_oof_seconds = (
        verifier_logistic_oof_seconds
        + verifier_lgbm_oof_seconds
        + admission_selection_seconds
        if config.candidate_control_enabled
        else time.perf_counter() - stage_started
    )

    stage_started = time.perf_counter()
    final_fit = _fit_final_models(
        config,
        train_features=train_features,
        train_labels=train_labels,
        micro_train_features=(micro_train_features if config.micro_enabled else None),
        micro_train_labels=(micro_train_labels if config.micro_enabled else None),
        train_candidate_features=train_candidate_features,
        train_candidate_labels=train_candidate_labels,
        verifier_c=selected_c,
        policy=policy,
        admission=selected_admission,
        blend_weight=selected_blend_weight,
    )
    fit_seconds = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    validation_features = _with_time_prior(
        data_source.validation.features, data_source.validation.windows
    )
    validation_window_scores = _positive_probability(
        final_fit.macro, validation_features
    )
    validation_windows = _windows_by_session(
        data_source.validation.windows, validation_window_scores
    )
    raw_validation_candidates = density_candidates(
        validation_windows, config.density
    )
    raw_micro_validation_candidates = []
    if config.micro_enabled:
        assert final_fit.micro is not None
        micro_validation_scores = _positive_probability(
            final_fit.micro, micro_validation.features
        )
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
            validation_candidates,
            validation_windows,
            micro_validation_windows,
            context_features_version=config.context_features_version,
            session_bounds_by_sid=_context_session_bounds(
                data_source,
                validation_candidates,
                data_source.validation.windows,
                micro_validation.windows,
            ),
        )
    else:
        validation_candidates, validation_candidate_features = _candidate_matrix(
            raw_validation_candidates,
            validation_windows,
            include_coverage=config.density.coverage_fix,
            window_batch=data_source.validation,
            feature_mode=config.verifier_feature_mode,
        )
    raw_union_candidate_count = len(validation_candidates)
    verifier_logistic_outer_seconds = 0.0
    verifier_lgbm_outer_seconds = 0.0
    if validation_candidates:
        verifier_started = time.perf_counter()
        logistic_validation_scores = _positive_probability(
            final_fit.verifier_logistic, validation_candidate_features
        )
        logistic_outer_elapsed = time.perf_counter() - verifier_started
        if config.candidate_control_enabled:
            verifier_logistic_outer_seconds = logistic_outer_elapsed
        if config.candidate_control_enabled:
            verifier_started = time.perf_counter()
            assert final_fit.verifier_lgbm is not None
            lgbm_validation_scores = _positive_probability(
                final_fit.verifier_lgbm, validation_candidate_features
            )
            verifier_lgbm_outer_seconds = time.perf_counter() - verifier_started
            validation_scores = (
                selected_blend_weight * logistic_validation_scores
                + (1.0 - selected_blend_weight) * lgbm_validation_scores
            )
        else:
            validation_scores = logistic_validation_scores
    else:
        validation_scores = np.empty(0, dtype=np.float64)
    validation_candidate_events = [
        candidate.event for candidate in validation_candidates
    ]
    validation_candidate_groups = _groups_for(
        validation_candidate_events,
        data_source.subject_by_session,
    )
    if config.candidate_control_enabled:
        admitted_indices = admit_candidates(
            validation_candidates,
            validation_scores,
            validation_candidate_groups,
            selected_admission,
        )
        admitted_rows = np.asarray(admitted_indices, dtype=np.int64)
        validation_candidates = [
            validation_candidates[index] for index in admitted_indices
        ]
        validation_candidate_events = [
            candidate.event for candidate in validation_candidates
        ]
        validation_candidate_features = validation_candidate_features[admitted_rows]
        validation_scores = validation_scores[admitted_rows]
        validation_candidate_groups = validation_candidate_groups[admitted_rows]
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
    observed_dimensions = (train_features.shape[1], train_candidate_features.shape[1])
    if config.micro_enabled:
        observed_dimensions += (micro_train_features.shape[1],)
    feature_dimensions = expected_feature_dimensions(
        config, macro_width=train_features.shape[1]
    )
    if observed_dimensions != feature_dimensions:
        raise ValueError("fitted feature dimensions do not match the run configuration")
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
    if config.candidate_control_enabled:
        timings.update({
            "verifier_logistic_oof": verifier_logistic_oof_seconds,
            "verifier_lgbm_oof": verifier_lgbm_oof_seconds,
            "admission_selection": admission_selection_seconds,
        })
    runtime = _runtime_diagnostics(peak_provider)
    result = FoldResult(
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
        selected_blend_weight=selected_blend_weight,
        selected_admission_nms_iou=(
            selected_admission.nms_iou if selected_admission is not None else None
        ),
        selected_admission_threshold=(
            selected_admission.threshold if selected_admission is not None else None
        ),
        selected_admission_subject_cap=(
            selected_admission.max_candidates_per_subject
            if selected_admission is not None else None
        ),
        raw_union_candidate_count=(
            raw_union_candidate_count if config.candidate_control_enabled else 0
        ),
        admitted_candidate_count=(
            len(validation_candidates) if config.candidate_control_enabled else 0
        ),
        verifier_logistic_oof_seconds=verifier_logistic_oof_seconds,
        verifier_lgbm_oof_seconds=verifier_lgbm_oof_seconds,
        verifier_logistic_outer_seconds=verifier_logistic_outer_seconds,
        verifier_lgbm_outer_seconds=verifier_lgbm_outer_seconds,
        subject_diagnostics=subject_diagnostics(
            selected_predictions,
            data_source.validation_truths,
            {
                sid: data_source.subject_by_session[sid]
                for sid in {window.sid for window in data_source.validation.windows}
            },
            runtime=runtime,
        ),
        runtime_diagnostics=runtime,
    )
    return _OuterRun(result=result, fit=final_fit)


def _run_outer_dataset(
    config: RunConfig,
    data_source: FoldDataset,
) -> FoldResult:
    """Public evaluation compatibility wrapper around the artifact-capable run."""

    return _execute_outer_dataset(config, data_source).result


def fit_outer_fold_for_promotion(
    config: RunConfig,
    source: object,
) -> tuple[FoldResult, _FinalModelFit, FoldDataset]:
    """Retrain one outer fold and retain only its train-selected fitted state.

    Promotion calls this instead of loading a cached result so each evidence
    bundle is a fresh fit.  The returned outer metrics are evidence metadata,
    never policy-selection input for deployment.
    """

    loader = getattr(source, "load_outer_fold", None)
    if not callable(loader):
        raise TypeError("promotion source must expose load_outer_fold")
    dataset = loader(config)
    if not isinstance(dataset, FoldDataset):
        raise TypeError("promotion source must return FoldDataset")
    execution = _execute_outer_dataset(config, dataset)
    return execution.result, execution.fit, dataset


def fit_full_target_deployment(
    config: RunConfig,
    data: FoldDataset,
    frozen_policy: Mapping[str, object],
) -> tuple[_FinalModelFit, int]:
    """Fit deployment models on one validated held-out-partition union.

    ``frozen_policy`` has already been aggregated from five train-only outer
    policies.  This routine uses target labels to train models, but never
    searches an event threshold, blend, admission setting, or budget.
    """

    required = {
        "micro_threshold", "blend_weight", "nms_iou", "admission_threshold",
        "max_candidates_per_subject", "threshold", "max_events_per_group", "verifier_c",
    }
    missing = sorted(required - set(frozen_policy))
    if missing:
        raise ValueError("deployment policy is missing: " + ", ".join(missing))
    if not config.micro_enabled or not config.candidate_control_enabled:
        raise ValueError("deployment fitting requires the registered multiscale candidate-control configuration")
    if data.micro_window_train is None or data.micro_candidate_train is None:
        raise ValueError("deployment fitting requires full-target micro windows")
    _validate_full_target_dataset(data)
    policy_values = {name: frozen_policy[name] for name in required}
    micro_threshold = float(policy_values["micro_threshold"])
    blend_weight = float(policy_values["blend_weight"])
    admission = CandidateAdmissionConfig(
        float(policy_values["nms_iou"]),
        float(policy_values["admission_threshold"]),
        policy_values["max_candidates_per_subject"],  # type: ignore[arg-type]
    )
    event_policy = EventSelectionPolicy(
        float(policy_values["threshold"]),
        policy_values["max_events_per_group"],  # type: ignore[arg-type]
        EventMetrics(0, 0, 0, 0.0, 0.0, 0.0),
    )
    verifier_c = float(policy_values["verifier_c"])
    if not all(math.isfinite(value) for value in (micro_threshold, blend_weight, verifier_c)):
        raise ValueError("deployment policy numeric values must be finite")

    macro_keep = np.asarray(data.window_train.labels) >= 0
    macro_labels = np.asarray(data.window_train.labels[macro_keep], dtype=np.int8)
    _validate_binary(macro_labels, "deployment macro window training")
    macro_features = _with_time_prior(data.window_train.features, data.window_train.windows)[macro_keep]
    micro_keep = _micro_training_keep(
        data.micro_window_train, data.train_truths, config.micro_positive_middle_fraction
    )
    micro_labels = np.asarray(data.micro_window_train.labels[micro_keep], dtype=np.int8)
    _validate_binary(micro_labels, "deployment micro window training")
    micro_features = data.micro_window_train.features[micro_keep]

    # Candidate feature construction needs window probabilities.  The final fit
    # below repeats these deterministic registered fits after labels/features
    # are frozen, so the serialized macro/micro models exactly match them.
    macro_probe = _window_estimator(config.seed + 2).fit(macro_features, macro_labels)
    micro_probe = _micro_window_estimator(config.seed + 12).fit(micro_features, micro_labels)
    candidate_macro_features = _with_time_prior(
        data.candidate_train.features, data.candidate_train.windows
    )
    macro_windows = _windows_by_session(
        data.candidate_train.windows,
        _positive_probability(macro_probe, candidate_macro_features),
    )
    micro_windows = _windows_by_session(
        data.micro_candidate_train.windows,
        _positive_probability(micro_probe, data.micro_candidate_train.features),
    )
    candidates = union_candidates(
        density_candidates(macro_windows, config.density),
        micro_candidates(micro_windows, micro_threshold, config.micro_candidate),
    )
    if not candidates:
        raise ValueError("full-target deployment fit produced no candidates")
    candidate_features = multiscale_verifier_features(
        candidates,
        macro_windows,
        micro_windows,
        context_features_version=config.context_features_version,
        session_bounds_by_sid=_context_session_bounds(
            data,
            candidates,
            data.candidate_train.windows,
            data.micro_candidate_train.windows,
        ),
    )
    if candidate_features.shape[1] != expected_feature_dimensions(config)[1]:
        raise ValueError(
            "full-target deployment verifier features do not match the configured width"
        )
    _validate_imputable_features(
        candidate_features,
        "full-target deployment verifier features",
    )
    candidate_labels = _candidate_labels(candidates, data.train_truths)
    _validate_binary(candidate_labels, "deployment verifier training")
    final_fit = _fit_final_models(
        config,
        train_features=macro_features,
        train_labels=macro_labels,
        micro_train_features=micro_features,
        micro_train_labels=micro_labels,
        train_candidate_features=candidate_features,
        train_candidate_labels=candidate_labels,
        verifier_c=verifier_c,
        policy=event_policy,
        admission=admission,
        blend_weight=blend_weight,
    )
    return final_fit, len(candidates)


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

    def deployment_input_files(self, configs: Sequence[RunConfig]) -> tuple[Path, ...]:
        """Fingerprint only the files permitted in the full-target union.

        Unlike :meth:`input_files`, this deliberately excludes every
        ``train``/``meal_train``/``no_meal_train`` cache.  The returned inputs
        document the held-out partition union used by the deployment models.
        """

        from src.data import manifests

        required: list[Path] = [manifests.INDEX_CSV, manifests.MEALS_CSV]
        for config in configs:
            required.append(self._split_path(config.outer_fold, "val"))
            if config.micro_enabled:
                required.append(self._micro_split_path(config.outer_fold, "val"))
            split_manifest = self.root / "cache" / "splits" / f"fold{config.outer_fold}.json"
            required.append(split_manifest)

        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("missing deployment inputs: " + ", ".join(missing))
        return tuple(sorted(set(required), key=lambda path: str(path)))

    @staticmethod
    def _deployment_file_state(path: Path) -> dict[str, object]:
        """Return a canonical, content-sensitive source-state record."""

        canonical = Path(path).resolve()
        if not canonical.exists():
            return {"path": str(canonical), "missing": True}
        digest = hashlib.sha256()
        with canonical.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        stat = canonical.stat()
        return {
            "path": str(canonical),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": digest.hexdigest(),
        }

    def deployment_input_state(
        self, configs: Sequence[RunConfig]
    ) -> tuple[dict[str, object], ...]:
        """Fingerprint every artifact read while fitting the deployment union.

        Required validation caches and manifests must exist, but session caches
        are intentionally represented even when absent: ``_eligible_truths``
        legally skips them.  This records both that decision and any later
        change in content or availability without passing missing paths to the
        outer-fold cache-key code path.
        """

        required = self.deployment_input_files(configs)
        session_paths: list[Path] = []
        for config in configs:
            manifest_path = self.root / "cache" / "splits" / f"fold{config.outer_fold}.json"
            with manifest_path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            validation_sessions = manifest["val_sessions"]
            if not isinstance(validation_sessions, list):
                raise ValueError(f"deployment split manifest has invalid val_sessions: {manifest_path}")
            # ``_eligible_truths`` uses the validation cache's window IDs,
            # rather than trusting the manifest alone.  Bind that exact read
            # set too, so a stale or inconsistent cache cannot hide a session
            # state change from the deployment provenance.
            validation_sessions = [
                *validation_sessions,
                *(window.sid for window in self._load_batch(
                    self._split_path(config.outer_fold, "val")
                ).windows),
            ]
            session_paths.extend(
                self.session_dir / f"{str(session_id)}.npz"
                for session_id in validation_sessions
            )
        paths = {Path(path).resolve() for path in (*required, *session_paths)}
        return tuple(
            self._deployment_file_state(path)
            for path in sorted(paths, key=lambda item: str(item))
        )

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
        session_bounds_by_sid = {
            str(row["session_id"]): (
                int(row["timeStamp.startTime"]),
                int(row["timeStamp.endTime"]),
            )
            for _, row in index.iterrows()
        }
        if any(end < start for start, end in session_bounds_by_sid.values()):
            raise ValueError("sensor index contains unordered session bounds")
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
            session_bounds_by_sid=session_bounds_by_sid,
            validation_truth_slices=validation_slices,
            micro_window_train=micro_batches.get("train"),
            micro_candidate_train=micro_batches.get("candidate_train"),
            micro_validation=micro_batches.get("val"),
            micro_cache_extraction_seconds=extraction_seconds,
        )

    def load_outer_train_only(self, config: RunConfig) -> OuterTrainOnlyData:
        """Read only canonical outer-train roles for inner OOF selection.

        This intentionally does not delegate to ``load_outer_fold``: that
        loader opens held-out NPZs and is therefore not legal for the E02
        inner gate, even when callers promise to ignore the result.
        """

        from src.data import manifests

        fold = config.outer_fold
        window_train = self._load_batch(self._split_path(fold, "train"))
        meal_train = self._load_batch(self._split_path(fold, "meal_train"))
        no_meal_train = self._load_batch(self._split_path(fold, "no_meal_train"))
        candidate_train = self._combine_batches(meal_train, no_meal_train)

        micro_window_train: WindowBatch | None = None
        micro_candidate_train: WindowBatch | None = None
        extraction_seconds = 0.0
        if config.micro_enabled:
            micro_window_train, seconds = self._load_micro_batch(
                self._micro_split_path(fold, "train"),
                self._micro_expected_metadata(config, "train"),
            )
            extraction_seconds += seconds
            micro_meal, seconds = self._load_micro_batch(
                self._micro_split_path(fold, "meal_train"),
                self._micro_expected_metadata(config, "meal_train"),
            )
            extraction_seconds += seconds
            micro_no_meal, seconds = self._load_micro_batch(
                self._micro_split_path(fold, "no_meal_train"),
                self._micro_expected_metadata(config, "no_meal_train"),
            )
            extraction_seconds += seconds
            micro_candidate_train = self._combine_batches(micro_meal, micro_no_meal)
            for name, macro_batch in (("micro_window_train", window_train), ("micro_candidate_train", candidate_train)):
                batch = micro_window_train if name == "micro_window_train" else micro_candidate_train
                allowed_sids = {window.sid for window in macro_batch.windows}
                keep = np.asarray([window.sid in allowed_sids for window in batch.windows], dtype=bool)
                filtered = WindowBatch(batch.features[keep], batch.labels[keep], tuple(
                    window for window, retained in zip(batch.windows, keep) if retained
                ))
                if name == "micro_window_train":
                    micro_window_train = filtered
                else:
                    micro_candidate_train = filtered

        index = manifests.load_sensor_index()
        subject_by_session = {
            str(row["session_id"]): str(row["externalid"])
            for _, row in index.iterrows()
        }
        session_bounds_by_sid = {
            str(row["session_id"]): (int(row["timeStamp.startTime"]), int(row["timeStamp.endTime"]))
            for _, row in index.iterrows()
        }
        if any(end < start for start, end in session_bounds_by_sid.values()):
            raise ValueError("sensor index contains unordered session bounds")
        train_sids = {window.sid for window in candidate_train.windows}
        train_truths, _ = self._eligible_truths(train_sids, candidate_train, subject_by_session)
        return OuterTrainOnlyData(
            window_train=window_train,
            candidate_train=candidate_train,
            train_truths=train_truths,
            subject_by_session=subject_by_session,
            session_bounds_by_sid=session_bounds_by_sid,
            micro_window_train=micro_window_train,
            micro_candidate_train=micro_candidate_train,
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
        "selected_blend_weight": result.selected_blend_weight,
        "selected_admission_nms_iou": result.selected_admission_nms_iou,
        "selected_admission_threshold": result.selected_admission_threshold,
        "selected_admission_subject_cap": result.selected_admission_subject_cap,
        "raw_union_candidate_count": result.raw_union_candidate_count,
        "admitted_candidate_count": result.admitted_candidate_count,
        "verifier_logistic_oof_seconds": result.verifier_logistic_oof_seconds,
        "verifier_lgbm_oof_seconds": result.verifier_lgbm_oof_seconds,
        "verifier_logistic_outer_seconds": result.verifier_logistic_outer_seconds,
        "verifier_lgbm_outer_seconds": result.verifier_lgbm_outer_seconds,
        "subject_diagnostics": dict(result.subject_diagnostics),
        "runtime_diagnostics": dict(result.runtime_diagnostics),
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
        selected_blend_weight=(
            float(payload["selected_blend_weight"])
            if payload.get("selected_blend_weight") is not None
            else None
        ),
        selected_admission_nms_iou=(
            float(payload["selected_admission_nms_iou"])
            if payload.get("selected_admission_nms_iou") is not None
            else None
        ),
        selected_admission_threshold=(
            float(payload["selected_admission_threshold"])
            if payload.get("selected_admission_threshold") is not None
            else None
        ),
        selected_admission_subject_cap=(
            int(payload["selected_admission_subject_cap"])
            if payload.get("selected_admission_subject_cap") is not None
            else None
        ),
        raw_union_candidate_count=int(payload.get("raw_union_candidate_count", 0)),
        admitted_candidate_count=int(payload.get("admitted_candidate_count", 0)),
        verifier_logistic_oof_seconds=float(
            payload.get("verifier_logistic_oof_seconds", 0.0)
        ),
        verifier_lgbm_oof_seconds=float(payload.get("verifier_lgbm_oof_seconds", 0.0)),
        verifier_logistic_outer_seconds=float(
            payload.get("verifier_logistic_outer_seconds", 0.0)
        ),
        verifier_lgbm_outer_seconds=float(
            payload.get("verifier_lgbm_outer_seconds", 0.0)
        ),
        subject_diagnostics=dict(payload.get("subject_diagnostics", {})),
        runtime_diagnostics=dict(payload.get("runtime_diagnostics", {
            "peak_working_set_bytes": None,
            "unavailable_reason": "not recorded by legacy cache",
            "cuda_peak_bytes": None,
            "ssl_runtime": None,
        })),
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
    configs: Sequence[RunConfig], results: Sequence[FoldResult],
    *,
    diagnostic_payloads: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Pool event counts and truth-weight candidate recall across outer folds."""
    if not configs or len(configs) != len(results):
        raise ValueError("configs and results must have equal nonzero lengths")
    if len({config.outer_fold for config in configs}) != len(configs):
        raise ValueError("duplicate outer folds are not allowed")
    if diagnostic_payloads is not None and len(diagnostic_payloads) != len(results):
        raise ValueError("diagnostic payloads must align with fold results")

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
    peaks = [
        value
        for result in results
        for value in (result.runtime_diagnostics.get("peak_working_set_bytes"),)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
    ]
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
        "runtime_diagnostics": {
            "peak_working_set_bytes": max(peaks) if peaks else None,
            "unavailable_reason": (
                None
                if peaks
                else "; ".join(sorted({
                    str(result.runtime_diagnostics.get("unavailable_reason"))
                    for result in results
                    if result.runtime_diagnostics.get("unavailable_reason")
                })) or "worker peak working set was not recorded"
            ),
            "aggregation": "max_worker_peak",
            "cuda_peak_bytes": None,
            "ssl_runtime": None,
        },
        "subject_diagnostics": aggregate_subject_diagnostics(
            diagnostic_payloads
            if diagnostic_payloads is not None
            else [result.subject_diagnostics for result in results]
        ),
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
        # Use raw LF bytes so diagnostics have the exact canonical stream that
        # promotion attests, including on Windows where text mode translates
        # newlines to CRLF.
        temporary.write_bytes(
            (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
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
    key = cache_key(config, expected_feature_dimensions(config), input_files)
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
