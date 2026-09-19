"""Standalone inference for a verified event-stack deployment bundle.

The current stack has no audited raw-session-to-63/47/56-feature runtime.  Its
public inference contract is therefore deliberately narrow: a caller supplies
precomputed candidate features plus the bundle feature-schema SHA-256.  This
prevents the deployment package from silently substituting a different feature
extractor or depending on training data.  A future raw-session adapter must
produce this exact JSON contract before it may be registered here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np

from event_stack.artifacts import load_event_stack_bundle
from event_stack.candidate_control import CandidateAdmissionConfig, admit_candidates
from event_stack.event_stack import (
    DensityConfig, EventRef, MicroCandidateConfig, apply_event_policy,
    density_candidates, micro_candidates, multiscale_verifier_features, union_candidates,
)
from event_stack.features.macro import MacroFeatureConfig, add_time_prior, extract_macro_windows
from event_stack.features.micro import extract_micro_windows
from event_stack.imu_features import MicroFeatureConfig
from event_stack.io.raw_session import RawSessionSource, load_raw_session
from event_stack.preprocessing.timeline import valid_imu_spans


_MODEL_NAMES = ("macro", "micro", "verifier_logistic", "verifier_lgbm")
_METADATA_NAMES = ("policy.json", "run_config.json", "feature_schema.json")
_RUNTIME_DEPENDENCIES = (
    ("numpy", "numpy"),
    ("joblib", "joblib"),
    ("scikit_learn", "scikit-learn"),
    ("lightgbm", "lightgbm"),
)
_DEVICE_CHOICES = ("auto", "cpu", "gpu", "cuda")
# TODO: Add an audited, source-controlled CUDA component identifier here only
# after its implementation and release verification are available.  A package
# file, manifest flag, or arbitrary callable is never a registration protocol.
_SUPPORTED_CUDA_ADAPTER_IDS = frozenset()
_CONTEXT_V1_SUFFIXES = (
    "pre_mean", "pre_max", "pre_std", "pre_above_fraction",
    "candidate_mean", "candidate_max", "candidate_std", "candidate_above_fraction",
    "post_mean", "post_max", "post_std", "post_above_fraction",
    "candidate_minus_pre_mean", "candidate_minus_pre_max",
    "candidate_minus_pre_std", "candidate_minus_pre_above_fraction",
    "candidate_minus_post_mean", "candidate_minus_post_max",
    "candidate_minus_post_std", "candidate_minus_post_above_fraction",
    "pre_coverage", "candidate_coverage", "post_coverage",
    "neighbor_run_count", "neighbor_run_total_duration_s", "neighbor_run_max_duration_s",
    "preceding_run_distance_s", "following_run_distance_s",
    "candidate_center_half_mass_fraction", "candidate_first_minus_second_mean",
)
_CONTEXT_V1_COLUMNS = tuple(
    f"{scale}_{suffix}"
    for scale in ("macro", "micro")
    for suffix in _CONTEXT_V1_SUFFIXES
)
_CONTEXT_V1_SCHEMA_HASH = hashlib.sha256(
    json.dumps(_CONTEXT_V1_COLUMNS, separators=(",", ":")).encode()
).hexdigest()


def _canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _schema_hash(schema: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(dict(schema), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _schema_widths(schema: Mapping[str, object]) -> dict[str, int]:
    """Accept repository schema-v1 maps and validate schema-v2 Context-v1."""

    def widths(value: object) -> dict[str, int] | None:
        if not isinstance(value, Mapping) or set(value) != {"macro", "micro", "verifier"}:
            return None
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in value.values()):
            return None
        return {name: int(value[name]) for name in ("macro", "micro", "verifier")}

    legacy = widths(schema)
    if legacy is not None:
        return legacy
    if not isinstance(schema, Mapping) or set(schema) != {"schema_version", "widths", "context"}:
        raise ValueError("feature schema is incompatible with the event-stack runtime")
    if schema["schema_version"] != 2:
        raise ValueError("feature schema version is unsupported")
    resolved_widths = widths(schema["widths"])
    context = schema["context"]
    if resolved_widths is None or not isinstance(context, Mapping) or set(context) != {"version", "columns", "schema_hash"}:
        raise ValueError("feature schema v2 is malformed")
    if context["version"] == "v1":
        if (
            context["columns"] != list(_CONTEXT_V1_COLUMNS)
            or context["schema_hash"] != _CONTEXT_V1_SCHEMA_HASH
            or resolved_widths["verifier"] != 56 + len(_CONTEXT_V1_COLUMNS)
        ):
            raise ValueError("feature schema Context-v1 columns, hash, or width is invalid")
    elif context["version"] is None:
        if context["columns"] != [] or context["schema_hash"] is not None:
            raise ValueError("feature schema without context must have empty context metadata")
    else:
        raise ValueError("feature schema context version is unsupported")
    return resolved_widths


def _has_registered_cuda_component() -> bool:
    """Resolve hardware support solely from the source-controlled registry."""

    return bool(_SUPPORTED_CUDA_ADAPTER_IDS)


def resolve_device(requested: str) -> str:
    """Resolve the only currently supported device without inferring CUDA support."""

    normalized = str(requested).lower()
    if normalized not in _DEVICE_CHOICES:
        raise ValueError("device must be one of: auto, cpu, gpu, cuda")
    if normalized == "gpu":
        normalized = "cuda"
    if normalized == "cpu":
        return "cpu"
    if normalized == "auto":
        return "cuda" if _has_registered_cuda_component() else "cpu"
    raise RuntimeError("forced CUDA requested, but no supported CUDA component is registered")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} cannot be read: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{label} must be a JSON object")
    return result


def _verify_bundle(bundle_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate the Task-4 bundle manifest before deserializing joblib objects."""

    if not bundle_path.is_dir() or bundle_path.is_symlink():
        raise ValueError("bundle manifest directory is missing or unsafe")
    manifest = _read_json(bundle_path / "manifest.json", "bundle manifest")
    if manifest.get("role") != "deployment":
        raise ValueError("bundle manifest role must be deployment for inference")
    models = manifest.get("models")
    files = manifest.get("files")
    if not isinstance(models, list) or sorted(models) != sorted(_MODEL_NAMES):
        raise ValueError("bundle manifest must declare the complete event-stack model set")
    if not isinstance(files, dict):
        raise ValueError("bundle manifest files must be an object")
    expected = {f"{name}.joblib" for name in _MODEL_NAMES} | set(_METADATA_NAMES)
    if set(files) != expected:
        raise ValueError("bundle manifest file set is incomplete or incompatible")
    actual = {path.name for path in bundle_path.iterdir() if path.is_file() and path.name != "manifest.json"}
    if actual != expected:
        raise ValueError("bundle manifest does not match bundle contents")
    for name in sorted(expected):
        path = bundle_path / name
        if path.is_symlink() or not path.is_file() or not isinstance(files.get(name), str):
            raise ValueError(f"bundle manifest entry is invalid: {name}")
        if _sha256(path) != files[name]:
            raise ValueError(f"bundle manifest SHA-256 mismatch for {name}")
    policy = _read_json(bundle_path / "policy.json", "policy")
    schema_raw = _read_json(bundle_path / "feature_schema.json", "feature schema")
    _schema_widths(schema_raw)
    return manifest, policy, schema_raw


def _runtime_capabilities(bundle_path: Path) -> bool:
    """Reject package CUDA claims; only the empty source registry can answer false."""

    runtime_manifest = bundle_path.parent / "runtime_manifest.json"
    if not runtime_manifest.is_file():
        return False
    payload = _read_json(runtime_manifest, "runtime manifest")
    files = payload.get("files")
    if not isinstance(files, dict):
        raise ValueError("runtime manifest files must be an object")
    package_root = bundle_path.parent
    actual = {
        path.relative_to(package_root).as_posix(): _sha256(path)
        for path in package_root.rglob("*")
        if path.is_file() and path.name != "runtime_manifest.json" and "__pycache__" not in path.parts
    }
    if any(Path(name).name == "cuda_adapter.py" for name in actual):
        raise ValueError("runtime manifest CUDA declarations/components are unsupported")
    if files != actual:
        raise ValueError("runtime manifest checksum does not match package contents")
    bundle_manifest_hash = payload.get("bundle_manifest_sha256")
    if not isinstance(bundle_manifest_hash, str) or _sha256(bundle_path / "manifest.json") != bundle_manifest_hash:
        raise ValueError("runtime manifest checksum does not match bundle manifest")
    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, dict):
        raise ValueError("runtime manifest capabilities must be an object")
    if capabilities:
        raise ValueError("runtime manifest CUDA declarations/components are unsupported")
    return _has_registered_cuda_component()


def _validate_runtime_dependencies(manifest: Mapping[str, Any]) -> None:
    """Refuse incompatible pickle runtimes before deserializing any model."""

    versions = manifest.get("dependency_versions")
    expected_keys = {"python", *(key for key, _ in _RUNTIME_DEPENDENCIES)}
    if not isinstance(versions, dict) or set(versions) != expected_keys:
        raise ValueError("bundle manifest dependency_versions are incomplete or incompatible")
    expected_python = versions["python"]
    if not isinstance(expected_python, str):
        raise ValueError("bundle manifest python runtime version is invalid")
    expected_parts = expected_python.split(".")
    actual_parts = platform.python_version().split(".")
    if (
        len(expected_parts) != 3
        or not all(part.isdigit() for part in expected_parts)
        or len(actual_parts) < 2
        or actual_parts[:2] != expected_parts[:2]
    ):
        raise ValueError(
            "Python runtime version mismatch: expected "
            f"{expected_python} (requires Python {'.'.join(expected_parts[:2])}.x), "
            f"installed {platform.python_version()}"
        )
    for manifest_key, distribution in _RUNTIME_DEPENDENCIES:
        expected = versions[manifest_key]
        if not isinstance(expected, str) or not expected:
            raise ValueError(f"bundle manifest {distribution} runtime version is invalid")
        try:
            installed = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError as exc:
            raise ValueError(
                f"{distribution} runtime version mismatch: expected {expected}, installed missing"
            ) from exc
        if installed != expected:
            raise ValueError(
                f"{distribution} runtime version mismatch: expected {expected}, installed {installed}"
            )


def _probability(model: object, row: list[float], width: int, name: str) -> float:
    values = np.asarray(row, dtype=np.float64)
    if values.shape != (width,) or not np.isfinite(values).all():
        raise ValueError(f"candidate {name} features must be {width} finite numbers")
    if not hasattr(model, "predict_proba"):
        raise ValueError(f"model {name} does not expose predict_proba")
    probabilities = np.asarray(model.predict_proba(values.reshape(1, -1)), dtype=np.float64)
    if probabilities.shape != (1, 2) or not np.isfinite(probabilities).all():
        raise ValueError(f"model {name} returned incompatible probabilities")
    return float(probabilities[0, 1])


def _iou(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    intersection = min(int(left["end_ms"]), int(right["end_ms"])) - max(
        int(left["start_ms"]), int(right["start_ms"])
    )
    if intersection <= 0:
        return 0.0
    union = max(int(left["end_ms"]), int(right["end_ms"])) - min(
        int(left["start_ms"]), int(right["start_ms"])
    )
    return intersection / union


def _number(policy: Mapping[str, Any], name: str, default: float) -> float:
    value = float(policy.get(name, default))
    if not math.isfinite(value):
        raise ValueError(f"policy {name} must be finite")
    return value


def _strict_positive_cap(policy: Mapping[str, Any], name: str) -> int | None:
    value = policy[name]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"policy {name} must be a positive integer or null")
    return value


def _select_events(rows: list[dict[str, Any]], policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    required = {
        "admission_threshold", "nms_iou", "max_candidates_per_subject",
        "threshold", "max_events_per_group",
    }
    missing = sorted(required - set(policy))
    if missing:
        raise ValueError("policy is missing frozen inference fields: " + ", ".join(missing))
    admission_threshold = _number(policy, "admission_threshold", 0.0)
    event_threshold = _number(policy, "threshold", 0.0)
    nms_iou = _number(policy, "nms_iou", 1.0)
    candidate_cap = _strict_positive_cap(policy, "max_candidates_per_subject")
    event_cap = _strict_positive_cap(policy, "max_events_per_group")
    if not 0.0 <= admission_threshold <= 1.0 or not 0.0 <= event_threshold <= 1.0 or not 0.0 <= nms_iou <= 1.0:
        raise ValueError("policy thresholds must be in [0, 1]")
    ranked = sorted(
        (row for row in rows if row["score"] >= admission_threshold),
        key=lambda row: (-row["score"], row["sid"], row["start_ms"], row["end_ms"]),
    )
    nms_kept: list[dict[str, Any]] = []
    for row in ranked:
        if any(row["sid"] == other["sid"] and _iou(row, other) >= nms_iou for other in nms_kept):
            continue
        nms_kept.append(row)
    admitted: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for row in nms_kept:
        subject_id = row["subject_id"]
        if candidate_cap is not None and counts.get(subject_id, 0) >= candidate_cap:
            continue
        admitted.append(row)
        counts[subject_id] = counts.get(subject_id, 0) + 1
    selected: list[dict[str, Any]] = []
    event_counts: dict[str, int] = {}
    for row in admitted:
        subject_id = row["subject_id"]
        if row["score"] < event_threshold:
            continue
        if event_cap is not None and event_counts.get(subject_id, 0) >= event_cap:
            continue
        selected.append(row)
        event_counts[subject_id] = event_counts.get(subject_id, 0) + 1
    return sorted(selected, key=lambda row: (row["sid"], row["start_ms"], row["end_ms"], row["score"]))


def predict_feature_payload(bundle_path: Path, payload: Mapping[str, Any], *, device: str = "auto") -> dict[str, Any]:
    """Score schema-verified precomputed candidate features into canonical events."""

    bundle_path = Path(bundle_path)
    manifest, policy, schema = _verify_bundle(bundle_path)
    widths = _schema_widths(schema)
    _runtime_capabilities(bundle_path)
    _validate_runtime_dependencies(manifest)
    resolved_device = resolve_device(device)
    if not isinstance(payload, Mapping):
        raise ValueError("input feature payload must be an object")
    unknown_payload = set(payload) - {"feature_schema", "schema_hash", "sessions"}
    if unknown_payload:
        raise ValueError("input feature payload has unknown fields: " + ", ".join(sorted(unknown_payload)))
    input_schema = payload.get("feature_schema")
    if input_schema != schema or payload.get("schema_hash") != _schema_hash(schema):
        raise ValueError("input feature schema hash does not match the deployment bundle")
    sessions = payload.get("sessions")
    if not isinstance(sessions, list):
        raise ValueError("input feature payload sessions must be an array")
    models = {name: joblib.load(bundle_path / f"{name}.joblib") for name in _MODEL_NAMES}
    blend_weight = _number(policy, "blend_weight", 0.5)
    if not 0.0 <= blend_weight <= 1.0:
        raise ValueError("policy blend_weight must be in [0, 1]")
    scored: list[dict[str, Any]] = []
    subject_by_sid: dict[str, str] = {}
    for session in sessions:
        if not isinstance(session, Mapping):
            raise ValueError("each session must be an object")
        unknown_session = set(session) - {"subject_id", "sid", "candidates"}
        if unknown_session:
            raise ValueError("session has unknown fields: " + ", ".join(sorted(unknown_session)))
        subject_id = session.get("subject_id")
        sid = session.get("sid")
        if not isinstance(subject_id, str) or not subject_id:
            raise ValueError("each session must have a nonempty string subject_id")
        if not isinstance(sid, str) or not sid:
            raise ValueError("each session must have a nonempty string sid")
        if sid in subject_by_sid:
            raise ValueError("each sid must be globally unique within the input sessions")
        subject_by_sid[sid] = subject_id
        candidates = session.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("each session candidates value must be an array")
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise ValueError("candidate entries must be objects")
            unknown_candidate = set(candidate) - {"start_ms", "end_ms", "macro", "micro", "verifier"}
            if unknown_candidate:
                raise ValueError("candidate has unknown fields: " + ", ".join(sorted(unknown_candidate)))
            try:
                start_ms = candidate["start_ms"]
                end_ms = candidate["end_ms"]
            except KeyError as exc:
                raise ValueError("candidate start_ms/end_ms must be integers") from exc
            if (
                isinstance(start_ms, bool) or not isinstance(start_ms, int)
                or isinstance(end_ms, bool) or not isinstance(end_ms, int)
            ):
                raise ValueError("candidate start_ms/end_ms must be integers")
            if end_ms <= start_ms:
                raise ValueError("candidate end_ms must be greater than start_ms")
            _probability(models["macro"], candidate.get("macro"), widths["macro"], "macro")
            _probability(models["micro"], candidate.get("micro"), widths["micro"], "micro")
            logistic = _probability(models["verifier_logistic"], candidate.get("verifier"), widths["verifier"], "verifier")
            lgbm = _probability(models["verifier_lgbm"], candidate.get("verifier"), widths["verifier"], "verifier")
            scored.append({
                "subject_id": subject_id, "sid": sid, "start_ms": start_ms, "end_ms": end_ms,
                "score": float(blend_weight * logistic + (1.0 - blend_weight) * lgbm),
            })
    cpu_output = {
        "events": [
            {key: row[key] for key in ("sid", "start_ms", "end_ms", "score")}
            for row in _select_events(scored, policy)
        ],
        "resolved_device": "cpu",
    }
    return cpu_output


def build_smoke_fixture(schema: Mapping[str, object]) -> dict[str, Any]:
    """Build a deterministic precomputed-feature fixture for package parity checks."""

    normalized = _schema_widths(schema)
    return {
        "feature_schema": dict(schema),
        "sessions": [{"subject_id": "fixture-subject", "sid": "fixture-session", "candidates": [{
            "start_ms": 0, "end_ms": 1000,
            "macro": [0.0] * normalized["macro"],
            "micro": [0.0] * normalized["micro"],
            "verifier": [0.0] * normalized["verifier"],
        }]}],
    }


def _trace_runtime(bundle_path: Path):
    """Load frozen raw-runtime state without constructing a Predictor."""
    bundle_path = Path(bundle_path)
    manifest = _read_json(bundle_path / "manifest.json", "bundle manifest")
    run_key = manifest.get("run_key")
    if not isinstance(run_key, str) or not run_key:
        raise ValueError("bundle manifest run key cannot be read")
    bundle = load_event_stack_bundle(bundle_path, expected_role="deployment", expected_run_key=run_key)
    folds = bundle.run_config.get("fold_configs")
    if not isinstance(folds, list) or not folds or not isinstance(folds[0], dict):
        raise ValueError("bundle has no frozen fold configuration")
    frozen = folds[0]
    density = DensityConfig(**frozen["density"])
    micro_candidate = MicroCandidateConfig(**frozen["micro_candidate"])
    macro_config = MacroFeatureConfig(
        window_ms=density.window_ms, stride_ms=density.stride_ms, coverage_min=density.coverage_min,
    )
    micro_config = MicroFeatureConfig(
        window_ms=micro_candidate.window_ms, stride_ms=micro_candidate.stride_ms,
        coverage_min=density.coverage_min, gravity_align=bool(frozen.get("micro_gravity_align")),
    )
    return bundle, density, micro_candidate, macro_config, micro_config


def _trace_probability(model: object, features: np.ndarray, width: int, name: str) -> np.ndarray:
    values = np.asarray(features)
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(f"{name} model features must have width {width}")
    if not len(values):
        return np.empty(0, dtype=np.float64)
    probabilities = np.asarray(model.predict_proba(values), dtype=np.float64)
    columns = np.flatnonzero(np.asarray(model.classes_) == 1)
    if probabilities.shape != (len(values), 2) or len(columns) != 1 or not np.isfinite(probabilities).all():
        raise ValueError(f"{name} model returned incompatible probabilities")
    return probabilities[:, int(columns[0])]


def _trace_by_session(windows, scores: np.ndarray) -> dict[str, list[tuple[int, int, float]]]:
    return {
        sid: [(window.start_ms, window.end_ms, float(score)) for window, score in zip(windows, scores) if window.sid == sid]
        for sid in sorted({window.sid for window in windows})
    }


def _compose_legacy_trace(bundle_path: Path, *, spans, bounds, macro_windows, macro_62, micro_windows,
                          micro_features, subject_by_sid: dict[str, str]):
    """Pre-Predictor candidate, verifier and decoder composition for parity only."""
    from .predictor import InferenceTrace

    bundle, density, micro_candidate, _, _ = _trace_runtime(bundle_path)
    macro_62 = np.asarray(macro_62, dtype=np.float32)
    macro_63 = add_time_prior(macro_62, macro_windows)
    micro_features = np.asarray(micro_features, dtype=np.float32)
    macro_scores = _trace_probability(bundle.models["macro"], macro_63, 63, "macro")
    micro_scores = _trace_probability(bundle.models["micro"], micro_features, 47, "micro")
    macro_by_sid = _trace_by_session(macro_windows, macro_scores)
    micro_by_sid = _trace_by_session(micro_windows, micro_scores)
    candidates = union_candidates(
        density_candidates(macro_by_sid, density),
        micro_candidates(micro_by_sid, float(bundle.run_config["micro_threshold"]), micro_candidate),
    )
    if candidates:
        context = multiscale_verifier_features(
            candidates, macro_by_sid, micro_by_sid, context_features_version="v1", session_bounds_by_sid=bounds,
        )
        logistic = _trace_probability(bundle.models["verifier_logistic"], context, 116, "verifier")
        lgbm = _trace_probability(bundle.models["verifier_lgbm"], context, 116, "verifier")
        scores = float(bundle.policy["blend_weight"]) * logistic + (1.0 - float(bundle.policy["blend_weight"])) * lgbm
    else:
        context = np.empty((0, 116), dtype=np.float64)
        scores = np.empty(0, dtype=np.float64)
    admission = CandidateAdmissionConfig(
        float(bundle.policy["nms_iou"]), float(bundle.policy["admission_threshold"]), bundle.policy["max_candidates_per_subject"],
    )
    admitted_indices = admit_candidates(candidates, scores, [subject_by_sid[item.event.sid] for item in candidates], admission)
    admitted = [candidates[index] for index in admitted_indices]
    admitted_scores = scores[list(admitted_indices)] if admitted_indices else np.empty(0, dtype=np.float64)
    final = apply_event_policy(
        [item.event for item in admitted], admitted_scores, [subject_by_sid[item.event.sid] for item in admitted],
        float(bundle.policy["threshold"]), bundle.policy["max_events_per_group"],
    )
    confidence = {(item.event.sid, item.event.start_ms, item.event.end_ms): float(score) for item, score in zip(admitted, admitted_scores)}
    rows = tuple({"session_id": item.event.sid, "start_ms": item.event.start_ms, "end_ms": item.event.end_ms,
                  "score": float(score), "has_macro": item.macro is not None, "has_micro": item.micro is not None}
                 for item, score in zip(candidates, scores))
    events = tuple({"id": index, "session_id": item.sid, "start_ms": item.start_ms, "end_ms": item.end_ms,
                    "duration_s": (item.end_ms - item.start_ms) / 1000.0,
                    "confidence": confidence[(item.sid, item.start_ms, item.end_ms)]}
                   for index, item in enumerate(final))
    return InferenceTrace(
        spans=tuple(spans), macro_features_62=macro_62, macro_features_63=macro_63,
        micro_features=micro_features, context_features=context[:, 56:], macro_probabilities=macro_scores,
        micro_probabilities=micro_scores, candidates=rows, admitted=tuple(rows[index] for index in admitted_indices),
        verifier_scores=scores, events=events, admission_threshold=float(admission.threshold),
        event_threshold=float(bundle.policy["threshold"]),
    )


def canonical_trace(raw_path: Path, bundle_path: Path, *, session_id: str, subject_id: str | None = None):
    """Trace raw inference through the canonical raw-session reader.

    This compatibility entry point is deliberately trace-only: it records the
    direct source graph for release parity and is not a second inference API.
    """
    _, _, _, macro_config, micro_config = _trace_runtime(bundle_path)
    session = load_raw_session(RawSessionSource(Path(raw_path), session_id, subject_id))
    spans = valid_imu_spans(session)
    bounds = {session_id: (min(item.start_ms for item in spans), max(item.end_ms for item in spans))}
    macro = extract_macro_windows(session, session_id=session_id, config=macro_config)
    micro = extract_micro_windows(session, session_id=session_id, config=micro_config)
    return _compose_legacy_trace(
        bundle_path, spans=[(session_id, int(item.start_ms), int(item.end_ms)) for item in spans], bounds=bounds,
        macro_windows=macro.windows, macro_62=macro.features, micro_windows=micro.windows,
        micro_features=micro.features, subject_by_sid={session_id: subject_id or session_id},
    )


def legacy_trace(raw_path: Path, bundle_path: Path, *, session_id: str, subject_id: str | None = None):
    """Trace the historical loader with the frozen canonical downstream graph.

    The legacy loader remains the independent producer boundary.  Candidate,
    verifier and decoder code are intentionally canonical; duplicating them
    would create an unmaintainable second algorithm implementation.
    """
    from src.data.loader import load_session_tsv

    # The legacy loader and slide feature producer are deliberately retained
    # here as an audit seam; neither enters the shipping Predictor graph.
    session = load_session_tsv(raw_path)
    _, _, _, _, micro_config = _trace_runtime(bundle_path)
    spans = valid_imu_spans(session)
    bounds = {session_id: (min(item.start_ms for item in spans), max(item.end_ms for item in spans))}
    # Legacy slide evidence is cache-backed and uses the original subject/session
    # identifier.  Its window geometry is remapped to the public raw-file ID.
    import sys
    scripts = str(Path(__file__).resolve().parents[3] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import slide_features
    produced = slide_features._process_session((subject_id, "[]", "val"))
    if produced is None:
        raise ValueError("legacy slide producer has no audited session cache")
    macro_62, _, window_rows = produced
    macro_windows = tuple(EventRef(session_id, int(start), int(end)) for _, start, end in window_rows)
    micro = extract_micro_windows(session, session_id=session_id, config=micro_config)
    return _compose_legacy_trace(
        bundle_path, spans=[(session_id, int(item.start_ms), int(item.end_ms)) for item in spans], bounds=bounds,
        macro_windows=macro_windows, macro_62=np.asarray(macro_62, dtype=np.float32),
        micro_windows=micro.windows, micro_features=micro.features,
        subject_by_sid={session_id: subject_id or session_id},
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent / "bundle")
    parser.add_argument("--input-features", type=Path, required=True, help="schema-hashed precomputed candidate feature JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=_DEVICE_CHOICES, default="auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = _read_json(args.input_features, "input feature payload")
        output = predict_feature_payload(args.bundle, payload, device=args.device)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"event-stack inference refused: {exc}", file=sys.stderr)
        return 2
    args.output.write_bytes(_canonical_json(output))
    print(f"resolved_device={output['resolved_device']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
