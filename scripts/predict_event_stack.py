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
import sys
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np


_MODEL_NAMES = ("macro", "micro", "verifier_logistic", "verifier_lgbm")
_METADATA_NAMES = ("policy.json", "run_config.json", "feature_schema.json")
_DEVICE_CHOICES = ("auto", "cpu", "gpu", "cuda")
# TODO: Add an audited, source-controlled CUDA component identifier here only
# after its implementation and release verification are available.  A package
# file, manifest flag, or arbitrary callable is never a registration protocol.
_SUPPORTED_CUDA_ADAPTER_IDS = frozenset()


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


def _schema_hash(schema: Mapping[str, int]) -> str:
    return hashlib.sha256(
        json.dumps(dict(schema), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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


def _verify_bundle(bundle_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
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
    if set(schema_raw) != {"macro", "micro", "verifier"} or any(
        isinstance(width, bool) or not isinstance(width, int) or width < 1
        for width in schema_raw.values()
    ):
        raise ValueError("feature schema is incompatible with the event-stack runtime")
    return manifest, policy, {name: int(width) for name, width in schema_raw.items()}


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
    _, policy, schema = _verify_bundle(bundle_path)
    _runtime_capabilities(bundle_path)
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
            _probability(models["macro"], candidate.get("macro"), schema["macro"], "macro")
            _probability(models["micro"], candidate.get("micro"), schema["micro"], "micro")
            logistic = _probability(models["verifier_logistic"], candidate.get("verifier"), schema["verifier"], "verifier")
            lgbm = _probability(models["verifier_lgbm"], candidate.get("verifier"), schema["verifier"], "verifier")
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


def build_smoke_fixture(schema: Mapping[str, int]) -> dict[str, Any]:
    """Build a deterministic precomputed-feature fixture for package parity checks."""

    normalized = {name: int(schema[name]) for name in ("macro", "micro", "verifier")}
    return {
        "feature_schema": normalized,
        "sessions": [{"subject_id": "fixture-subject", "sid": "fixture-session", "candidates": [{
            "start_ms": 0, "end_ms": 1000,
            "macro": [0.0] * normalized["macro"],
            "micro": [0.0] * normalized["micro"],
            "verifier": [0.0] * normalized["verifier"],
        }]}],
    }


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
