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


def _cuda_available() -> bool:
    """Check CUDA lazily so CPU tree-only environments do not require torch."""

    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def resolve_device(requested: str, *, has_cuda_component: bool) -> str:
    """Resolve an honest backend without ever treating CPU trees as CUDA models."""

    normalized = str(requested).lower()
    if normalized not in _DEVICE_CHOICES:
        raise ValueError("device must be one of: auto, cpu, gpu, cuda")
    if normalized == "gpu":
        normalized = "cuda"
    if normalized == "cpu":
        return "cpu"
    if normalized == "auto":
        return "cuda" if has_cuda_component and _cuda_available() else "cpu"
    if not has_cuda_component:
        raise RuntimeError("forced CUDA requested, but bundle has no CUDA-capable component")
    if not _cuda_available():
        raise RuntimeError("forced CUDA requested, but CUDA is not available")
    return "cuda"


def assert_backend_parity(cpu_output: Mapping[str, Any], cuda_output: Mapping[str, Any]) -> None:
    """Enforce the future CUDA-component parity contract before release.

    A CUDA adapter may differ only in score by at most 1e-5; event identity and
    geometry are a hard equality constraint so a floating-point wobble cannot
    silently alter the decoded event set.
    """

    cpu_events = cpu_output.get("events")
    cuda_events = cuda_output.get("events")
    if not isinstance(cpu_events, list) or not isinstance(cuda_events, list):
        raise ValueError("backend parity outputs must contain event arrays")
    if len(cpu_events) != len(cuda_events):
        raise ValueError("backend parity event geometry differs")
    for cpu_event, cuda_event in zip(cpu_events, cuda_events):
        if not isinstance(cpu_event, Mapping) or not isinstance(cuda_event, Mapping):
            raise ValueError("backend parity event entries must be objects")
        cpu_geometry = tuple(cpu_event.get(key) for key in ("sid", "start_ms", "end_ms"))
        cuda_geometry = tuple(cuda_event.get(key) for key in ("sid", "start_ms", "end_ms"))
        if cpu_geometry != cuda_geometry:
            raise ValueError("backend parity event geometry differs")
        try:
            difference = abs(float(cpu_event["score"]) - float(cuda_event["score"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("backend parity scores must be numeric") from exc
        if not math.isfinite(difference) or difference > 1e-5:
            raise ValueError("backend parity score difference exceeds 1e-5")


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
    """Read optional package capabilities; un-packaged Task-4 bundles are CPU-only."""

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
    if files != actual:
        raise ValueError("runtime manifest checksum does not match package contents")
    bundle_manifest_hash = payload.get("bundle_manifest_sha256")
    if not isinstance(bundle_manifest_hash, str) or _sha256(bundle_path / "manifest.json") != bundle_manifest_hash:
        raise ValueError("runtime manifest checksum does not match bundle manifest")
    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, dict):
        raise ValueError("runtime manifest capabilities must be an object")
    return bool(capabilities.get("has_cuda_component", False))


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


def _select_events(rows: list[dict[str, Any]], policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    admission_threshold = _number(policy, "admission_threshold", 0.0)
    event_threshold = _number(policy, "event_threshold", admission_threshold)
    nms_iou = _number(policy, "nms_iou", 1.0)
    cap_value = policy.get("max_candidates_per_subject")
    cap = None if cap_value is None else int(cap_value)
    if not 0.0 <= admission_threshold <= 1.0 or not 0.0 <= event_threshold <= 1.0 or not 0.0 <= nms_iou <= 1.0:
        raise ValueError("policy thresholds must be in [0, 1]")
    if cap is not None and cap < 1:
        raise ValueError("policy max_candidates_per_subject must be positive or null")
    ranked = sorted(
        (row for row in rows if row["score"] >= admission_threshold and row["score"] >= event_threshold),
        key=lambda row: (-row["score"], row["sid"], row["start_ms"], row["end_ms"]),
    )
    kept: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for row in ranked:
        if cap is not None and counts.get(row["sid"], 0) >= cap:
            continue
        if any(row["sid"] == other["sid"] and _iou(row, other) >= nms_iou for other in kept):
            continue
        kept.append(row)
        counts[row["sid"]] = counts.get(row["sid"], 0) + 1
    return sorted(kept, key=lambda row: (row["sid"], row["start_ms"], row["end_ms"], row["score"]))


def predict_feature_payload(bundle_path: Path, payload: Mapping[str, Any], *, device: str = "auto") -> dict[str, Any]:
    """Score schema-verified precomputed candidate features into canonical events."""

    bundle_path = Path(bundle_path)
    _, policy, schema = _verify_bundle(bundle_path)
    resolved_device = resolve_device(device, has_cuda_component=_runtime_capabilities(bundle_path))
    if not isinstance(payload, Mapping):
        raise ValueError("input feature payload must be an object")
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
    for session in sessions:
        if not isinstance(session, Mapping) or not isinstance(session.get("sid"), str) or not session["sid"]:
            raise ValueError("each session must have a nonempty string sid")
        candidates = session.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("each session candidates value must be an array")
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise ValueError("candidate entries must be objects")
            try:
                start_ms = int(candidate["start_ms"])
                end_ms = int(candidate["end_ms"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("candidate start_ms/end_ms must be integers") from exc
            if end_ms <= start_ms:
                raise ValueError("candidate end_ms must be greater than start_ms")
            _probability(models["macro"], candidate.get("macro"), schema["macro"], "macro")
            _probability(models["micro"], candidate.get("micro"), schema["micro"], "micro")
            logistic = _probability(models["verifier_logistic"], candidate.get("verifier"), schema["verifier"], "verifier")
            lgbm = _probability(models["verifier_lgbm"], candidate.get("verifier"), schema["verifier"], "verifier")
            scored.append({
                "sid": session["sid"], "start_ms": start_ms, "end_ms": end_ms,
                "score": float(blend_weight * logistic + (1.0 - blend_weight) * lgbm),
            })
    return {"events": _select_events(scored, policy), "resolved_device": resolved_device}


def build_smoke_fixture(schema: Mapping[str, int]) -> dict[str, Any]:
    """Build a deterministic precomputed-feature fixture for package parity checks."""

    normalized = {name: int(schema[name]) for name in ("macro", "micro", "verifier")}
    return {
        "feature_schema": normalized,
        "sessions": [{"sid": "fixture-session", "candidates": [{
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
