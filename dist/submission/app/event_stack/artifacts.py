"""Verified, atomic persistence for fitted event-stack model bundles.

The module deliberately knows nothing about training data.  A caller must supply
already-fitted models and the exact train-only policy that produced them.  This
keeps artifact writing separate from evaluation and makes promotion callers
provide an explicit, auditable training implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

import joblib
import numpy as np
import sklearn

from event_stack.context_features import CONTEXT_V1_COLUMNS, CONTEXT_V1_SCHEMA_HASH


PROMOTION_F1_FLOOR = 0.5432098765432098
_BUNDLE_VERSION = 1
_ROLES = frozenset(("outer-fold-evidence", "deployment"))
_METADATA_FILENAMES = frozenset(
    ("policy.json", "run_config.json", "feature_schema.json")
)
_PROMOTION_ATTESTATION_VERSION = 1
_PROMOTION_SUMMARY_FILENAME = "promotion_summary.json"
_PROMOTION_ATTESTATION_FILENAME = "promotion_attestation.json"
_INCUMBENT_REGISTRY_VERSION = 1
_INCUMBENT_REGISTRY_FIELDS = frozenset(
    (
        "schema_version",
        "run_key",
        "f1",
        "summary_sha256",
        "attestation_sha256",
        "diagnostic_set_sha256",
    )
)
_RELEASE_TOKENS: set[str] = set()


def _is_canonical_absolute_path(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return path.is_absolute() and str(path.resolve(strict=False)) == value


def _is_source_fingerprint(item: object) -> bool:
    """Accept exactly one canonical present or absent input-state record."""

    if not isinstance(item, dict) or not _is_canonical_absolute_path(item.get("path")):
        return False
    if set(item) == {"path", "missing"}:
        return item["missing"] is True
    if set(item) != {"path", "size", "mtime_ns", "sha256"}:
        return False
    size = item["size"]
    mtime_ns = item["mtime_ns"]
    digest = item["sha256"]
    return (
        isinstance(size, int)
        and not isinstance(size, bool)
        and size >= 0
        and isinstance(mtime_ns, int)
        and not isinstance(mtime_ns, bool)
        and mtime_ns >= 0
        and isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )


class PromotionContractError(RuntimeError):
    """Promotion cannot proceed without a qualifying, explicit training contract."""


def _valid_widths(value: object) -> dict[str, int] | None:
    if not isinstance(value, Mapping) or set(value) != {"macro", "micro", "verifier"}:
        return None
    widths: dict[str, int] = {}
    for name, width in value.items():
        if (
            not isinstance(width, (int, np.integer))
            or isinstance(width, (bool, np.bool_))
            or int(width) < 1
        ):
            return None
        widths[str(name)] = int(width)
    return widths


def normalize_feature_schema(value: object) -> dict[str, object]:
    """Validate schema-v1 width maps and schema-v2 Context-v1 metadata."""

    legacy = _valid_widths(value)
    if legacy is not None:
        return legacy
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version", "widths", "context"
    }:
        raise ValueError("feature_schema must be a v1 width map or a v2 schema object")
    if value["schema_version"] != 2:
        raise ValueError("feature_schema version is unsupported")
    widths = _valid_widths(value["widths"])
    context = value["context"]
    if widths is None or not isinstance(context, Mapping) or set(context) != {
        "version", "columns", "schema_hash"
    }:
        raise ValueError("feature_schema v2 is malformed")
    version = context["version"]
    columns = context["columns"]
    schema_hash = context["schema_hash"]
    if version == "v1":
        if list(columns) != list(CONTEXT_V1_COLUMNS) or schema_hash != CONTEXT_V1_SCHEMA_HASH:
            raise ValueError("feature_schema Context-v1 columns or hash are invalid")
        if widths["verifier"] != 56 + len(CONTEXT_V1_COLUMNS):
            raise ValueError("feature_schema Context-v1 verifier width is invalid")
        normalized_context = {
            "version": "v1",
            "columns": list(CONTEXT_V1_COLUMNS),
            "schema_hash": CONTEXT_V1_SCHEMA_HASH,
        }
    elif version is None:
        if columns != [] or schema_hash is not None:
            raise ValueError("feature_schema without context must have empty context metadata")
        normalized_context = {"version": None, "columns": [], "schema_hash": None}
    else:
        raise ValueError("feature_schema context version is unsupported")
    return {"schema_version": 2, "widths": widths, "context": normalized_context}


@dataclass(frozen=True)
class EventStackBundle:
    """Fitted models plus immutable inference and evidence metadata."""

    models: Mapping[str, object]
    policy: Mapping[str, object]
    run_config: Mapping[str, object]
    feature_schema: Mapping[str, object]
    metrics: Mapping[str, object]
    source_fingerprints: tuple[Mapping[str, object], ...]
    role: str
    diagnostics: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise ValueError(f"role must be one of {sorted(_ROLES)}")
        if not self.models:
            raise ValueError("models must not be empty")
        for name in self.models:
            if not isinstance(name, str) or not name or Path(name).name != name:
                raise ValueError("model names must be nonempty file-name components")
        object.__setattr__(self, "feature_schema", normalize_feature_schema(self.feature_schema))
        fingerprints = self.source_fingerprints
        if not fingerprints or not all(_is_source_fingerprint(item) for item in fingerprints):
            raise ValueError("source_fingerprints must contain canonical present or missing state records")
        paths = [str(item["path"]) for item in fingerprints]
        if len(paths) != len(set(paths)):
            raise ValueError("source_fingerprints must not repeat paths")
        if self.role == "outer-fold-evidence":
            diagnostics = self.diagnostics
            if diagnostics is None:
                diagnostics = {
                    "schema_version": 1,
                    "subjects": {},
                    "distribution": {
                        "included": 0,
                        "excluded_empty": 0,
                        "f1_percentiles": {
                            "p0": None, "p25": None, "p50": None,
                            "p75": None, "p100": None,
                        },
                    },
                    "runtime": {
                        "peak_working_set_bytes": None,
                        "unavailable_reason": "diagnostics were not provided by this legacy trainer",
                        "cuda_peak_bytes": None,
                        "ssl_runtime": None,
                    },
                }
            if not isinstance(diagnostics, Mapping):
                raise ValueError("outer-fold diagnostics must be an object")
            object.__setattr__(self, "diagnostics", dict(diagnostics))


class PromotionTrainer(Protocol):
    """Legal training implementation supplied by Task 6, not this persistence layer."""

    def __call__(self, summary: Mapping[str, object]) -> Mapping[str, EventStackBundle]: ...


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _stable_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            default=_json_default,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _diagnostic_set_sha256(run_root: Path) -> str:
    """Hash the canonical complete outer-fold diagnostic evidence set."""

    entries: dict[str, str] = {}
    for fold in range(5):
        name = f"outer-fold-{fold}"
        path = run_root / name / "diagnostics.json"
        if not path.is_file() or _is_link_or_reparse_point(path):
            raise PromotionContractError(f"incumbent diagnostic evidence is missing: {name}")
        entries[name] = _sha256(path)
    return hashlib.sha256(_stable_json_bytes(entries)).hexdigest()


def incumbent_registry_payload(run_root: Path) -> dict[str, object]:
    """Derive the canonical registry payload from an already verified run root."""

    root = Path(run_root)
    problems = verify_promotion_attestation(root, expected_run_key=root.name)
    if problems:
        raise PromotionContractError(
            "incumbent attestation verification failed: " + "; ".join(problems)
        )
    try:
        summary = json.loads((root / _PROMOTION_SUMMARY_FILENAME).read_text(encoding="utf-8"))
        run_key, f1 = _promotion_summary_contract(summary)
    except (OSError, json.JSONDecodeError, PromotionContractError) as exc:
        raise PromotionContractError("incumbent summary cannot be verified") from exc
    return {
        "schema_version": _INCUMBENT_REGISTRY_VERSION,
        "run_key": run_key,
        "f1": f1,
        "summary_sha256": _sha256(root / _PROMOTION_SUMMARY_FILENAME),
        "attestation_sha256": _sha256(root / _PROMOTION_ATTESTATION_FILENAME),
        "diagnostic_set_sha256": _diagnostic_set_sha256(root),
    }


def load_incumbent_registry(
    path: Path, *, run_root: Path | None = None
) -> dict[str, object]:
    """Load a tracked release floor only when it still matches verified evidence."""

    registry_path = Path(path)
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromotionContractError("incumbent registry cannot be read") from exc
    if not isinstance(payload, dict) or set(payload) != _INCUMBENT_REGISTRY_FIELDS:
        raise PromotionContractError("incumbent registry has an unsupported schema")
    if payload.get("schema_version") != _INCUMBENT_REGISTRY_VERSION:
        raise PromotionContractError("incumbent registry version is unsupported")
    run_key = payload.get("run_key")
    if not isinstance(run_key, str) or not run_key or Path(run_key).name != run_key:
        raise PromotionContractError("incumbent registry run key is invalid")
    try:
        f1 = float(payload["f1"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PromotionContractError("incumbent registry F1 is invalid") from exc
    if not math.isfinite(f1) or not 0.0 <= f1 <= 1.0:
        raise PromotionContractError("incumbent registry F1 is invalid")
    root = Path(run_root) if run_root is not None else (
        registry_path.resolve().parents[1] / "models" / "event_stack" / run_key
    )
    expected = incumbent_registry_payload(root)
    if payload != expected:
        mismatches = [key for key in sorted(_INCUMBENT_REGISTRY_FIELDS) if payload.get(key) != expected.get(key)]
        raise PromotionContractError(
            "incumbent registry does not match attested " + ", ".join(mismatches)
        )
    return dict(payload)


def verify_current_promoted_release(root: Path) -> tuple[str, ...]:
    """Return problems found in the tracked, active promoted release.

    This is intentionally a read-only release lock: it validates the registry,
    its attested run, and the exact active run key without consulting training
    data or rebuildable caches.
    """
    base = Path(root)
    try:
        registry_path = base / "release" / "event_stack_incumbent.json"
        registry = load_incumbent_registry(registry_path)
        run_root = base / "models" / "event_stack" / str(registry["run_key"])
        problems = verify_promotion_attestation(run_root, expected_run_key=str(registry["run_key"]))
        if problems:
            return tuple(problems)
        expected = incumbent_registry_payload(run_root)
        if registry != expected:
            return ("incumbent registry does not match attested release",)
    except (PromotionContractError, OSError, ValueError) as exc:
        return (str(exc),)
    return ()


def load_current_promoted_release(root: Path) -> Mapping[str, object]:
    """Load the immutable promoted release summary after release-lock checks."""
    base = Path(root)
    problems = verify_current_promoted_release(base)
    if problems:
        raise PromotionContractError("current promoted release verification failed: " + "; ".join(problems))
    registry = load_incumbent_registry(base / "release" / "event_stack_incumbent.json")
    run_root = base / "models" / "event_stack" / str(registry["run_key"])
    summary = json.loads((run_root / _PROMOTION_SUMMARY_FILENAME).read_text(encoding="utf-8"))
    return {"run_key": registry["run_key"], "aggregate": summary}


def validate_candidate_against_incumbent(
    candidate_f1: float, incumbent_f1: float
) -> None:
    """Require a strict aggregate F1 improvement over the active release."""

    try:
        candidate = float(candidate_f1)
        incumbent = float(incumbent_f1)
    except (TypeError, ValueError) as exc:
        raise PromotionContractError("candidate and incumbent F1 must be numeric") from exc
    if not math.isfinite(candidate) or not math.isfinite(incumbent):
        raise PromotionContractError("candidate and incumbent F1 must be finite")
    if not candidate > incumbent:
        raise PromotionContractError(
            f"candidate aggregate F1 {candidate:.10f} does not exceed incumbent {incumbent:.10f}"
        )


def issue_release_transaction_token() -> str:
    """Create a single-use in-process capability for the release orchestrator."""

    token = uuid.uuid4().hex
    _RELEASE_TOKENS.add(token)
    return token


def consume_release_transaction_token(token: object) -> None:
    """Consume a capability exactly once, refusing direct active-release writes."""

    if not isinstance(token, str) or token not in _RELEASE_TOKENS:
        raise PromotionContractError(
            "active release updates require a one-use release orchestrator transaction token"
        )
    _RELEASE_TOKENS.remove(token)


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ("git", "rev-parse", "HEAD"),
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _dependency_versions() -> dict[str, str | None]:
    try:
        import lightgbm

        lightgbm_version: str | None = lightgbm.__version__
    except ImportError:
        lightgbm_version = None
    return {
        "python": platform.python_version(),
        "joblib": joblib.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lightgbm_version,
    }


def _is_link_or_reparse_point(path: Path) -> bool:
    """Identify link-like paths without resolving a caller-controlled path."""

    if path.is_symlink():
        return True
    try:
        attributes = path.stat(follow_symlinks=False).st_file_attributes
    except (AttributeError, OSError):
        return False
    return bool(attributes & 0x400)  # Windows FILE_ATTRIBUTE_REPARSE_POINT


def _trusted_event_stack_root(event_stack_root: Path) -> Path:
    """Normalize an explicit caller-authorized root without accepting symlinks."""

    root = Path(event_stack_root).absolute()
    if root.name != "event_stack":
        raise ValueError("trusted event_stack_root must be named event_stack")
    current = root
    while True:
        if current.exists() and _is_link_or_reparse_point(current):
            raise ValueError("trusted event_stack_root must not contain symlink/reparse components")
        if current.parent == current:
            break
        current = current.parent
    return root


def _event_stack_parent(destination: Path, *, event_stack_root: Path) -> Path:
    """Validate that a bundle destination is anchored below one trusted root."""

    root = _trusted_event_stack_root(event_stack_root)
    destination = Path(destination).absolute()
    try:
        relative = destination.relative_to(root)
    except ValueError as exc:
        raise ValueError("bundle destination is outside the trusted event_stack_root") from exc
    if len(relative.parts) not in (1, 2) or destination.name in {"", ".", ".."}:
        raise ValueError(
            "bundle destination must be directly under the trusted root or one run-key beneath it"
        )
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if current.exists() and _is_link_or_reparse_point(current):
            raise ValueError("bundle destination must not traverse symlink/reparse components")
    if destination.exists() and _is_link_or_reparse_point(destination):
        raise ValueError("bundle destination must not be a symlink/reparse point")
    return destination.parent


def _safe_remove(path: Path) -> None:
    """Remove exactly one known temporary/backup entry without traversing links."""

    if _is_link_or_reparse_point(path) or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def cleanup_stale_bundle_temporary_directories(
    destination: Path, *, event_stack_root: Path
) -> tuple[Path, ...]:
    """Remove only this run key's sibling temp/backup paths; never traverse links."""

    destination = Path(destination)
    parent = _event_stack_parent(destination, event_stack_root=event_stack_root)
    if not parent.exists():
        return ()
    prefixes = (
        f".{destination.name}.tmp-",
        f".{destination.name}.backup-",
    )
    removed: list[Path] = []
    for entry in sorted(parent.iterdir(), key=lambda item: item.name):
        if entry.name.startswith(prefixes):
            _safe_remove(entry)
            removed.append(entry)
    return tuple(removed)


def _write_bundle_contents(
    temporary: Path, destination: Path, bundle: EventStackBundle
) -> None:
    temporary.mkdir(parents=False)
    for name in sorted(bundle.models):
        joblib.dump(bundle.models[name], temporary / f"{name}.joblib")
    (temporary / "policy.json").write_bytes(_stable_json_bytes(dict(bundle.policy)))
    (temporary / "run_config.json").write_bytes(
        _stable_json_bytes(dict(bundle.run_config))
    )
    (temporary / "feature_schema.json").write_bytes(
        _stable_json_bytes(bundle.feature_schema)
    )
    if bundle.diagnostics is not None:
        (temporary / "diagnostics.json").write_bytes(
            _stable_json_bytes(dict(bundle.diagnostics))
        )
    files = {
        path.name: _sha256(path)
        for path in sorted(temporary.iterdir(), key=lambda item: item.name)
        if path.is_file()
    }
    manifest = {
        "bundle_version": _BUNDLE_VERSION,
        "run_key": destination.name,
        "role": bundle.role,
        "models": sorted(bundle.models),
        "files": files,
        "git_sha": _git_sha(),
        "source_fingerprints": [dict(item) for item in bundle.source_fingerprints],
        "dependency_versions": _dependency_versions(),
        "metrics": dict(bundle.metrics),
        "development_evidence_warning": (
            "Outer-fold evidence is development evidence only and is not an "
            "untouched generalization estimate."
        ),
    }
    (temporary / "manifest.json").write_bytes(_stable_json_bytes(manifest))


def verify_bundle_manifest(
    destination: Path, *, expected_run_key: str | None = None
) -> tuple[str, ...]:
    """Return deterministic validation messages; an empty tuple means verified."""

    destination = Path(destination)
    problems: list[str] = []
    if not destination.is_dir() or destination.is_symlink():
        return ("bundle directory is missing or is a symlink",)
    manifest_path = destination / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return (f"manifest cannot be read: {exc}",)
    if manifest.get("bundle_version") != _BUNDLE_VERSION:
        problems.append("unsupported bundle version")
    if manifest.get("run_key") != (expected_run_key or destination.name):
        problems.append("manifest run key does not match directory")
    if manifest.get("role") not in _ROLES:
        problems.append("manifest role is invalid")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        problems.append("manifest files must be a nonempty object")
        return tuple(problems)
    models = manifest.get("models")
    valid_model_names = (
        isinstance(models, list)
        and all(
            isinstance(name, str) and name and Path(name).name == name
            for name in models
        )
        and len(set(models)) == len(models)
    )
    if not valid_model_names:
        problems.append("manifest model entries do not match serialized model files")
        expected_model_files: set[str] = set()
    else:
        expected_model_files = {f"{name}.joblib" for name in models}
    expected_names = _METADATA_FILENAMES | expected_model_files
    if manifest.get("role") == "outer-fold-evidence":
        expected_names = expected_names | {"diagnostics.json"}
    actual_names = {
        path.name
        for path in destination.iterdir()
        if path.is_file() and path.name != "manifest.json"
    }
    listed_names = set(files)
    if not _METADATA_FILENAMES <= listed_names or not _METADATA_FILENAMES <= actual_names:
        problems.append("required metadata files are missing")
    if listed_names != expected_names or actual_names != expected_names:
        problems.append("manifest file set does not match required metadata and models")
    for name, expected_hash in sorted(files.items()):
        path = destination / name
        if Path(name).name != name or not path.is_file() or path.is_symlink():
            problems.append(f"invalid manifest file entry: {name}")
        elif not isinstance(expected_hash, str) or _sha256(path) != expected_hash:
            problems.append(f"SHA-256 mismatch for {name}")
    metadata: dict[str, Mapping[str, object]] = {}
    for name in sorted(_METADATA_FILENAMES):
        path = destination / name
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"metadata JSON cannot be read for {name}: {exc}")
            continue
        if not isinstance(payload, dict):
            problems.append(f"metadata JSON must be an object: {name}")
            continue
        metadata[name] = payload
    feature_schema = metadata.get("feature_schema.json")
    if feature_schema is not None:
        try:
            normalize_feature_schema(feature_schema)
        except ValueError as exc:
            problems.append(str(exc))
    if manifest.get("role") == "outer-fold-evidence":
        diagnostic_path = destination / "diagnostics.json"
        try:
            diagnostics = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"diagnostics JSON cannot be read: {exc}")
        else:
            if not isinstance(diagnostics, dict):
                problems.append("diagnostics JSON must be an object")
    if not isinstance(manifest.get("metrics"), dict):
        problems.append("manifest metrics must be an object")
    fingerprints = manifest.get("source_fingerprints")
    if not isinstance(fingerprints, list) or not fingerprints or not all(
        _is_source_fingerprint(item) for item in fingerprints
    ):
        problems.append("manifest source_fingerprints must be canonical present or missing state records")
    elif len({str(item["path"]) for item in fingerprints}) != len(fingerprints):
        problems.append("manifest source_fingerprints must not repeat paths")
    return tuple(problems)


def write_event_stack_bundle(
    destination: Path, bundle: EventStackBundle, *, event_stack_root: Path
) -> None:
    """Write a complete verified bundle then atomically install it at *destination*."""

    destination = Path(destination)
    root = _trusted_event_stack_root(event_stack_root)
    parent = _event_stack_parent(destination, event_stack_root=root)
    parent.mkdir(parents=True, exist_ok=True)
    if _is_link_or_reparse_point(parent) or _is_link_or_reparse_point(root):
        raise ValueError("trusted event_stack_root must not contain symlink/reparse components")
    temporary = parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    backup = parent / f".{destination.name}.backup-{uuid.uuid4().hex}"
    moved_existing = False
    promoted_new = False
    installed = False
    try:
        _write_bundle_contents(temporary, destination, bundle)
        problems = verify_bundle_manifest(temporary, expected_run_key=destination.name)
        if problems:
            raise ValueError("temporary bundle failed verification: " + "; ".join(problems))
        if destination.exists():
            os.replace(destination, backup)
            moved_existing = True
        os.replace(temporary, destination)
        promoted_new = True
        problems = verify_bundle_manifest(destination)
        if problems:
            raise ValueError("installed bundle failed verification: " + "; ".join(problems))
        installed = True
        if backup.exists() or backup.is_symlink():
            _safe_remove(backup)
    except Exception:
        if promoted_new and (destination.exists() or destination.is_symlink()):
            _safe_remove(destination)
        if moved_existing and (backup.exists() or backup.is_symlink()):
            os.replace(backup, destination)
        raise
    finally:
        if temporary.exists() or temporary.is_symlink():
            _safe_remove(temporary)
        if installed and (backup.exists() or backup.is_symlink()):
            _safe_remove(backup)


def load_event_stack_bundle(
    destination: Path, *, expected_role: str | None = None, expected_run_key: str | None = None
) -> EventStackBundle:
    """Load a bundle only after all serialized files pass manifest verification."""

    destination = Path(destination)
    problems = verify_bundle_manifest(destination, expected_run_key=expected_run_key)
    if problems:
        raise ValueError("bundle manifest verification failed: " + "; ".join(problems))
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    role = str(manifest["role"])
    if expected_role is not None and role != expected_role:
        raise ValueError(
            f"bundle role is {role!r}, expected {expected_role!r}; roles are not interchangeable"
        )
    models = {
        name: joblib.load(destination / f"{name}.joblib")
        for name in manifest["models"]
    }
    return EventStackBundle(
        models=models,
        policy=json.loads((destination / "policy.json").read_text(encoding="utf-8")),
        run_config=json.loads((destination / "run_config.json").read_text(encoding="utf-8")),
        feature_schema=json.loads(
            (destination / "feature_schema.json").read_text(encoding="utf-8")
        ),
        metrics=dict(manifest["metrics"]),
        source_fingerprints=tuple(manifest["source_fingerprints"]),
        role=role,
        diagnostics=(
            json.loads((destination / "diagnostics.json").read_text(encoding="utf-8"))
            if (destination / "diagnostics.json").is_file()
            else None
        ),
    )


def _verify_run_bundles(
    run_root: Path, bundles: Mapping[str, EventStackBundle]
) -> None:
    """Verify and deserialize every staged bundle before a run-root install."""

    for key in sorted(bundles):
        bundle_path = run_root / key
        problems = verify_bundle_manifest(bundle_path, expected_run_key=key)
        if problems:
            raise ValueError(
                f"staged bundle {key!r} failed verification: " + "; ".join(problems)
            )
        load_event_stack_bundle(bundle_path, expected_role=bundles[key].role)


def _promotion_summary_contract(summary: Mapping[str, object]) -> tuple[str, float]:
    """Validate the immutable aggregate evidence required for a promotion."""

    try:
        run_key = str(summary["experiment_key"])
        f1 = float(summary["outer_metrics"]["f1"])  # type: ignore[index]
    except (KeyError, TypeError, ValueError) as exc:
        raise PromotionContractError(
            "summary must contain experiment_key and aggregate outer_metrics.f1"
        ) from exc
    if not run_key or Path(run_key).name != run_key:
        raise PromotionContractError("summary experiment key is not a safe directory name")
    if not math.isfinite(f1) or not 0.0 <= f1 <= 1.0:
        raise PromotionContractError("aggregate F1 must be finite and in [0, 1]")
    folds = summary.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise PromotionContractError("summary must contain the complete five outer-fold records")
    seen: set[int] = set()
    for fold in folds:
        if not isinstance(fold, Mapping):
            raise PromotionContractError("summary outer-fold records must be objects")
        value = fold.get("outer_fold")
        config_hash = fold.get("config_hash")
        if isinstance(value, bool) or not isinstance(value, int) or value in seen:
            raise PromotionContractError("summary outer-fold records must have unique integer outer_fold values")
        if not isinstance(config_hash, str) or not config_hash:
            raise PromotionContractError("summary outer-fold records must carry config_hash values")
        seen.add(value)
    if seen != {0, 1, 2, 3, 4}:
        raise PromotionContractError("summary outer-fold records must be exactly folds 0 through 4")
    return run_key, f1


def _write_promotion_attestation(
    run_root: Path,
    summary: Mapping[str, object],
    bundles: Mapping[str, EventStackBundle],
    *,
    expected_run_key: str | None = None,
) -> None:
    """Bind canonical aggregate evidence and every staged bundle to one run key."""

    run_key, f1 = _promotion_summary_contract(summary)
    if expected_run_key is not None and expected_run_key != run_key:
        raise PromotionContractError("promotion run root does not match summary experiment key")
    summary_path = run_root / _PROMOTION_SUMMARY_FILENAME
    summary_path.write_bytes(_stable_json_bytes(dict(summary)))
    attest_bundles = {
        key: {
            "role": bundles[key].role,
            "manifest_sha256": _sha256(run_root / key / "manifest.json"),
            **({"diagnostics_sha256": _sha256(run_root / key / "diagnostics.json")}
               if bundles[key].role == "outer-fold-evidence" else {}),
        }
        for key in sorted(bundles)
    }
    payload = {
        "attestation_version": _PROMOTION_ATTESTATION_VERSION,
        "run_key": run_key,
        "aggregate_summary": {
            "filename": _PROMOTION_SUMMARY_FILENAME,
            "sha256": _sha256(summary_path),
        },
        "gate": {"version": 1, "floor": PROMOTION_F1_FLOOR, "f1": f1},
        "bundles": attest_bundles,
    }
    (run_root / _PROMOTION_ATTESTATION_FILENAME).write_bytes(_stable_json_bytes(payload))


def verify_promotion_attestation(
    run_root: Path, *, expected_run_key: str | None = None
) -> tuple[str, ...]:
    """Verify structural promotion provenance; this is not a cryptographic signature."""

    root = Path(run_root)
    if not root.is_dir() or _is_link_or_reparse_point(root):
        return ("promotion run root is missing or unsafe",)
    try:
        attestation = json.loads((root / _PROMOTION_ATTESTATION_FILENAME).read_text(encoding="utf-8"))
        summary_bytes = (root / _PROMOTION_SUMMARY_FILENAME).read_bytes()
        summary = json.loads(summary_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return (f"promotion attestation cannot be read: {exc}",)
    if not isinstance(attestation, dict) or not isinstance(summary, dict):
        return ("promotion attestation and summary must be objects",)
    problems: list[str] = []
    try:
        run_key, f1 = _promotion_summary_contract(summary)
    except PromotionContractError as exc:
        return (str(exc),)
    if _stable_json_bytes(summary) != summary_bytes:
        problems.append("promotion summary is not canonical JSON")
    if attestation.get("attestation_version") != _PROMOTION_ATTESTATION_VERSION:
        problems.append("promotion attestation version is unsupported")
    if attestation.get("run_key") != run_key or (
        expected_run_key is not None and expected_run_key != run_key
    ) or (expected_run_key is None and root.name != run_key):
        problems.append("promotion attestation run key does not match run root")
    aggregate = attestation.get("aggregate_summary")
    if not isinstance(aggregate, dict) or aggregate.get("filename") != _PROMOTION_SUMMARY_FILENAME or aggregate.get("sha256") != _sha256(root / _PROMOTION_SUMMARY_FILENAME):
        problems.append("promotion attestation aggregate summary hash does not match")
    gate = attestation.get("gate")
    if not isinstance(gate, dict) or gate.get("version") != 1 or gate.get("floor") != PROMOTION_F1_FLOOR or gate.get("f1") != f1 or not f1 > PROMOTION_F1_FLOOR:
        problems.append("promotion attestation gate does not match the qualifying aggregate result")
    expected_roles = {**{f"outer-fold-{fold}": "outer-fold-evidence" for fold in range(5)}, "deployment": "deployment"}
    entries = attestation.get("bundles")
    if not isinstance(entries, dict) or set(entries) != set(expected_roles):
        problems.append("promotion attestation does not bind the complete run structure")
        return tuple(problems)
    for key, role in expected_roles.items():
        entry = entries[key]
        manifest_path = root / key / "manifest.json"
        hash_matches = (
            manifest_path.is_file()
            and isinstance(entry, dict)
            and entry.get("manifest_sha256") == _sha256(manifest_path)
        )
        if not isinstance(entry, dict) or entry.get("role") != role or not hash_matches:
            problems.append(f"promotion attestation manifest hash does not match {key}")
            continue
        if role == "outer-fold-evidence":
            diagnostic_path = root / key / "diagnostics.json"
            if (
                not diagnostic_path.is_file()
                or entry.get("diagnostics_sha256") != _sha256(diagnostic_path)
            ):
                problems.append(f"promotion attestation diagnostics hash does not match {key}")
                continue
        manifest_problems = verify_bundle_manifest(root / key, expected_run_key=key)
        if manifest_problems:
            problems.append(f"promotion attestation bundle is invalid for {key}")
    return tuple(problems)


def _install_run_root(
    destination: Path,
    staging: Path,
    bundles: Mapping[str, EventStackBundle],
) -> None:
    """Atomically replace a complete run root, restoring any verified predecessor."""

    parent = destination.parent
    backup = parent / f".{destination.name}.backup-{uuid.uuid4().hex}"
    moved_existing = False
    promoted_new = False
    installed = False
    try:
        if destination.exists():
            if _is_link_or_reparse_point(destination):
                raise ValueError("promotion destination must not be a symlink/reparse point")
            os.replace(destination, backup)
            moved_existing = True
        os.replace(staging, destination)
        promoted_new = True
        _verify_run_bundles(destination, bundles)
        attestation_problems = verify_promotion_attestation(destination)
        if attestation_problems:
            raise ValueError("installed promotion attestation failed verification: " + "; ".join(attestation_problems))
        installed = True
        if backup.exists() or _is_link_or_reparse_point(backup):
            _safe_remove(backup)
    except Exception:
        if promoted_new and (destination.exists() or _is_link_or_reparse_point(destination)):
            _safe_remove(destination)
        if moved_existing and (backup.exists() or _is_link_or_reparse_point(backup)):
            os.replace(backup, destination)
        raise
    finally:
        if staging.exists() or _is_link_or_reparse_point(staging):
            _safe_remove(staging)
        if installed and (backup.exists() or _is_link_or_reparse_point(backup)):
            _safe_remove(backup)


def promote_summary(
    summary_path: Path,
    *,
    output_root: Path,
    trainer: PromotionTrainer | None = None,
    release_token: str | None = None,
    active_release: bool = False,
) -> tuple[Path, ...]:
    """Promote only a strictly improved summary through an injected legal trainer.

    This function creates no paths before both the F1 gate and trainer contract
    have succeeded.  The trainer must return five outer-fold evidence bundles and
    exactly one ``deployment`` bundle; Task 6 owns the legal all-target trainer.
    """

    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping):
        raise PromotionContractError("summary must be a JSON object")
    # Keep the gate first: malformed/non-improving inputs still cause no writes.
    try:
        f1 = float(summary["outer_metrics"]["f1"])  # type: ignore[index]
    except (KeyError, TypeError, ValueError) as exc:
        raise PromotionContractError("summary lacks aggregate outer_metrics.f1") from exc
    if not math.isfinite(f1) or not 0.0 <= f1 <= 1.0:
        raise PromotionContractError("aggregate F1 must be finite and in [0, 1]")
    if not f1 > PROMOTION_F1_FLOOR:
        raise PromotionContractError(
            f"aggregate F1 must be strictly greater than {PROMOTION_F1_FLOOR:.10f}; got {f1:.10f}"
        )
    if trainer is None:
        raise PromotionContractError(
            "no legal full-target trainer is registered; inject a trainer that "
            "constructs subject-safe full-target data before promotion"
        )
    run_key, _ = _promotion_summary_contract(summary)
    bundles = trainer(summary)
    if not isinstance(bundles, Mapping) or not bundles:
        raise PromotionContractError("trainer must return a nonempty mapping of bundles")
    if any(not isinstance(bundle, EventStackBundle) for bundle in bundles.values()):
        raise PromotionContractError("trainer must return EventStackBundle values")
    deployment = [key for key, bundle in bundles.items() if bundle.role == "deployment"]
    evidence = [key for key, bundle in bundles.items() if bundle.role == "outer-fold-evidence"]
    if len(deployment) != 1 or len(evidence) != 5 or len(bundles) != 6:
        raise PromotionContractError(
            "trainer must return exactly five outer-fold-evidence bundles and one deployment bundle"
        )
    for key in sorted(bundles):
        if not isinstance(key, str) or not key or Path(key).name != key:
            raise PromotionContractError("trainer bundle keys must be safe directory names")
    event_stack_root = _trusted_event_stack_root(Path(output_root) / "event_stack")
    if active_release or Path(output_root).absolute() == (Path(__file__).resolve().parents[2] / "models").absolute():
        consume_release_transaction_token(release_token)
    destination = event_stack_root / run_key
    _event_stack_parent(destination / "deployment", event_stack_root=event_stack_root)
    event_stack_root.mkdir(parents=True, exist_ok=True)
    if event_stack_root.is_symlink():
        raise ValueError("trusted event_stack_root must not contain symlink components")
    staging = event_stack_root / f".{run_key}.staging-{uuid.uuid4().hex}"
    try:
        staging.mkdir(parents=False)
        for key in sorted(bundles):
            bundle_destination = staging / key
            _write_bundle_contents(bundle_destination, bundle_destination, bundles[key])
        _verify_run_bundles(staging, bundles)
        _write_promotion_attestation(staging, summary, bundles)
        attestation_problems = verify_promotion_attestation(
            staging, expected_run_key=run_key
        )
        if attestation_problems:
            raise ValueError(
                "staged promotion attestation failed verification: "
                + "; ".join(attestation_problems)
            )
        _install_run_root(destination, staging, bundles)
    finally:
        if staging.exists() or _is_link_or_reparse_point(staging):
            _safe_remove(staging)
    return tuple(destination / key for key in sorted(bundles))
