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


PROMOTION_F1_FLOOR = 0.47863247863247865
_BUNDLE_VERSION = 1
_ROLES = frozenset(("outer-fold-evidence", "deployment"))
_METADATA_FILENAMES = frozenset(
    ("policy.json", "run_config.json", "feature_schema.json")
)


class PromotionContractError(RuntimeError):
    """Promotion cannot proceed without a qualifying, explicit training contract."""


@dataclass(frozen=True)
class EventStackBundle:
    """Fitted models plus immutable inference and evidence metadata."""

    models: Mapping[str, object]
    policy: Mapping[str, object]
    run_config: Mapping[str, object]
    feature_schema: Mapping[str, int]
    metrics: Mapping[str, object]
    source_fingerprints: tuple[Mapping[str, object], ...]
    role: str

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise ValueError(f"role must be one of {sorted(_ROLES)}")
        if not self.models:
            raise ValueError("models must not be empty")
        for name in self.models:
            if not isinstance(name, str) or not name or Path(name).name != name:
                raise ValueError("model names must be nonempty file-name components")
        if any(
            not isinstance(name, str)
            or not isinstance(width, (int, np.integer))
            or isinstance(width, (bool, np.bool_))
            or int(width) < 1
            for name, width in self.feature_schema.items()
        ):
            raise ValueError("feature_schema must map names to positive integer widths")


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


def _trusted_event_stack_root(event_stack_root: Path) -> Path:
    """Normalize an explicit caller-authorized root without accepting symlinks."""

    root = Path(event_stack_root).absolute()
    if root.name != "event_stack":
        raise ValueError("trusted event_stack_root must be named event_stack")
    current = root
    while True:
        if current.exists() and current.is_symlink():
            raise ValueError("trusted event_stack_root must not contain symlink components")
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
        if current.exists() and current.is_symlink():
            raise ValueError("bundle destination must not traverse symlink components")
    if destination.exists() and destination.is_symlink():
        raise ValueError("bundle destination must not be a symlink")
    return destination.parent


def _safe_remove(path: Path) -> None:
    """Remove exactly one known temporary/backup entry without traversing links."""

    if path.is_symlink() or path.is_file():
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
        _stable_json_bytes(
            {key: int(value) for key, value in bundle.feature_schema.items()}
        )
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
    if feature_schema is not None and any(
        not isinstance(name, str)
        or not isinstance(width, (int, np.integer))
        or isinstance(width, (bool, np.bool_))
        or int(width) < 1
        for name, width in feature_schema.items()
    ):
        problems.append("feature schema must map names to positive integer widths")
    if not isinstance(manifest.get("metrics"), dict):
        problems.append("manifest metrics must be an object")
    fingerprints = manifest.get("source_fingerprints")
    if not isinstance(fingerprints, list) or not all(
        isinstance(item, dict) for item in fingerprints
    ):
        problems.append("manifest source_fingerprints must be an array of objects")
    return tuple(problems)


def write_event_stack_bundle(
    destination: Path, bundle: EventStackBundle, *, event_stack_root: Path
) -> None:
    """Write a complete verified bundle then atomically install it at *destination*."""

    destination = Path(destination)
    root = _trusted_event_stack_root(event_stack_root)
    parent = _event_stack_parent(destination, event_stack_root=root)
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or root.is_symlink():
        raise ValueError("trusted event_stack_root must not contain symlink components")
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
    destination: Path, *, expected_role: str | None = None
) -> EventStackBundle:
    """Load a bundle only after all serialized files pass manifest verification."""

    destination = Path(destination)
    problems = verify_bundle_manifest(destination)
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
    )


def _verify_run_bundles(
    run_root: Path, bundles: Mapping[str, EventStackBundle]
) -> None:
    """Verify and deserialize every staged bundle before a run-root install."""

    for key in sorted(bundles):
        bundle_path = run_root / key
        problems = verify_bundle_manifest(bundle_path)
        if problems:
            raise ValueError(
                f"staged bundle {key!r} failed verification: " + "; ".join(problems)
            )
        load_event_stack_bundle(bundle_path, expected_role=bundles[key].role)


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
            if destination.is_symlink():
                raise ValueError("promotion destination must not be a symlink")
            os.replace(destination, backup)
            moved_existing = True
        os.replace(staging, destination)
        promoted_new = True
        _verify_run_bundles(destination, bundles)
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
        if staging.exists() or staging.is_symlink():
            _safe_remove(staging)
        if installed and (backup.exists() or backup.is_symlink()):
            _safe_remove(backup)


def promote_summary(
    summary_path: Path,
    *,
    output_root: Path,
    trainer: PromotionTrainer | None = None,
) -> tuple[Path, ...]:
    """Promote only a strictly improved summary through an injected legal trainer.

    This function creates no paths before both the F1 gate and trainer contract
    have succeeded.  The trainer must return five outer-fold evidence bundles and
    exactly one ``deployment`` bundle; Task 6 owns the legal all-target trainer.
    """

    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    try:
        f1 = float(summary["outer_metrics"]["f1"])
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
    run_key = str(summary.get("experiment_key") or Path(summary_path).stem)
    if Path(run_key).name != run_key:
        raise PromotionContractError("summary experiment key is not a safe directory name")
    for key in sorted(bundles):
        if not isinstance(key, str) or not key or Path(key).name != key:
            raise PromotionContractError("trainer bundle keys must be safe directory names")
    event_stack_root = _trusted_event_stack_root(Path(output_root) / "event_stack")
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
        _install_run_root(destination, staging, bundles)
    finally:
        if staging.exists() or staging.is_symlink():
            _safe_remove(staging)
    return tuple(destination / key for key in sorted(bundles))
