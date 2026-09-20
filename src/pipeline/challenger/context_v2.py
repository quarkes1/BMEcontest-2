"""Preregistered, inner-only E02 Context-v2 challenger gate.

This module deliberately cannot execute a formal outer evaluation or promote a
model.  It owns only the development/selection evidence required to decide if
such an evaluation may be considered in a later, separately reviewed task.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from src.pipeline.challenger.protection import (
    assert_snapshot_unchanged,
    protected_snapshot,
)
from src.pipeline.context_features_v2 import context_schema_descriptor
from src.pipeline.event_stack import DensityConfig, MicroCandidateConfig
from src.pipeline.runner import (
    FilesystemDataSource,
    InnerOOFResult,
    RunConfig,
    run_outer_train_inner_oof,
)


E02_ID = "E02"
E02_NAMESPACE = "e02-context-v2"
E02_NUMPY_VERSION = "2.4.6"
E02_BASE_RUN = "160afaf81debf1ee"
_EXPECTED_CONFIG_HASHES = (
    "afcf9609a2f6925a", "8b3e27bd2796d038", "b463226db1e6073a",
    "3f26fcb2172cd882", "0199041f39a09f5d",
)


def _validate_experiment_id(experiment_id: str) -> str:
    if not re.fullmatch(r"E02(?:-\d{3})?", experiment_id):
        raise ValueError("E02 experiment id must be E02 or E02-NNN")
    return experiment_id


def registered_e02_config(outer_fold: int, *, workers: int) -> RunConfig:
    """Return the sole permitted E02 surface, differing only by context version."""

    if outer_fold not in range(5):
        raise ValueError("E02 outer_fold must be in [0, 4]")
    if workers < 1:
        raise ValueError("E02 workers must be positive")
    return RunConfig(
        outer_fold=outer_fold, inner_splits=4, seed=20260908, no_tcn=True,
        workers=workers, device="auto", subject_cap_grid=(),
        verifier_feature_mode="probability", verifier_c_grid=(0.1,),
        density=DensityConfig(
            stride_ms=15_000, window_ms=240_000, bridge_ms=60_000,
            density_ms=600_000, min_positive=10, coverage_min=.80,
            merge_ms=120_000, context_ms=1_200_000, coverage_fix=False,
            window_threshold=.28838,
        ),
        micro_enabled=True, context_features_version="v2",
        micro_gravity_align=True,
        micro_threshold_grid=(.10, .20, .30, .40, .50),
        micro_candidate=MicroCandidateConfig(
            stride_ms=7_500, window_ms=15_000, smooth_sigma_ms=30_000,
            smooth_radius_ms=60_000, merge_ms=180_000,
            min_duration_ms=60_000, context_ms=1_200_000,
        ),
        micro_positive_middle_fraction=None, external_fd_weight_grid=(0.0,),
        candidate_control_enabled=True, admission_nms_iou_grid=(.3, .5, .7),
        admission_threshold_grid=(.2, .35, .5, .65),
        admission_subject_cap_grid=(3, 4, 5, 6, 8),
        verifier_blend_weight_grid=(.0, .25, .5, .75, 1.0),
        admission_minimum_recall=.80,
    )


def e02_cache_key(config: RunConfig, *, code_sha: str, numpy_version: str) -> str:
    """Content-address E02 cache independently of the incumbent cache ABI."""

    descriptor = context_schema_descriptor("v2")
    payload = {
        "namespace": E02_NAMESPACE,
        "config": asdict(config),
        "code_sha": code_sha,
        "numpy_version": numpy_version,
        "context_v2_schema_hash": descriptor["ordered_column_sha256"],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes((json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _require_numpy() -> None:
    if np.__version__ != E02_NUMPY_VERSION:
        raise RuntimeError(f"E02 requires NumPy {E02_NUMPY_VERSION}, found {np.__version__}")


def _git_sha(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    )
    sha = completed.stdout.strip()
    if len(sha) != 40 or any(char not in "0123456789abcdef" for char in sha):
        raise RuntimeError("could not resolve a full current git SHA")
    return sha


def _validate_incumbent(root: Path) -> None:
    path = root / "release" / "event_stack_incumbent.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("protected incumbent registry is unreadable") from exc
    if payload.get("run_key") != E02_BASE_RUN or payload.get("f1") != 0.6514285714285715:
        raise RuntimeError("protected incumbent registry does not identify E02 baseline")


def _require_e02_environment(root: Path, *, code_sha: str | None = None) -> str:
    """Check every pre-write safety condition and return the bound source SHA."""

    _require_numpy()
    root = Path(root).resolve()
    _validate_incumbent(root)
    # Snapshot also rejects reparse points under frozen evidence/distribution.
    protected_snapshot(root)
    return code_sha or _git_sha(root)


def _validate_e02_output(root: Path, output: Path, *, experiment_id: str = E02_ID) -> Path:
    root = Path(root).resolve()
    expected = root / "outputs" / "challenger_70" / "experiments" / _validate_experiment_id(experiment_id)
    actual = Path(output).resolve(strict=False)
    if actual != expected:
        raise ValueError("E02 output must be exactly outputs/challenger_70/experiments/E02")
    return expected


def _plan_payload(code_sha: str, *, experiment_id: str = E02_ID, derived_from: str | None = None) -> dict[str, object]:
    descriptor = context_schema_descriptor("v2")
    config = registered_e02_config(0, workers=5)
    return {
        "experiment_id": _validate_experiment_id(experiment_id),
        "hypothesis": "Session-local raw-evidence competition removes false-positive episodes while retaining true positives.",
        "base_run": E02_BASE_RUN,
        "code_sha": code_sha,
        "numpy_version": E02_NUMPY_VERSION,
        "feature_version": "context-v2",
        "context_v2_schema_hash": descriptor["ordered_column_sha256"],
        "allowed_hyperparameter_grid": {
            "fold_config_hashes": list(_EXPECTED_CONFIG_HASHES),
            **asdict(config),
        },
        "primary_metric": "aggregate_episode_f1",
        "secondary_metrics": ["tp", "fp", "short_recall", "fold_f1"],
        "selection_protocol": "outer-train inner-OOF only",
        "development_evidence_warning": "Inner gate is development/selection evidence only, not independent validation.",
    } | ({"derived_from": derived_from} if derived_from is not None else {})


def _identical_json(path: Path, payload: Mapping[str, object]) -> bool:
    try:
        # JSON has no tuple type.  Compare canonical JSON values so immutable
        # preregistration can be reopened by the exact same code/config.
        normalized = json.loads(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return json.loads(path.read_text(encoding="utf-8")) == normalized
    except (OSError, json.JSONDecodeError):
        return False


def write_e02_preregistration(
    root: Path, output: Path, *, code_sha: str | None = None,
    experiment_id: str = E02_ID, derived_from: str | None = None,
) -> Path:
    """Atomically install immutable E02 plan and schema after preflight only."""

    bound_sha = _require_e02_environment(root, code_sha=code_sha)
    experiment_id = _validate_experiment_id(experiment_id)
    output = _validate_e02_output(root, output, experiment_id=experiment_id)
    plan = _plan_payload(bound_sha, experiment_id=experiment_id, derived_from=derived_from)
    descriptor = context_schema_descriptor("v2")
    paths = (output / "plan.json", output / "context_v2_schema.json")
    existing = [path for path in paths if path.exists()]
    if existing:
        if len(existing) != len(paths) or not _identical_json(paths[0], plan) or not _identical_json(paths[1], descriptor):
            raise RuntimeError("E02 preregistration is immutable")
        return paths[0]
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("E02 preregistration output must be empty")
    _atomic_json(paths[0], plan)
    try:
        _atomic_json(paths[1], descriptor)
    except Exception:
        paths[0].unlink(missing_ok=True)
        raise
    return paths[0]


def _metrics_dict(metrics: object) -> dict[str, int | float]:
    return {
        "n_tp": int(getattr(metrics, "n_tp")),
        "n_pred": int(getattr(metrics, "n_pred")),
        "n_true": int(getattr(metrics, "n_true")),
        "f1": float(getattr(metrics, "f1")),
    }


def _pooled(rows: list[Mapping[str, int | float]]) -> dict[str, int | float]:
    tp = sum(int(row["n_tp"]) for row in rows)
    pred = sum(int(row["n_pred"]) for row in rows)
    true = sum(int(row["n_true"]) for row in rows)
    return {"n_tp": tp, "n_pred": pred, "n_true": true, "f1": 2 * tp / (pred + true) if pred + true else 0.0}


def _frozen_v1_inner(root: Path) -> dict[str, int | float]:
    path = root / "models" / "event_stack" / E02_BASE_RUN / "promotion_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("inner_metrics")
    if not isinstance(value, Mapping):
        raise RuntimeError("frozen incumbent inner evidence is missing")
    return {key: (float(value[key]) if key == "f1" else int(value[key])) for key in ("n_tp", "n_pred", "n_true", "f1")}


def gate_passes(report: Mapping[str, object]) -> bool:
    """Apply the complete fixed E02 mechanism gate without discretionary ties."""

    try:
        if report["v1_parity"] is not True:
            return False
        v1, v2 = report["v1"], report["v2"]
        if not isinstance(v1, Mapping) or not isinstance(v2, Mapping):
            return False
        if float(v2["f1"]) <= float(v1["f1"]):
            return False
        if int(v2["n_pred"]) - int(v2["n_tp"]) >= int(v1["n_pred"]) - int(v1["n_tp"]):
            return False
        if int(v1["n_tp"]) - int(v2["n_tp"]) > 2:
            return False
        folds = report["folds"]
        if not isinstance(folds, list) or len(folds) != 5:
            return False
        return sum(float(row["v2_f1"]) >= float(row["v1_f1"]) for row in folds if isinstance(row, Mapping)) >= 3
    except (KeyError, TypeError, ValueError):
        return False


def _write_inner_cache(root: Path, config: RunConfig, result: InnerOOFResult, code_sha: str) -> Path:
    key = e02_cache_key(config, code_sha=code_sha, numpy_version=np.__version__)
    path = root / "cache" / "challenger_70" / E02_NAMESPACE / key / f"fold{config.outer_fold}.json"
    payload = {
        "namespace": E02_NAMESPACE, "config": asdict(config), "code_sha": code_sha,
        "numpy_version": np.__version__, "metrics": _metrics_dict(result.inner_metrics),
        "verifier_feature_count": result.verifier_feature_count,
    }
    if path.exists():
        if not _identical_json(path, payload):
            raise RuntimeError("E02 cache identity collision")
    else:
        _atomic_json(path, payload)
    return path


def _append_terminal_registry(root: Path, report: Mapping[str, object], code_sha: str, *, experiment_id: str) -> None:
    """Append only a stopped gate result; successful full evidence is Task 4."""

    if bool(report.get("passed")):
        return
    path = root / "outputs" / "challenger_70" / "experiment_registry.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line).get("experiment_id") == experiment_id:
                raise RuntimeError("terminal E02 registry row already exists")
    row = {"experiment_id": experiment_id, "status": "stopped_inner_gate", "code_sha": code_sha,
           "hypothesis": _plan_payload(code_sha, experiment_id=experiment_id)["hypothesis"], "gate_passed": False,
           "inner_v1_f1": report["v1"]["f1"], "inner_v2_f1": report["v2"]["f1"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def run_e02_inner_gate(
    root: Path, output: Path, *, workers: int,
    source_factory: Callable[[Path], FilesystemDataSource] = FilesystemDataSource,
    experiment_id: str = E02_ID, derived_from: str | None = None,
) -> dict[str, object]:
    """Run paired v1/v2 outer-train-only selection evidence for all five folds."""

    code_sha = _require_e02_environment(root)
    root = Path(root).resolve()
    experiment_id = _validate_experiment_id(experiment_id)
    output = _validate_e02_output(root, output, experiment_id=experiment_id)
    write_e02_preregistration(root, output, code_sha=code_sha, experiment_id=experiment_id, derived_from=derived_from)
    before = protected_snapshot(root)
    source = source_factory(root)
    folds: list[dict[str, object]] = []
    v1_rows: list[Mapping[str, int | float]] = []
    v2_rows: list[Mapping[str, int | float]] = []
    for outer_fold in range(5):
        v2_config = registered_e02_config(outer_fold, workers=workers)
        v1_config = replace(v2_config, context_features_version="v1")
        data = source.load_outer_train_only(v1_config)
        v1_result = run_outer_train_inner_oof(data, v1_config)
        v2_result = run_outer_train_inner_oof(data, v2_config)
        _write_inner_cache(root, v1_config, v1_result, code_sha)
        _write_inner_cache(root, v2_config, v2_result, code_sha)
        v1 = _metrics_dict(v1_result.inner_metrics)
        v2 = _metrics_dict(v2_result.inner_metrics)
        v1_rows.append(v1); v2_rows.append(v2)
        folds.append({"outer_fold": outer_fold, "v1_f1": v1["f1"], "v2_f1": v2["f1"], "v1": v1, "v2": v2})
    v1, v2 = _pooled(v1_rows), _pooled(v2_rows)
    frozen = _frozen_v1_inner(root)
    v1_parity = v1 == frozen
    report: dict[str, object] = {
        "experiment_id": experiment_id, "phase": "inner-gate", "code_sha": code_sha,
        "numpy_version": np.__version__, "development_evidence": True,
        "not_independent_validation": True, "v1_parity": v1_parity,
        "frozen_v1": frozen, "v1": v1, "v2": v2, "folds": folds,
    }
    report["passed"] = gate_passes(report)
    assert_snapshot_unchanged(before, root)
    target = output / "inner_gate.json"
    if target.exists():
        raise RuntimeError("E02 inner gate is immutable")
    _atomic_json(target, report)
    _append_terminal_registry(root, report, code_sha, experiment_id=experiment_id)
    return report
