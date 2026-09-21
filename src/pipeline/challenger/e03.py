"""Preregistered E03 v1 hard-negative inner gate.

This module is deliberately limited to outer-train-only development evidence.
It has no outer-fold execution, release, or distribution entry point.
"""

from __future__ import annotations

import hashlib
import json
import os
import math
import re
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from src.pipeline.challenger.hard_negative import (
    HardNegativeEvidence, assert_outer_subject_disjoint, hard_negative_spec_descriptor,
    mine_hard_negatives,
)
from src.pipeline.challenger.protection import assert_snapshot_unchanged, protected_snapshot
from src.pipeline.challenger.context_v2 import (
    _git_changed_paths, _git_name_status, _git_output, _is_runtime_detritus,
)
from src.pipeline.context_features import CONTEXT_V1_COLUMNS
from src.pipeline.crossfit import crossfit_predict_proba
from src.pipeline.event_stack import EventMetrics
from src.pipeline import runner as _runner
from src.pipeline.runner import FilesystemDataSource, InnerOOFResult, OuterTrainOnlyData, RunConfig, run_outer_train_inner_oof


E03_ID = "E03-001"
E03_NAMESPACE = "e03-v1-hard-negative"
E03_BASE_RUN = "160afaf81debf1ee"
E03_NUMPY_VERSION = "2.4.6"


def registered_e03_config(outer_fold: int, *, workers: int) -> RunConfig:
    """Return E03's sole v1 control surface."""
    if outer_fold not in range(5) or workers < 1:
        raise ValueError("E03 requires outer fold 0..4 and positive workers")
    # This is the exact registered v1 Context-v2-control surface.
    from src.pipeline.challenger.context_v2 import registered_e02_config
    return replace(registered_e02_config(outer_fold, workers=workers), context_features_version="v1")


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _git_sha(root: Path) -> str:
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _e03_evidence_paths(root: Path, output: Path) -> frozenset[str]:
    try:
        relative = Path(output).absolute().relative_to(Path(root).resolve())
    except ValueError as exc:
        raise RuntimeError("E03 evidence output escapes repository root") from exc
    prefix = relative.as_posix()
    return frozenset({
        f"{prefix}/plan.json", f"{prefix}/inner_gate.json",
        "outputs/challenger_70/experiment_registry.jsonl",
    })


def verify_e03_code_identity(root: Path, output: Path, code_sha: str) -> str:
    """Fail closed unless E03's plan still names the executable source.

    This deliberately shares E02's hardened Git semantics: a plan commit may
    only be followed by this experiment's immutable evidence and append-only
    registry; any source, docs, configuration, rename, copy, symlink, or
    relevant dirty path invalidates the identity.
    """
    root, output = Path(root).resolve(), _validate_output(root, output)
    if not isinstance(code_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", code_sha):
        raise RuntimeError("E03 plan has an invalid code SHA")
    if output.is_symlink():
        raise RuntimeError("E03 evidence output may not be a symlink")
    allowed = _e03_evidence_paths(root, output)
    for name in ("plan.json", "inner_gate.json"):
        if (output / name).is_symlink():
            raise RuntimeError("E03 immutable evidence may not be a symlink")
    head = _git_sha(root)
    if subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", code_sha, head]).returncode:
        raise RuntimeError("E03 plan code SHA is not an ancestor of current HEAD")
    changes = _git_name_status(root, code_sha, head)
    for status, paths in changes:
        if status[0] not in {"A", "M"} or len(paths) != 1 or paths[0] not in allowed:
            raise RuntimeError(f"E03 code identity has non-evidence descendant changes: {(status, paths)!r}")
        if status[0] == "M" and paths[0] != "outputs/challenger_70/experiment_registry.jsonl":
            raise RuntimeError(f"E03 immutable evidence was modified: {paths[0]!r}")
    changed = {path for _status, paths in changes for path in paths}
    dirty = _git_changed_paths(root, "diff", "--name-only", "-z")
    dirty |= _git_changed_paths(root, "diff", "--cached", "--name-only", "-z")
    dirty |= _git_changed_paths(root, "ls-files", "--others", "--exclude-standard", "-z")
    dirty |= _git_changed_paths(
        root, "ls-files", "--others", "--ignored", "--exclude-standard", "-z", "--",
        "src", "scripts", "tests", "docs", "requirements.txt", "pyproject.toml", "pytest.ini",
        "setup.cfg", "setup.py", "environment.yml", "environment.yaml", "release",
    )
    dirty = {path for path in dirty if not _is_runtime_detritus(path)}
    if dirty:
        raise RuntimeError(f"E03 code identity has dirty non-evidence paths: {sorted(dirty)!r}")
    for path in sorted(changed & allowed):
        entry = _git_output(root, "ls-tree", head, "--", path).decode("utf-8").strip()
        if not entry or not entry.startswith("100644 blob "):
            raise RuntimeError("E03 immutable evidence must be a regular tracked blob")
    return code_sha


def _require_e03_environment(root: Path, *, code_sha: str | None = None) -> str:
    if np.__version__ != E03_NUMPY_VERSION:
        raise RuntimeError(f"E03 requires NumPy {E03_NUMPY_VERSION}, found {np.__version__}")
    _frozen_outer_anchor(Path(root))
    protected_snapshot(Path(root))
    return code_sha or _git_sha(Path(root))


def _validate_output(root: Path, output: Path) -> Path:
    expected = Path(root).resolve() / "outputs" / "challenger_70" / "experiments" / E03_ID
    if Path(output).resolve(strict=False) != expected:
        raise ValueError("E03 output must be exactly outputs/challenger_70/experiments/E03-001")
    return expected


def _file_fingerprint(path: Path, *, role: str) -> dict[str, object]:
    """Fingerprint one required train-only input without interpreting it."""
    canonical = Path(path).resolve()
    if not canonical.is_file():
        raise FileNotFoundError(f"E03 required {role} input is missing: {canonical}")
    digest = hashlib.sha256()
    with canonical.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = canonical.stat()
    return {"role": role, "path": str(canonical), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}


def e03_train_only_input_fingerprints(
    root: Path, config: RunConfig, *, index_path: Path | None = None, meals_path: Path | None = None,
) -> tuple[dict[str, object], ...]:
    """Return the exact E03 train-only source state, never a validation path."""
    root = Path(root).resolve()
    cache = root / "cache"
    records: list[tuple[str, Path]] = [
        ("macro_window_train", cache / "slide" / f"fold{config.outer_fold}_train.npz"),
        ("macro_meal_train", cache / "slide" / f"fold{config.outer_fold}_meal_train.npz"),
        ("macro_no_meal_train", cache / "slide" / f"fold{config.outer_fold}_no_meal_train.npz"),
        ("outer_split", cache / "splits" / f"fold{config.outer_fold}.json"),
    ]
    if config.micro_enabled:
        records[3:3] = [
            ("micro_window_train", cache / "micro15" / f"fold{config.outer_fold}_train.npz"),
            ("micro_meal_train", cache / "micro15" / f"fold{config.outer_fold}_meal_train.npz"),
            ("micro_no_meal_train", cache / "micro15" / f"fold{config.outer_fold}_no_meal_train.npz"),
        ]
    if index_path is None or meals_path is None:
        from src.data import manifests
        index_path, meals_path = manifests.INDEX_CSV, manifests.MEALS_CSV
    split_path = cache / "splits" / f"fold{config.outer_fold}.json"
    try:
        split = json.loads(split_path.read_text(encoding="utf-8"))
        train_sessions = split["train_sessions"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError("E03 split manifest lacks train session provenance") from exc
    if not isinstance(train_sessions, list) or any(not isinstance(sid, str) or not sid for sid in train_sessions):
        raise RuntimeError("E03 split manifest has invalid train sessions")
    # _micro_expected_metadata and _eligible_truths read these exact train
    # session caches.  Do not inspect val_sessions here: E03's cache identity
    # is intentionally train-only.
    records.extend(("train_session_raw", cache / "sessions" / f"{sid}.npz") for sid in sorted(set(train_sessions)))
    records.extend((("sensor_index_and_bounds", Path(index_path)), ("meal_truth", Path(meals_path))))
    return tuple(_file_fingerprint(path, role=role) for role, path in records)


def assert_e03_outer_subject_disjoint(train_subjects: Sequence[str] | set[str], outer_subjects: Sequence[str] | set[str]) -> None:
    """Reject any held-out subject entering E03 train-only construction."""
    overlap = sorted(set(train_subjects) & set(outer_subjects))
    if overlap:
        raise RuntimeError("E03 outer-validation subject overlap: " + ", ".join(overlap))


def _e03_outer_subject_ids(root: Path, config: RunConfig) -> frozenset[str]:
    """Load only held-out *identifiers* for the mandatory provenance guard."""
    root = Path(root).resolve()
    split_path = root / "cache" / "splits" / f"fold{config.outer_fold}.json"
    try:
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        val_sessions = payload["val_sessions"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError("E03 split manifest lacks outer subject guard") from exc
    if not isinstance(val_sessions, list) or any(not isinstance(sid, str) or not sid for sid in val_sessions):
        raise RuntimeError("E03 split manifest has invalid outer subject identifiers")
    from src.data import manifests
    index = manifests.load_sensor_index()
    mapping = {str(row["session_id"]): str(row["externalid"]) for _, row in index.iterrows()}
    missing = sorted(set(val_sessions) - set(mapping))
    if missing:
        raise RuntimeError("E03 outer subject guard lacks index mappings: " + ", ".join(missing))
    return frozenset(mapping[sid] for sid in val_sessions)


class _E03TrainOnlySource:
    """Narrow source seam: no caller can request an outer fold through E03."""

    def __init__(self, source: object) -> None:
        self._source = source

    def load_outer_train_only(self, config: RunConfig) -> OuterTrainOnlyData:
        return self._source.load_outer_train_only(config)  # type: ignore[attr-defined,no-any-return]

    def load_outer_fold(self, _config: RunConfig) -> object:
        raise RuntimeError("E03 outer-validation loader access is forbidden")


def assert_e03_outer_validation_inaccessible(source: object) -> _E03TrainOnlySource:
    """Install an executable sentinel blocking held-out dataset access."""
    if not callable(getattr(source, "load_outer_train_only", None)):
        raise RuntimeError("E03 requires an outer-train-only loader")
    return _E03TrainOnlySource(source)


def _mining_hash() -> str:
    return hashlib.sha256(json.dumps(hard_negative_spec_descriptor(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def e03_cache_key(config: RunConfig, *, code_sha: str, input_fingerprints: Sequence[Mapping[str, object]], arm: str = "hard_negative_v1") -> str:
    """Content-address only E03 cache inputs; never share an incumbent cache."""
    if arm not in {"threshold_only", "hard_negative_v1"}:
        raise ValueError("E03 cache arm is not registered")
    payload = {"namespace": E03_NAMESPACE, "arm": arm, "config": asdict(config), "code_sha": code_sha,
        "numpy_version": np.__version__, "context_v1_schema_hash": hashlib.sha256("\n".join(CONTEXT_V1_COLUMNS).encode()).hexdigest(),
        "mining_spec_sha256": _mining_hash(), "verifier_parameters": _runner.VERIFIER_MODEL_PARAMETERS,
        "verifier_lgbm_parameters": _runner.VERIFIER_LGBM_PARAMETERS,
        "imputer_contract": "incumbent median imputation", "candidate_order_contract": "canonical raw-union order",
        "policy_tie_order_contract": "incumbent runner tuple order", "input_fingerprints": [dict(x) for x in input_fingerprints]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _frozen_outer_anchor(root: Path) -> dict[str, int | float | str]:
    """Read and cross-check the fixed released outer result; never reselect it."""
    root = Path(root).resolve()
    try:
        registry = json.loads((root / "release" / "event_stack_incumbent.json").read_text(encoding="utf-8"))
        summary = json.loads((root / "models" / "event_stack" / E03_BASE_RUN / "promotion_summary.json").read_text(encoding="utf-8"))
        metrics = summary["outer_metrics"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError("verified frozen E03 incumbent outer evidence is unreadable") from exc
    anchor = {"run_key": E03_BASE_RUN, "tp": int(metrics["n_tp"]), "eligible": int(metrics["n_true"]),
              "pred": int(metrics["n_pred"]), "fp": int(metrics["n_pred"]) - int(metrics["n_tp"]), "f1": float(metrics["f1"])}
    if registry.get("run_key") != E03_BASE_RUN or float(registry.get("f1", -1)) != anchor["f1"]:
        raise RuntimeError("protected incumbent registry disagrees with E03 outer anchor")
    if anchor != {"run_key": E03_BASE_RUN, "tp": 114, "eligible": 153, "pred": 197, "fp": 83, "f1": 0.6514285714285715}:
        raise RuntimeError("protected incumbent evidence does not identify E03 baseline")
    return anchor


def _plan_payload(code_sha: str, *, root: Path | None = None) -> dict[str, object]:
    config = registered_e03_config(0, workers=5)
    return {
        "experiment_id": E03_ID,
        "experiment_slug": E03_NAMESPACE,
        "derived_from": "E02-005",
        "derivation_rationale": "E02-005 failed its TP-loss gate; E03 isolates v1 weighting rather than combining a failed v2 representation.",
        "hypothesis": "Inner-OOF weighted v1 hard negatives reduce episode false positives beyond a same-grid threshold-only control.",
        "base_run": E03_BASE_RUN, "code_sha": code_sha, "numpy_version": E03_NUMPY_VERSION,
        "incumbent_outer_anchor": _frozen_outer_anchor(root) if root is not None else {"run_key": E03_BASE_RUN, "tp": 114, "eligible": 153, "pred": 197, "fp": 83, "f1": 0.6514285714285715},
        "comparison_arms": ["threshold_only", "hard_negative_v1"],
        "anchor_semantics": "read-only verified strict outer evidence; not an inner-selected E03 arm",
        "feature_version": "v1", "context_v1_schema_hash": hashlib.sha256("\n".join(CONTEXT_V1_COLUMNS).encode()).hexdigest(),
        "mining_spec": hard_negative_spec_descriptor(), "mining_spec_sha256": _mining_hash(),
        "allowed_hyperparameter_grid": {**asdict(config), "hard_negative_weight": [2.0]},
        "primary_metric": "aggregate_episode_f1", "secondary_metrics": ["tp", "fp", "short_recall", "fold_f1"],
        "selection_protocol": "outer-train inner-OOF only",
        "candidate_order_contract": "canonical raw-union order is retained through mining and fitting",
        "imputer_contract": "incumbent median imputation; no learned transform outside each fit",
        "policy_tie_order_contract": "incumbent runner tuple order",
        "development_evidence_warning": "Inner gate is development evidence only, not independent validation.",
    }


def write_e03_preregistration(root: Path, output: Path, *, code_sha: str | None = None) -> Path:
    """Atomically write E03's immutable preregistration before a gate run."""
    root, output = Path(root).resolve(), _validate_output(root, output)
    plan = output / "plan.json"
    if plan.exists():
        _require_e03_environment(root)
        try:
            existing = json.loads(plan.read_text(encoding="utf-8"))
            plan_sha = str(existing["code_sha"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError("E03 preregistration is immutable") from exc
        if code_sha is not None and code_sha != plan_sha:
            raise RuntimeError("E03 preregistration is immutable")
        bound = verify_e03_code_identity(root, output, plan_sha)
        if existing != _plan_payload(bound, root=root):
            raise RuntimeError("E03 preregistration is immutable")
        return plan
    bound = _require_e03_environment(root, code_sha=code_sha)
    payload = _plan_payload(bound, root=root)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("E03 preregistration output must be empty")
    _atomic_json(plan, payload)
    return plan


def _write_e03_inner_cache(root: Path, config: RunConfig, *, arm: str, code_sha: str,
                           input_fingerprints: Sequence[Mapping[str, object]], metrics: EventMetrics) -> Path:
    """Write one atomic, isolated E03 cache record bound to all source inputs."""
    key = e03_cache_key(config, code_sha=code_sha, input_fingerprints=input_fingerprints, arm=arm)
    path = Path(root).resolve() / "cache" / "challenger_70" / "e03" / key / f"fold{config.outer_fold}_{arm}.json"
    payload = {"namespace": E03_NAMESPACE, "arm": arm, "config": asdict(config), "code_sha": code_sha,
               "input_fingerprints": [dict(row) for row in input_fingerprints], "metrics": _metrics_dict(metrics)}
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise RuntimeError("E03 cache identity collision")
    else:
        _atomic_json(path, payload)
    return path


def append_e03_terminal_registry(root: Path, report: Mapping[str, object], *, code_sha: str) -> Path:
    """Atomically append one terminal E03 row; duplicate experiment IDs fail closed."""
    path = Path(root).resolve() / "outputs" / "challenger_70" / "experiment_registry.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.parent / f".experiment_registry.{E03_ID}.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("concurrent or stale E03 terminal registry writer") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.close(descriptor)
        existing: list[dict[str, object]] = []
        if path.exists():
            existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(row.get("experiment_id") == E03_ID for row in existing):
            raise RuntimeError("terminal E03 registry row already exists")
        row = {"experiment_id": E03_ID, "code_sha": code_sha, **dict(report)}
        content = "".join(json.dumps(item, sort_keys=True) + "\n" for item in (*existing, row))
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(content, encoding="utf-8", newline="\n")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        lock.unlink(missing_ok=True)
    return path


def _metrics_dict(value: EventMetrics) -> dict[str, int | float]:
    return {"n_tp": int(value.n_tp), "n_pred": int(value.n_pred), "n_true": int(value.n_true), "f1": float(value.f1)}


def e03_gate_passes(control: Sequence[EventMetrics], hard: Sequence[EventMetrics]) -> bool:
    """Apply the fixed four-predicate E03 gate, with no tolerance."""
    if len(control) != 5 or len(hard) != 5:
        return False
    if any(not np.isfinite(item.f1) for item in (*control, *hard)):
        return False
    c_tp, h_tp = sum(x.n_tp for x in control), sum(x.n_tp for x in hard)
    c_pred, h_pred = sum(x.n_pred for x in control), sum(x.n_pred for x in hard)
    c_true, h_true = sum(x.n_true for x in control), sum(x.n_true for x in hard)
    c_f1 = 2 * c_tp / (c_true + c_pred) if c_true + c_pred else 0.0
    h_f1 = 2 * h_tp / (h_true + h_pred) if h_true + h_pred else 0.0
    return h_f1 > c_f1 and (c_pred - c_tp) - (h_pred - h_tp) >= 4 and c_tp - h_tp <= 2 and sum(h.f1 >= c.f1 for c, h in zip(control, hard, strict=True)) >= 3


@dataclass(frozen=True)
class _CandidateEvidence:
    candidates: tuple[object, ...]
    features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray


def _candidate_evidence(data: OuterTrainOnlyData, config: RunConfig) -> _CandidateEvidence:
    """Rebuild frozen raw-union v1 candidates using outer-train data only."""
    _runner._validate_batch(data.window_train, "window_train")
    _runner._validate_batch(data.candidate_train, "candidate_train")
    window_groups = _runner._groups_for(data.window_train.windows, data.subject_by_session)
    candidate_groups = _runner._groups_for(data.candidate_train.windows, data.subject_by_session)
    keep = np.asarray(data.window_train.labels) >= 0
    labels = np.asarray(data.window_train.labels[keep], dtype=np.int8)
    _runner._validate_binary(labels, "window training")
    windows = crossfit_predict_proba(
        _runner._with_time_prior(data.window_train.features, data.window_train.windows)[keep], labels, window_groups[keep],
        _runner._with_time_prior(data.candidate_train.features, data.candidate_train.windows), candidate_groups,
        config.inner_splits, estimator_factory=lambda: _runner._window_estimator(config.seed),
    )
    macro_windows = _runner._windows_by_session(data.candidate_train.windows, windows.probabilities)
    raw_macro = _runner.density_candidates(macro_windows, config.density)
    if not config.micro_enabled or data.micro_window_train is None or data.micro_candidate_train is None:
        raise ValueError("E03 requires the incumbent micro path")
    micro_groups = _runner._groups_for(data.micro_window_train.windows, data.subject_by_session)
    micro_candidate_groups = _runner._groups_for(data.micro_candidate_train.windows, data.subject_by_session)
    micro_keep = _runner._micro_training_keep(data.micro_window_train, data.train_truths, config.micro_positive_middle_fraction)
    micro_labels = np.asarray(data.micro_window_train.labels[micro_keep], dtype=np.int8)
    _runner._validate_binary(micro_labels, "micro window training")
    micro = crossfit_predict_proba(data.micro_window_train.features[micro_keep], micro_labels, micro_groups[micro_keep],
        data.micro_candidate_train.features, micro_candidate_groups, config.inner_splits,
        estimator_factory=lambda: _runner._micro_window_estimator(config.seed + 10))
    micro_windows = _runner._windows_by_session(data.micro_candidate_train.windows, micro.probabilities)
    micro_selection = _runner.select_micro_candidate_threshold(micro_windows, data.train_truths, config.micro_threshold_grid, config.micro_candidate)
    candidates = tuple(_runner.union_candidates(raw_macro, _runner.micro_candidates(micro_windows, micro_selection.threshold, config.micro_candidate)))
    features = _runner.multiscale_verifier_features(candidates, macro_windows, micro_windows, context_features_version="v1",
        session_bounds_by_sid=_runner._context_session_bounds(data, candidates, data.candidate_train.windows, data.micro_candidate_train.windows))
    if features.shape != (len(candidates), 116):
        raise RuntimeError("E03 requires frozen 116D v1 verifier features")
    candidate_labels = _runner._candidate_labels(candidates, data.train_truths)
    groups = _runner._groups_for([candidate.event for candidate in candidates], data.subject_by_session)
    return _CandidateEvidence(candidates, features, candidate_labels, groups)


def build_e03_inner_weights(data: OuterTrainOnlyData, config: RunConfig):
    """Construct weights from logistic-only subject OOF scores, never outer data.

    Candidate reconstruction remains intentionally in the incumbent runner; the
    E03 helper is a narrow guard until the weighted inner evaluator is added.
    """
    if not isinstance(data, OuterTrainOnlyData):
        raise TypeError("E03 inner weights require OuterTrainOnlyData")
    if config.context_features_version != "v1":
        raise ValueError("E03 accepts frozen Context-v1 only")
    evidence = _candidate_evidence(data, config)
    oof = crossfit_predict_proba(evidence.features, evidence.labels, evidence.groups, evidence.features, evidence.groups,
        config.inner_splits, estimator_factory=lambda: _runner._verifier_estimator(config.seed + 20, .1))
    result = mine_hard_negatives(HardNegativeEvidence(evidence.candidates, evidence.features, evidence.labels, oof.probabilities, evidence.groups))
    return evidence, result


def _select_policy(evidence: _CandidateEvidence, scores: np.ndarray, data: OuterTrainOnlyData, config: RunConfig):
    """Use the incumbent candidate-control grid/tie order on inner OOF scores."""
    candidates = evidence.candidates
    choices = []
    admissions = tuple(_runner.CandidateAdmissionConfig(n, threshold, cap)
        for n in config.admission_nms_iou_grid for threshold in config.admission_threshold_grid for cap in config.admission_subject_cap_grid)
    for admission_rank, admission in enumerate(admissions):
        rows = np.asarray(_runner.admit_candidates(candidates, scores, evidence.groups, admission), dtype=np.int64)
        events = [candidates[i].event for i in rows]
        admitted_scores, groups = scores[rows], evidence.groups[rows]
        admission_metrics = _runner.compute_event_metrics(events, data.train_truths)
        policy = _runner.select_event_policy(events, admitted_scores, data.train_truths, groups, max_events_options=config.subject_cap_grid)
        rank = (policy.metrics.f1, policy.metrics.ppv, -len(rows), 0, 0, -admission_rank, -policy.threshold, -float(policy.max_events_per_group or math.inf))
        choices.append((admission_metrics.sensitivity, rank, policy))
    feasible = [choice for choice in choices if choice[0] >= config.admission_minimum_recall]
    return max(feasible or choices, key=lambda x: x[1] if feasible else (x[0], *x[1]))[2]


def _weighted_inner_result(data: OuterTrainOnlyData, config: RunConfig, weights: np.ndarray | None) -> InnerOOFResult:
    evidence = _candidate_evidence(data, config)
    logistic = crossfit_predict_proba(evidence.features, evidence.labels, evidence.groups, evidence.features, evidence.groups, config.inner_splits,
        estimator_factory=lambda: _runner._verifier_estimator(config.seed + 20, .1), fit_sample_weight=weights).probabilities
    lgbm = crossfit_predict_proba(evidence.features, evidence.labels, evidence.groups, evidence.features, evidence.groups, config.inner_splits,
        estimator_factory=lambda: _runner._verifier_lgbm_estimator(config.seed + 21), fit_sample_weight=weights).probabilities
    best = None
    for blend in config.verifier_blend_weight_grid:
        policy = _select_policy(evidence, blend * logistic + (1 - blend) * lgbm, data, config)
        rank = (policy.metrics.f1, policy.metrics.ppv, -blend)
        if best is None or rank > best[0]: best = (rank, policy)
    assert best is not None
    subjects = frozenset(evidence.groups)
    return InnerOOFResult(best[1].metrics, 116, subjects, subjects, subjects)


def run_e03_inner_gate(root: Path, output: Path, *, workers: int, source_factory: Callable[[Path], FilesystemDataSource] = FilesystemDataSource) -> dict[str, object]:
    """Run only the E03 inner gate; formal outer evaluation is intentionally absent."""
    root, output = Path(root).resolve(), _validate_output(root, output)
    _require_e03_environment(root)
    plan_path = output / "plan.json"
    if not plan_path.exists():
        raise RuntimeError("E03 requires a committed immutable preregistration before the gate")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    code_sha = verify_e03_code_identity(root, output, str(plan.get("code_sha")))
    before = protected_snapshot(root)
    source = assert_e03_outer_validation_inaccessible(source_factory(root))
    folds, control_rows, hard_rows = [], [], []
    for fold in range(5):
        config = registered_e03_config(fold, workers=workers)
        input_fingerprints = e03_train_only_input_fingerprints(root, config)
        outer_subject_ids = _e03_outer_subject_ids(root, config)
        data = source.load_outer_train_only(config)
        if not isinstance(data, OuterTrainOnlyData):
            raise TypeError("E03 source must return OuterTrainOnlyData")
        assert_e03_outer_subject_disjoint(set(data.subject_by_session.values()), outer_subject_ids)
        evidence, mined = build_e03_inner_weights(data, config)
        control = _weighted_inner_result(data, config, None)
        hard = _weighted_inner_result(data, config, mined.sample_weight)
        control_rows.append(control.inner_metrics); hard_rows.append(hard.inner_metrics)
        _write_e03_inner_cache(root, config, arm="threshold_only", code_sha=code_sha,
                               input_fingerprints=input_fingerprints, metrics=control.inner_metrics)
        _write_e03_inner_cache(root, config, arm="hard_negative_v1", code_sha=code_sha,
                               input_fingerprints=input_fingerprints, metrics=hard.inner_metrics)
        folds.append({"outer_fold": fold, "threshold_only": _metrics_dict(control.inner_metrics), "hard_negative_v1": _metrics_dict(hard.inner_metrics),
            "outer_validation_accessible": False, "outer_subject_guard": {
                "passed": True, "subject_count": len(outer_subject_ids),
                "subject_ids_sha256": hashlib.sha256("\n".join(sorted(outer_subject_ids)).encode()).hexdigest(),
            }, "input_fingerprints": [dict(row) for row in input_fingerprints],
            "hard_negative": {"hard_count": int(mined.hard_mask.sum()), "category_counts": {key: int(value.sum()) for key, value in mined.category_mask.items()}, "weight_sum": float(mined.sample_weight.sum())}})
    report: dict[str, object] = {"experiment_id": E03_ID, "phase": "inner-gate", "code_sha": code_sha, "development_evidence": True,
        "not_independent_validation": True, "outer_validation_accessible": False,
        "incumbent_outer_anchor": _frozen_outer_anchor(root), "inner_control_arms": ["threshold_only", "hard_negative_v1"], "folds": folds,
        "threshold_only": _pooled_metrics(control_rows), "hard_negative_v1": _pooled_metrics(hard_rows)}
    report["passed"] = e03_gate_passes(control_rows, hard_rows)
    assert_snapshot_unchanged(before, root)
    target = output / "inner_gate.json"
    if target.exists(): raise RuntimeError("E03 inner gate is immutable")
    _atomic_json(target, report)
    append_e03_terminal_registry(root, {"status": "passed_inner_gate" if report["passed"] else "stopped_inner_gate", "passed": bool(report["passed"]),
        "inner_threshold_only_f1": report["threshold_only"]["f1"], "inner_hard_negative_f1": report["hard_negative_v1"]["f1"]}, code_sha=code_sha)
    return report


def _pooled_metrics(values: Sequence[EventMetrics]) -> dict[str, int | float]:
    tp, pred, true = (sum(getattr(v, name) for v in values) for name in ("n_tp", "n_pred", "n_true"))
    return {"n_tp": int(tp), "n_pred": int(pred), "n_true": int(true), "f1": 2 * tp / (true + pred) if true + pred else 0.0}
