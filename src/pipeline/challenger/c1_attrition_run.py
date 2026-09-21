"""Sealed, read-only writer for Amendment A1's E03.5 C1 audit.

This boundary deliberately accepts injected replay/audit functions for small
tests, while its defaults use only the attested incumbent replay.  It does not
import any training or registry machinery.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

from src.pipeline.challenger.c1_attrition import MODES, audit_trace
from src.pipeline.challenger.protection import assert_snapshot_unchanged, protected_snapshot
from src.pipeline.challenger.replay import replay_outer_fold


OUTPUT_FILES = (
    "manifest.json", "candidate_attrition.csv", "truth_attrition.csv",
    "counterfactual_events.csv", "counterfactual_summary.json", "oracle.csv", "report.md",
)
_EXPECTED_METRICS = (114, 153, 197)
_TARGET_COLUMNS = ("truth_key", "duration_slice", "fn_stage")
_RUN_KEY = "160afaf81debf1ee"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _value_sha256(value: object) -> str:
    """Hash immutable JSON-compatible provenance without changing it."""
    if isinstance(value, Mapping):
        value = {str(key): item for key, item in value.items()}
    if hasattr(value, "tolist"):
        value = value.tolist()
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _git_sha(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _truth_keys_from_e01(root: Path) -> tuple[str, ...]:
    path = root / "outputs/challenger_70/error_atlas/fn.csv"
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("E01 FN evidence is missing or unsafe")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(name not in reader.fieldnames for name in _TARGET_COLUMNS):
            raise RuntimeError("E01 FN evidence has no target identity columns")
        keys = [str(row["truth_key"]) for row in reader
                if row["duration_slice"] == "short" and row["fn_stage"] == "FN-C1"]
    if len(keys) != 11 or len(set(keys)) != 11:
        raise RuntimeError("E01 must identify exactly eleven unique short FN-C1 truth keys")
    return tuple(keys)


def _fold_of_key(key: str) -> int:
    try:
        fold = int(key.split(":", 1)[0])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid E01 truth key: {key!r}") from exc
    if fold not in range(5):
        raise RuntimeError(f"invalid E01 outer fold in truth key: {key!r}")
    return fold


def _metrics(trace: object) -> tuple[int, int, int]:
    value = trace["metrics"] if isinstance(trace, Mapping) else getattr(trace, "metrics")
    if isinstance(value, tuple) and len(value) == 3:
        return tuple(int(item) for item in value)
    return tuple(int(getattr(value, name)) for name in ("n_tp", "n_true", "n_pred"))


def _assert_output_target(root: Path, output: Path) -> tuple[Path, Path]:
    base = Path(root).resolve()
    expected = base / "outputs/challenger_70/e03_5_c1_attrition"
    target = Path(output).absolute()
    if target != expected:
        raise RuntimeError("E03.5 output must be the exact immutable audit root")
    parent = expected.parent
    if parent.is_symlink() or not parent.is_dir():
        raise RuntimeError("E03.5 output parent is missing or unsafe")
    if expected.exists() or expected.is_symlink():
        raise RuntimeError("E03.5 immutable output already exists")
    return base, expected


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    if not fields:
        fields = ["diagnostic_only"]
        rows = ({"diagnostic_only": True},)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _csv_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _as_rows(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _report(truth_rows: Sequence[Mapping[str, object]], summary: Mapping[str, object]) -> str:
    causes: dict[str, int] = defaultdict(int)
    for row in truth_rows:
        causes[str(row.get("first_total_loss_stage", "unknown"))] += 1
    lines = [
        "# E03.5 C1 Attrition Audit",
        "",
        "> **DIAGNOSTIC ONLY / USES GT for oracle rows.** This sealed-outer audit is not a challenger result and has no performance headline.",
        "",
        "## Cause counts",
        "",
        "| First total-loss stage | C1 short truths | terminal-reason sets |",
        "| --- | ---: | --- |",
    ]
    reasons: dict[str, set[str]] = defaultdict(set)
    for row in truth_rows:
        value = row.get("terminal_reasons", ())
        if isinstance(value, str):
            reasons[str(row.get("first_total_loss_stage", "unknown"))].add(value)
        elif isinstance(value, Sequence):
            reasons[str(row.get("first_total_loss_stage", "unknown"))].update(str(item) for item in value)
    lines.extend(f"| {name} | {count} | {', '.join(sorted(reasons[name])) or '—'} |" for name, count in sorted(causes.items()))
    lines.extend(["", "## Frozen-score single-operation counterfactuals", "",
                  "| Intervention | C1 newly reachable | C1 newly matched | C1 downstream still lost | Non-C1 newly matched/lost | Extra nonmatching candidates | Gross introduced final FP | Removed incumbent FP | Net final FP |",
                  "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |"])
    for mode in MODES:
        row = summary[mode]
        lines.append(
            "| {mode} | {reachable} | {matched} | {lost} | {collateral} / {collateral_lost} | {extra} | {gross} | {removed} | {net} |".format(
                mode=mode, reachable=len(row["newly_reachable_c1_truth_keys"]),
                matched=len(row["newly_matched_c1_truth_keys"]),
                lost=len(row["c1_truth_keys_still_lost_downstream"]),
                collateral=len(row["newly_matched_non_c1_truth_keys"]),
                collateral_lost=len(row["lost_non_c1_truth_keys"]),
                extra=row["extra_nonmatching_candidates"], gross=row["gross_introduced_final_fp"],
                removed=row["removed_incumbent_final_fp"], net=row["net_final_fp_difference"],
            )
        )
    lines.extend(["", "No parameter or challenger recommendation is made here. E04/E05 remain paused pending human review.", ""])
    return "\n".join(lines)


def run_c1_attrition(
    root: Path,
    output: Path,
    *,
    replay: Callable[[Path, int], object] = replay_outer_fold,
    audit: Callable[[object], Mapping[str, object]] = audit_trace,
    snapshot: Callable[[Path], Mapping[str, object]] = protected_snapshot,
    assert_unchanged: Callable[[Mapping[str, object], Path], None] = assert_snapshot_unchanged,
    git_sha: Callable[[Path], str] = _git_sha,
) -> Path:
    """Write exactly seven immutable E03.5 diagnostic files after sealed replay."""
    base, target = _assert_output_target(root, output)
    before = snapshot(base)
    expected_keys = _truth_keys_from_e01(base)
    expected_by_fold: dict[int, set[str]] = defaultdict(set)
    for key in expected_keys:
        expected_by_fold[_fold_of_key(key)].add(key)

    audits: list[Mapping[str, object]] = []
    traces: list[object] = []
    total = [0, 0, 0]
    for fold in range(5):
        trace = replay(base, fold)
        fold_metrics = _metrics(trace)
        total[:] = [left + right for left, right in zip(total, fold_metrics, strict=True)]
        result = audit(trace)
        target_keys = set(str(key) for key in result.get("target_truth_keys", ()))
        if target_keys != expected_by_fold[fold]:
            raise RuntimeError("audit target set does not exactly reproduce E01 short FN-C1 evidence")
        audits.append(result)
        traces.append(trace)
    if tuple(total) != _EXPECTED_METRICS:
        raise RuntimeError("sealed incumbent replay does not reproduce 114/153/197")

    truth_rows = [row for result in audits for row in _as_rows(result.get("truth_rows"))]
    candidate_rows = [row for result in audits for row in _as_rows(result.get("candidate_rows"))]
    oracle_rows = [dict(row, diagnostic_only="DIAGNOSTIC ONLY / USES GT")
                   for result in audits for row in _as_rows(result.get("oracle_rows"))]
    if {str(row.get("truth_key")) for row in truth_rows} != set(expected_keys) or len(truth_rows) != 11:
        raise RuntimeError("audit truth rows do not exactly cover E01's eleven targets")
    if {str(row.get("truth_key")) for row in oracle_rows} != set(expected_keys) or len(oracle_rows) != 11:
        raise RuntimeError("oracle rows do not exactly cover E01's eleven targets")

    summary: dict[str, dict[str, object]] = {}
    event_rows: list[Mapping[str, object]] = []
    for mode in MODES:
        per_fold = [result.get("ablations", {}).get(mode) for result in audits]
        if any(not isinstance(row, Mapping) for row in per_fold):
            raise RuntimeError(f"audit has no complete {mode} result")
        rows = [dict(row) for row in per_fold if isinstance(row, Mapping)]
        keys = ("newly_reachable_c1_truth_keys", "newly_matched_c1_truth_keys",
                "c1_truth_keys_still_lost_downstream", "newly_matched_non_c1_truth_keys",
                "lost_non_c1_truth_keys")
        summary[mode] = {key: tuple(sorted({str(item) for row in rows for item in row.get(key, ())})) for key in keys}
        for key in ("extra_nonmatching_candidates", "gross_introduced_final_fp", "removed_incumbent_final_fp", "net_final_fp_difference"):
            summary[mode][key] = sum(int(row.get(key, 0)) for row in rows)
        for fold, row in enumerate(rows):
            for event in _as_rows(row.get("event_diffs")):
                event_rows.append(dict(event, intervention=mode, outer_fold=fold))

    spec = base / "docs/superpowers/specs/2026-09-21-amendment-a1-e03-5-c1-attrition-design.md"
    e01 = base / "outputs/challenger_70/error_atlas/fn.csv"
    policies: list[object] = []
    scores: list[object] = []
    for result, trace in zip(audits, traces, strict=True):
        policy = result.get("frozen_policy") if isinstance(result, Mapping) else None
        if policy is None:
            policy = trace.get("policy") if isinstance(trace, Mapping) else getattr(trace, "policy", None)
        policies.append(policy)
        scores.append(result.get("frozen_blend_scores") if isinstance(result, Mapping) else None)
    promotion = base / f"models/event_stack/{_RUN_KEY}/promotion_summary.json"
    manifest = {
        "diagnostic_only": True,
        "warning": "DIAGNOSTIC ONLY; oracle rows use GT and cannot be used for promotion or selection.",
        "scope": "E03.5 Candidate-Control Recovery / C1 Attrition Audit",
        "incumbent_run_key": _RUN_KEY,
        "incumbent_pooled_counts": {"tp": 114, "eligible_truth": 153, "predictions": 197},
        "target_truth_keys": list(expected_keys),
        "source_hashes": {"e01_fn_csv_sha256": _sha256(e01), "amendment_spec_sha256": _sha256(spec)},
        "attested_evidence_hashes": ({"promotion_summary_sha256": _sha256(promotion)} if promotion.is_file() else {}),
        "frozen_policy_sha256_by_fold": {str(fold): _value_sha256(policy) for fold, policy in enumerate(policies)},
        "frozen_blend_score_sha256_by_fold": {str(fold): _value_sha256(score) for fold, score in enumerate(scores)},
        "code_sha": git_sha(base), "protected_snapshot": before,
        "output_files": list(OUTPUT_FILES), "registry_written": False,
    }
    stage = target.parent / f".{target.name}.staging-{uuid.uuid4().hex}"
    try:
        stage.mkdir()
        _write_json(stage / "manifest.json", manifest)
        _write_csv(stage / "candidate_attrition.csv", candidate_rows)
        _write_csv(stage / "truth_attrition.csv", truth_rows)
        _write_csv(stage / "counterfactual_events.csv", event_rows)
        _write_json(stage / "counterfactual_summary.json", {"diagnostic_only": True, "interventions": summary})
        _write_csv(stage / "oracle.csv", oracle_rows)
        (stage / "report.md").write_text(_report(truth_rows, summary), encoding="utf-8")
        if tuple(sorted(path.name for path in stage.iterdir())) != tuple(sorted(OUTPUT_FILES)):
            raise RuntimeError("E03.5 staging surface is incomplete")
        assert_unchanged(before, base)
        if target.exists() or target.is_symlink():
            raise RuntimeError("E03.5 immutable output already exists")
        os.replace(stage, target)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    assert_unchanged(before, base)
    return target
