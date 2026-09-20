"""Run the frozen E00 reproduction and E01 error atlas.

This command is deliberately an evidence consumer.  It never imports the
training entry point and does not create the challenger output directory until
all incumbent evidence has been validated and all five folds have replayed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.pipeline.artifacts import verify_promotion_attestation
from src.pipeline.challenger.atlas import (
    atlas_summaries,
    classify_false_negatives,
    false_positive_rows,
    render_report,
)
from src.pipeline.challenger.protection import assert_snapshot_unchanged, protected_snapshot
from src.pipeline.challenger.replay import replay_outer_fold


RUN_KEY = "160afaf81debf1ee"
EXPECTED = {"tp": 114, "eligible": 153, "pred": 197, "fp": 83, "fn": 39}
_REPARSE_POINT = 0x400


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read evidence JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"evidence JSON must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot determine current git SHA") from exc


def _link_or_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        return bool(path.stat(follow_symlinks=False).st_file_attributes & _REPARSE_POINT)
    except (AttributeError, OSError):
        return False


def _validate_output_path(root: Path, output: Path) -> Path:
    """Validate the lexical output boundary before any output-side write."""

    root_abs = Path(os.path.abspath(os.fspath(root)))
    output_abs = Path(os.path.abspath(os.fspath(output)))
    expected = root_abs / "outputs" / "challenger_70"
    if os.path.normcase(str(output_abs)) != os.path.normcase(str(expected)):
        raise ValueError("challenger output must be exactly outputs/challenger_70")
    try:
        relative_parts = output_abs.relative_to(root_abs).parts
    except ValueError as exc:
        raise ValueError("challenger output is outside project root") from exc
    current = root_abs
    for part in relative_parts:
        current = current / part
        if current.exists() and _link_or_reparse(current):
            raise ValueError(f"challenger output path contains symlink/reparse component: {current}")
    protected = (
        root_abs / "release" / "event_stack_incumbent.json",
        root_abs / "models" / "event_stack" / RUN_KEY,
        root_abs / "dist" / "inference",
        root_abs / "dist" / "submission",
        root_abs / "dist" / "visual",
        root_abs / "dist" / "event_stack",
    )
    for path in protected:
        try:
            common = os.path.normcase(os.path.commonpath((str(output_abs), str(path))))
            if common in {os.path.normcase(str(output_abs)), os.path.normcase(str(path))}:
                raise ValueError("challenger output overlaps protected release input")
        except ValueError as exc:
            if "overlaps protected" in str(exc):
                raise
    return output_abs


def validate_registry(root: Path, expected_run_key: str = RUN_KEY) -> Mapping[str, object]:
    """Validate the release pointer and complete attested incumbent surface."""

    root = Path(root).resolve()
    registry = _read_json(root / "release" / "event_stack_incumbent.json")
    actual = registry.get("run_key", registry.get("incumbent_run_key"))
    if actual != expected_run_key:
        raise RuntimeError(f"incumbent registry run key is not {expected_run_key}")
    run_root = root / "models" / "event_stack" / expected_run_key
    problems = verify_promotion_attestation(run_root, expected_run_key=expected_run_key)
    if problems:
        raise RuntimeError("incumbent attestation is invalid: " + "; ".join(problems))
    summary = _read_json(run_root / "promotion_summary.json")
    if summary.get("experiment_key") != expected_run_key:
        raise RuntimeError("incumbent summary run key disagrees with registry")
    metrics = summary.get("outer_metrics")
    if not isinstance(metrics, Mapping):
        raise RuntimeError("incumbent summary has no outer metrics")
    if float(metrics.get("f1", -1.0)) != float(registry.get("f1", -2.0)):
        raise RuntimeError("incumbent registry F1 disagrees with attested summary")
    folds = summary.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise RuntimeError("incumbent summary must contain five folds")
    for fold in range(5):
        bundle = run_root / f"outer-fold-{fold}"
        if not (bundle / "manifest.json").is_file() or not (bundle / "run_config.json").is_file():
            raise RuntimeError(f"incumbent outer-fold-{fold} evidence is incomplete")
    return summary


def atomic_install(target: Path, content: bytes) -> None:
    """Install one file atomically, cleaning its temporary sibling on failure."""

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        return b"\n"
    names = sorted({str(name) for row in rows for name in row})
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in names})
        handle.seek(0)
        return handle.read().encode("utf-8")


def _output_payload(root: Path, summary: Mapping[str, object], traces: Sequence[object], code_sha: str) -> dict[str, bytes]:
    atlas = atlas_summaries(traces)
    fn_rows = [row for trace in traces for row in classify_false_negatives(trace)]
    fp_rows = [row for trace in traces for row in false_positive_rows(trace)]
    metrics = atlas["incumbent_metrics"]
    if metrics != EXPECTED:
        raise RuntimeError(f"challenger aggregate does not match incumbent: {metrics}")
    outer = summary.get("outer_metrics")
    if not isinstance(outer, Mapping):
        raise RuntimeError("attested summary has no aggregate outer metrics")
    baseline = {
        "base_git_sha": code_sha,
        "incumbent_run_key": RUN_KEY,
        "incumbent_f1": float(outer["f1"]),
        "tp": int(outer["n_tp"]), "eligible": int(outer["n_true"]),
        "pred": int(outer["n_pred"]),
        "fp": int(outer["n_pred"]) - int(outer["n_tp"]),
        "fn": int(outer["n_true"]) - int(outer["n_tp"]),
    }
    provenance = {
        "run_key": RUN_KEY,
        "code_sha": code_sha,
        "outer_folds": [trace.fold for trace in traces],
        "iou_threshold": 0.25,
        "replay_only": True,
        "training_called": False,
        "protected_snapshot_sha256": hashlib.sha256(
            _json_bytes({k: list(v) for k, v in protected_snapshot(root).items()})
        ).hexdigest(),
    }
    timestamp = datetime.now(timezone.utc).isoformat()
    registry_rows = [
        {"experiment_id": "E00", "timestamp": timestamp, "git_sha": code_sha,
         "hypothesis": "Frozen incumbent evidence reproduces exactly.",
         "outer_f1": baseline["incumbent_f1"], "tp": EXPECTED["tp"], "fp": EXPECTED["fp"], "status": "complete"},
        {"experiment_id": "E01", "timestamp": timestamp, "git_sha": code_sha,
         "hypothesis": "The deterministic atlas identifies FN/FP failure modes without fitting.",
         "outer_f1": baseline["incumbent_f1"], "tp": EXPECTED["tp"], "fp": EXPECTED["fp"], "status": "complete"},
    ]
    plans = {
        "experiments/E00/plan.json": {
            "experiment_id": "E00", "hypothesis": "Frozen incumbent evidence reproduces exactly.",
            "base_run": RUN_KEY, "code_sha": code_sha, "feature_version": "macro-v1+micro-v1+context-v1",
            "allowed_hyperparameter_grid": {}, "primary_metric": "aggregate_episode_f1",
            "secondary_metrics": ["tp", "fp", "short_recall", "fold_f1"],
            "selection_protocol": "frozen outer evidence replay; no fitting",
        },
        "experiments/E01/plan.json": {
            "experiment_id": "E01", "hypothesis": "The deterministic atlas identifies actionable FN/FP strata without selection.",
            "base_run": RUN_KEY, "code_sha": code_sha, "feature_version": "macro-v1+micro-v1+context-v1",
            "allowed_hyperparameter_grid": {}, "primary_metric": "descriptive_error_atlas",
            "secondary_metrics": ["fn_stage", "fp_taxonomy", "context_v1_residual_separation"],
            "selection_protocol": "frozen outer evidence replay; no fitting",
        },
    }
    payload: dict[str, bytes] = {
        "baseline_audit.json": _json_bytes(baseline),
        "error_atlas/fn.csv": _csv_bytes(fn_rows),
        "error_atlas/fn_summary.json": _json_bytes({**atlas, "rows": len(fn_rows)}),
        "error_atlas/fp.csv": _csv_bytes(fp_rows),
        "error_atlas/fp_summary.json": _json_bytes({"rows": len(fp_rows), "taxonomy": atlas["fp_taxonomy"], "support": atlas["fp_support"], "by_fold": atlas["fp_by_fold"], "by_subject": atlas["fp_by_subject"]}),
        "error_atlas/report.md": render_report(atlas).encode("utf-8"),
        "provenance.json": _json_bytes(provenance),
        "experiment_registry.jsonl": b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for row in registry_rows),
    }
    payload.update({path: _json_bytes(value) for path, value in plans.items()})
    return payload


def _atomic_install_tree(target: Path, payload: Mapping[str, bytes]) -> None:
    target = Path(target)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=parent))
    backup = target.with_name(f".{target.name}.backup-{os.getpid()}")
    moved = False
    try:
        for relative, content in payload.items():
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        if target.exists():
            os.replace(target, backup)
            moved = True
        os.replace(staging, target)
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if target.exists():
            shutil.rmtree(target)
        if moved and backup.exists():
            os.replace(backup, target)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if backup.exists():
            shutil.rmtree(backup)


def run_challenger(root: Path, output: Path) -> Path:
    root = Path(os.path.abspath(os.fspath(root)))
    output = _validate_output_path(root, Path(output))
    summary = validate_registry(root)
    before = protected_snapshot(root)
    traces = tuple(replay_outer_fold(root, fold) for fold in range(5))
    aggregate = (sum(trace.metrics.n_tp for trace in traces), sum(trace.metrics.n_true for trace in traces), sum(trace.metrics.n_pred for trace in traces))
    if aggregate != (EXPECTED["tp"], EXPECTED["eligible"], EXPECTED["pred"]):
        raise RuntimeError(f"E00 aggregate mismatch: {aggregate}")
    payload = _output_payload(root, summary, traces, _git_sha(root))
    _atomic_install_tree(output, payload)
    assert_snapshot_unchanged(before, root)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("outputs/challenger_70"))
    args = parser.parse_args()
    run_challenger(args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
