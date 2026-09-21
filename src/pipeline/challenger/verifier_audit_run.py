"""Immutable writer for Amendment A2's diagnostic-only E04 verifier audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

from src.pipeline.challenger.protection import assert_snapshot_unchanged, protected_snapshot
from src.pipeline.challenger.replay import replay_outer_fold
from src.pipeline.challenger.verifier_audit import audit_verifier_trace, summarize_verifier_audits


OUTPUT_FILES = (
    "manifest.json", "report.md", "reachable_fn_candidates.csv", "tp_reference.csv",
    "fp_reference.csv", "score_margin_analysis.json", "lr_lgbm_disagreement.json",
    "context_contribution.json", "separability_by_fold.json", "candidate_examples",
)
_EXPECTED_METRICS = (114, 153, 197)
_RUN_KEY = "160afaf81debf1ee"
_E01_STAGES = {"FN-C1", "FN-C2"}
_E01_FOLD_COUNTS = (3, 1, 10, 6, 11)
_SPEC = "docs/superpowers/specs/2026-09-21-amendment-a2-e04-verifier-failure-audit-design.md"
_A1_SPEC = "docs/superpowers/specs/2026-09-21-amendment-a1-e03-5-c1-attrition-design.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _value_sha256(value: object) -> str:
    if isinstance(value, Mapping):
        value = {str(key): item for key, item in value.items()}
    if hasattr(value, "tolist"):
        value = value.tolist()
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _git_sha(root: Path) -> str:
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()


def _metrics(trace: object) -> tuple[int, int, int]:
    value = trace["metrics"] if isinstance(trace, Mapping) else getattr(trace, "metrics")
    if isinstance(value, tuple) and len(value) == 3:
        return tuple(int(item) for item in value)
    return tuple(int(getattr(value, name)) for name in ("n_tp", "n_true", "n_pred"))


def _fold(trace: object) -> int:
    value = trace["fold"] if isinstance(trace, Mapping) else getattr(trace, "fold")
    if not isinstance(value, int) or value not in range(5):
        raise RuntimeError("replay trace has invalid outer fold")
    return value


def _assert_output_target(root: Path, output: Path) -> tuple[Path, Path]:
    base = Path(root).resolve()
    expected = base / "outputs/challenger_70/e04_verifier_audit"
    target = Path(output).absolute()
    if target != expected:
        raise RuntimeError("E04 output must be the exact immutable audit root")
    parent = expected.parent
    if parent.is_symlink() or not parent.is_dir():
        raise RuntimeError("E04 output parent is missing or unsafe")
    if expected.exists() or expected.is_symlink():
        raise RuntimeError("E04 immutable output already exists")
    return base, expected


def _require_windows_no_replace() -> None:
    """Require Windows ``rename`` semantics, which reject an existing target."""
    if os.name != "nt":
        raise RuntimeError("E04 immutable install requires Windows no-replace rename semantics")


def _e01_targets(root: Path) -> tuple[str, ...]:
    path = root / "outputs/challenger_70/error_atlas/fn.csv"
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("E01 FN evidence is missing or unsafe")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(not {"truth_key", "outer_fold", "fn_stage", "union_raw_reachable"}.issubset(row) for row in rows):
        raise RuntimeError("E01 FN evidence has no raw-reachable identity columns")
    targets = [str(row["truth_key"]) for row in rows
               if row["fn_stage"] in _E01_STAGES and str(row["union_raw_reachable"]).lower() == "true"]
    per_fold = Counter(int(row["outer_fold"]) for row in rows
                       if row["fn_stage"] in _E01_STAGES and str(row["union_raw_reachable"]).lower() == "true")
    if len(targets) != 31 or len(set(targets)) != 31 or tuple(per_fold[index] for index in range(5)) != _E01_FOLD_COUNTS:
        raise RuntimeError("E01 must identify exactly 31 raw-reachable FN targets (27 C1, 4 C2; folds 3/1/10/6/11)")
    stages = Counter(row["fn_stage"] for row in rows if str(row["truth_key"]) in set(targets))
    if stages != Counter({"FN-C1": 27, "FN-C2": 4}):
        raise RuntimeError("E01 raw-reachable target stage split is invalid")
    return tuple(targets)


def _rows(value: object) -> list[dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _audit_keys(audit: Mapping[str, object], name: str) -> set[str]:
    return {str(row.get("truth_key")) for row in _rows(audit.get(name))}


def _validate_audits(audits: Sequence[Mapping[str, object]], expected: Sequence[str]) -> None:
    expected_set = set(expected)
    by_fold: dict[int, set[str]] = defaultdict(set)
    for key in expected:
        try:
            fold = int(key.split(":", 1)[0])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid E01 truth key: {key!r}") from exc
        by_fold[fold].add(key)
    if len(audits) != 5 or {audit.get("fold") for audit in audits} != set(range(5)):
        raise RuntimeError("E04 audit requires exactly five distinct folds")
    for audit in audits:
        fold = audit["fold"]
        candidate_keys = _audit_keys(audit, "reachable_fn_candidates")
        if candidate_keys != by_fold[fold]:
            raise RuntimeError("audit target set does not exactly reproduce E01 raw-reachable FN evidence")
        for witness in ("best_iou_witnesses", "best_score_witnesses"):
            if _audit_keys(audit, witness) != by_fold[fold]:
                raise RuntimeError("audit witnesses do not exactly reproduce E01 raw-reachable FN evidence")
    if set().union(*(_audit_keys(audit, "reachable_fn_candidates") for audit in audits)) != expected_set:
        raise RuntimeError("audit target set does not exactly reproduce E01 raw-reachable FN evidence")


def _validate_examples(audits: Sequence[Mapping[str, object]]) -> None:
    """Keep the data-only example subset bounded and tied to best-score witnesses."""
    for audit in audits:
        examples = _rows(audit.get("examples"))
        if len(examples) > 2:
            raise RuntimeError("E04 permits at most two candidate examples per fold")
        witnesses = {(str(row.get("truth_key")), row.get("raw_index"))
                     for row in _rows(audit.get("best_score_witnesses"))}
        coordinates = [(str(row.get("truth_key")), row.get("raw_index")) for row in examples]
        if len(coordinates) != len(set(coordinates)) or not set(coordinates).issubset(witnesses):
            raise RuntimeError("E04 candidate examples must be unique best-score witnesses")
        if any(row.get("diagnostic_only_uses_gt") is not True for row in examples):
            raise RuntimeError("E04 candidate example is not explicitly GT-aware")


def _clean(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return _clean(value.item())
    if isinstance(value, Mapping):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(_clean(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = sorted({str(key) for row in rows for key in row}) or ["diagnostic_only"]
    emitted = rows or ({"diagnostic_only": True},)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in emitted:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _csv_value(value: object) -> object:
    value = _clean(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _trace_field(trace: object, name: str) -> object:
    return trace.get(name) if isinstance(trace, Mapping) else getattr(trace, name, None)


def _example_filename(fold: int, truth_key: object, raw_index: object) -> str:
    """Create a deterministic data filename without incorporating untrusted key syntax."""
    coordinate = f"{fold}|{truth_key}|{raw_index}"
    return f"fold-{fold}_{hashlib.sha256(coordinate.encode('utf-8')).hexdigest()[:16]}.json"


def _report(expected: Sequence[str], summary: Mapping[str, object]) -> str:
    """Render descriptive evidence only; this report cannot express a new policy."""
    mechanisms = dict(summary.get("mechanisms", {}))
    def mechanism_text(name: str) -> tuple[str, str]:
        witness = dict(mechanisms.get(name, {}))
        counts = dict(dict(witness.get("pooled", {})).get("counts", {}))
        support = {str(fold): int(dict(value).get("denominator", 0))
                   for fold, value in dict(witness.get("per_fold", {})).items() if isinstance(value, Mapping)}
        values = ", ".join(f"{label}={counts.get(label, 0)}"
                           for label in ("A", "B", "C", "D", "B_model_disagreement", "unsupported", "unresolved"))
        folds = ", ".join(f"fold {fold}: {count}" for fold, count in sorted(support.items())) or "unsupported"
        return values, folds

    iou_counts, iou_support = mechanism_text("best_iou")
    score_counts, score_support = mechanism_text("best_score")
    disagreement = dict(summary.get("lr_lgbm_disagreement", {}))
    pooled_disagreement = dict(disagreement.get("pooled", {}))
    flagged = dict(pooled_disagreement.get("best_score_fn", {})).get("flagged", "unsupported")
    context = dict(summary.get("context_contribution", {}))
    pooled_context = dict(context.get("pooled", {}))
    score_context = dict(pooled_context.get("best_score_fn", {}))
    lr_context = dict(score_context.get("lr_context", {})).get("median", "unsupported")
    lgbm_context = dict(score_context.get("lgbm_context", {})).get("median", "unsupported")
    separability = dict(summary.get("separability_by_witness", {}))
    score_features = dict(separability.get("best_score", {}))
    stable = sorted(name for name, value in score_features.items()
                    if isinstance(value, Mapping) and value.get("direction") == "stable_descriptive")
    return "\n".join((
        "# E04 Verifier Failure Audit", "",
        "> **DIAGNOSTIC ONLY / USES GT.** This sealed-outer evidence explains frozen incumbent scores; it is not a challenger result.", "",
        "## Raw-reachable false-negative population", "",
        f"The E01-bound target population contains {len(expected)} truths: 27 C1 and 4 C2 (fold counts 3/1/10/6/11).",
        "", "## Mechanism counts", "",
        f"best-IoU witness labels are non-exclusive: {iou_counts}. best-IoU fold support: {iou_support}.",
        f"best-score witness labels are non-exclusive: {score_counts}. best-score fold support: {score_support}.",
        "", "## LR/LGBM and Context findings", "",
        f"The fixed |LR−LGBM| ≥ 0.20 descriptive flag appears on {flagged} best-score witnesses. "
        f"Pooled best-score Context median contributions are LR={lr_context} and LGBM={lgbm_context} in each fitted model's raw-margin space.",
        f"Stable descriptive feature directions among best-score witnesses: {', '.join(stable) if stable else 'none supported'}.",
        "", "## Limits", "",
        "Witness rows are GT-aware. Descriptive score margins, contributions, and feature directions neither tune a threshold nor select a policy or model.", "",
    ))


def run_verifier_audit(
    root: Path, output: Path, *, replay: Callable[[Path, int], object] = replay_outer_fold,
    audit: Callable[[object], Mapping[str, object]] = audit_verifier_trace,
    summarize: Callable[[Sequence[Mapping[str, object]]], Mapping[str, object]] = summarize_verifier_audits,
    snapshot: Callable[[Path], Mapping[str, object]] = protected_snapshot,
    assert_unchanged: Callable[[Mapping[str, object], Path], None] = assert_snapshot_unchanged,
    git_sha: Callable[[Path], str] = _git_sha,
) -> Path:
    """Replay five sealed folds and atomically install the sole E04 artifact root."""
    _require_windows_no_replace()
    base, target = _assert_output_target(root, output)
    before = snapshot(base)
    expected = _e01_targets(base)
    audits: list[Mapping[str, object]] = []
    traces: list[object] = []
    totals = [0, 0, 0]
    for fold in range(5):
        trace = replay(base, fold)
        if _fold(trace) != fold:
            raise RuntimeError("replay returned a trace for the wrong outer fold")
        totals[:] = [left + right for left, right in zip(totals, _metrics(trace), strict=True)]
        result = audit(trace)
        if not isinstance(result, Mapping) or result.get("fold") != fold:
            raise RuntimeError("E04 audit result has invalid fold identity")
        audits.append(result)
        traces.append(trace)
    if tuple(totals) != _EXPECTED_METRICS:
        raise RuntimeError("sealed incumbent replay does not reproduce 114/153/197")
    _validate_audits(audits, expected)
    _validate_examples(audits)
    summary = summarize(audits)
    if not isinstance(summary, Mapping) or set(summary) != {"score_margin_analysis", "lr_lgbm_disagreement", "context_contribution", "mechanisms", "separability_by_witness"}:
        raise RuntimeError("E04 summary is incomplete")

    e01 = base / "outputs/challenger_70/error_atlas/fn.csv"
    spec, a1 = base / _SPEC, base / _A1_SPEC
    promotion = base / f"models/event_stack/{_RUN_KEY}/promotion_summary.json"
    policies = [_trace_field(trace, "policy") for trace in traces]
    blend_scores = [
        [row.get("blend_score") if isinstance(row, Mapping) else None
         for row in (_trace_field(trace, "candidate_rows") or ())]
        for trace in traces
    ]
    manifest = {
        "diagnostic_only": True, "warning": "DIAGNOSTIC ONLY / USES GT; no training, tuning, policy change, release, or promotion.",
        "scope": "E04 sealed verifier failure and separability audit", "incumbent_run_key": _RUN_KEY,
        "incumbent_pooled_counts": {"tp": 114, "eligible_truth": 153, "predictions": 197},
        "target_truth_keys": list(expected), "target_definition": "unmatched E01 FN with raw-union IoU >= 0.25; C0 excluded",
        "group_definitions": {"reachable_fn_candidates": "GT-aware truth/raw candidate pairs", "tp_reference": "matched final events", "fp_reference": "unmatched final events"},
        "source_hashes": {"e01_fn_csv_sha256": _sha256(e01), "amendment_a2_spec_sha256": _sha256(spec), "amendment_a1_spec_sha256": _sha256(a1)},
        "attested_evidence_hashes": ({
            "promotion_summary_sha256": _sha256(promotion),
            **({"promotion_attestation_sha256": _sha256(promotion.with_name("promotion_attestation.json"))}
               if promotion.with_name("promotion_attestation.json").is_file() else {}),
        } if promotion.is_file() else {}),
        "frozen_policy_sha256_by_fold": {str(index): _value_sha256(value) for index, value in enumerate(policies)},
        "frozen_blend_score_sha256_by_fold": {str(index): _value_sha256(value) for index, value in enumerate(blend_scores)},
        "code_sha": git_sha(base), "protected_snapshot": before, "output_files": list(OUTPUT_FILES), "registry_written": False,
    }
    # Verify the frozen release before even creating an output staging surface.
    assert_unchanged(before, base)
    stage = target.parent / f".{target.name}.staging-{uuid.uuid4().hex}"
    try:
        stage.mkdir()
        _write_json(stage / "manifest.json", manifest)
        _write_csv(stage / "reachable_fn_candidates.csv", [row for audit in audits for row in _rows(audit.get("reachable_fn_candidates"))])
        _write_csv(stage / "tp_reference.csv", [row for audit in audits for row in _rows(audit.get("tp_reference"))])
        _write_csv(stage / "fp_reference.csv", [row for audit in audits for row in _rows(audit.get("fp_reference"))])
        for summary_name, filename in (
            ("score_margin_analysis", "score_margin_analysis.json"),
            ("lr_lgbm_disagreement", "lr_lgbm_disagreement.json"),
            ("context_contribution", "context_contribution.json"),
            ("separability_by_witness", "separability_by_fold.json"),
        ):
            _write_json(stage / filename, {"diagnostic_only": True, **dict(summary[summary_name])})
        examples = stage / "candidate_examples"
        examples.mkdir()
        for audit_row in audits:
            for row in _rows(audit_row.get("examples")):
                name = _example_filename(audit_row["fold"], row.get("truth_key"), row.get("raw_index"))
                _write_json(examples / name, {**row, "diagnostic_only_uses_gt": True})
        (stage / "report.md").write_text(_report(expected, summary), encoding="utf-8")
        if tuple(sorted(path.name for path in stage.iterdir())) != tuple(sorted(OUTPUT_FILES)):
            raise RuntimeError("E04 staging surface is incomplete")
        # Both guards occur before the atomic install: a failed guard can only
        # remove our private staging tree, never a concurrently replaced target.
        assert_unchanged(before, base)
        if target.exists() or target.is_symlink():
            raise RuntimeError("E04 immutable output already exists")
        assert_unchanged(before, base)
        # Windows ``os.rename`` fails if an output appeared after the check;
        # unlike replace, it cannot overwrite immutable diagnostic evidence.
        os.rename(stage, target)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return target
