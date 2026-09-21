"""Sealed-output contracts for the E03.5 C1 attrition diagnostic."""

from __future__ import annotations

import csv
from pathlib import Path
from types import MappingProxyType

import pytest


def _fn_csv(root: Path, keys: list[str]) -> None:
    target = root / "outputs/challenger_70/error_atlas/fn.csv"
    target.parent.mkdir(parents=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("truth_key", "duration_slice", "fn_stage"))
        writer.writeheader()
        for key in keys:
            writer.writerow({"truth_key": key, "duration_slice": "short", "fn_stage": "FN-C1"})
    spec = root / "docs/superpowers/specs/2026-09-21-amendment-a1-e03-5-c1-attrition-design.md"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text("test amendment\n", encoding="utf-8")


def _audit(keys: list[str]) -> dict[str, object]:
    return {
        "target_truth_keys": tuple(keys),
        "truth_rows": tuple({"truth_key": key, "outer_fold": index % 5,
                               "first_total_loss_stage": "nms", "terminal_reasons": "nms_suppressed"}
                            for index, key in enumerate(keys)),
        "candidate_rows": (),
        "oracle_rows": tuple({"truth_key": key, "outcome": "DIAGNOSTIC ONLY / USES GT"} for key in keys),
        "ablations": {
            mode: {"newly_admitted_indices": (), "extra_nonmatching_candidates": 0,
                   "newly_reachable_c1_truth_keys": (), "newly_matched_c1_truth_keys": (),
                   "c1_truth_keys_still_lost_downstream": (), "newly_matched_non_c1_truth_keys": (),
                   "lost_non_c1_truth_keys": (), "gross_introduced_final_fp": 0,
                   "removed_incumbent_final_fp": 0, "net_final_fp_difference": 0, "event_diffs": ()}
            for mode in ("disable_nms_only", "disable_admission_threshold_only", "disable_subject_cap_only")
        },
    }


def _metrics(fold: int) -> tuple[int, int, int]:
    return ((22, 30, 39), (22, 30, 39), (22, 30, 39), (22, 30, 39), (26, 33, 41))[fold]


def test_runner_atomically_emits_exactly_seven_diagnostic_files_without_registry_write(tmp_path: Path):
    """The sealed audit has one immutable artifact surface and no performance registry side effect."""
    from src.pipeline.challenger.c1_attrition_run import OUTPUT_FILES, run_c1_attrition

    keys = [f"{index % 5}:sid-{index}:0:1" for index in range(11)]
    _fn_csv(tmp_path, keys)
    registry = tmp_path / "outputs/challenger_70/experiment_registry.jsonl"
    registry.write_text('{"historical": true}\n', encoding="utf-8")
    output = tmp_path / "outputs/challenger_70/e03_5_c1_attrition"

    installed = run_c1_attrition(
        tmp_path, output,
        replay=lambda _root, fold: {"fold": fold, "metrics": _metrics(fold)},
        audit=lambda trace: _audit([key for index, key in enumerate(keys) if index % 5 == trace["fold"]]),
        snapshot=lambda _root: {"protected": ("file", "before")},
        assert_unchanged=lambda _before, _root: None,
        git_sha=lambda _root: "a" * 40,
    )

    assert installed == output
    assert tuple(sorted(path.name for path in output.iterdir())) == tuple(sorted(OUTPUT_FILES))
    assert registry.read_text(encoding="utf-8") == '{"historical": true}\n'
    assert "DIAGNOSTIC ONLY" in (output / "manifest.json").read_text(encoding="utf-8")
    assert "outer_f1" not in (output / "report.md").read_text(encoding="utf-8")


def test_runner_rejects_non_exact_e01_scope_and_existing_or_symlink_output(tmp_path: Path):
    """An audit may not silently redefine E01's eleven targets or replace evidence."""
    from src.pipeline.challenger.c1_attrition_run import run_c1_attrition

    keys = [f"{index % 5}:sid-{index}:0:1" for index in range(11)]
    _fn_csv(tmp_path, keys)
    output = tmp_path / "outputs/challenger_70/e03_5_c1_attrition"
    common = dict(
        replay=lambda _root, fold: {"fold": fold, "metrics": _metrics(fold)},
        snapshot=lambda _root: {}, assert_unchanged=lambda _before, _root: None,
        git_sha=lambda _root: "a" * 40,
    )
    with pytest.raises(RuntimeError, match="E01"):
        run_c1_attrition(tmp_path, output, audit=lambda trace: _audit(["wrong"]), **common)
    output.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="immutable"):
        run_c1_attrition(tmp_path, output, audit=lambda trace: _audit([]), **common)


def test_runner_rejects_bad_incumbent_parity_before_it_can_write(tmp_path: Path):
    """A non-attested replay total is a hard stop, not a diagnostic result."""
    from src.pipeline.challenger.c1_attrition_run import run_c1_attrition

    keys = [f"{index % 5}:sid-{index}:0:1" for index in range(11)]
    _fn_csv(tmp_path, keys)
    output = tmp_path / "outputs/challenger_70/e03_5_c1_attrition"
    with pytest.raises(RuntimeError, match="114/153/197"):
        run_c1_attrition(
            tmp_path, output, replay=lambda _root, fold: {"fold": fold, "metrics": (1, 1, 1)},
            audit=lambda trace: _audit([key for key in keys if int(key.split(":", 1)[0]) == trace["fold"]]),
            snapshot=lambda _root: {}, assert_unchanged=lambda _before, _root: None,
            git_sha=lambda _root: "a" * 40,
        )
    assert not output.exists()


def test_runner_cleans_staging_when_protected_snapshot_changes(tmp_path: Path):
    """A snapshot failure is all-or-nothing: neither final nor staging output survives."""
    from src.pipeline.challenger.c1_attrition_run import run_c1_attrition

    keys = [f"{index % 5}:sid-{index}:0:1" for index in range(11)]
    _fn_csv(tmp_path, keys)
    output = tmp_path / "outputs/challenger_70/e03_5_c1_attrition"
    calls = 0

    def unchanged(_before, _root):
        nonlocal calls
        calls += 1
        raise RuntimeError("protected release input changed")

    with pytest.raises(RuntimeError, match="protected release"):
        run_c1_attrition(
            tmp_path, output, replay=lambda _root, fold: {"fold": fold, "metrics": _metrics(fold)},
            audit=lambda trace: _audit([key for key in keys if int(key.split(":", 1)[0]) == trace["fold"]]),
            snapshot=lambda _root: {"release": ("file", "frozen")}, assert_unchanged=unchanged,
            git_sha=lambda _root: "a" * 40,
        )
    assert calls == 1
    assert not output.exists()
    assert not list(output.parent.glob(".e03_5_c1_attrition.staging-*"))


def test_runner_cleans_staging_on_writer_failure_and_never_calls_model_methods(tmp_path: Path, monkeypatch):
    """A write fault cannot leave partial evidence, and the runner's seam is replay-only."""
    import src.pipeline.challenger.c1_attrition_run as runner

    keys = [f"{index % 5}:sid-{index}:0:1" for index in range(11)]
    _fn_csv(tmp_path, keys)
    output = tmp_path / "outputs/challenger_70/e03_5_c1_attrition"
    original = runner._write_json

    def fail_summary(path, value):
        if path.name == "counterfactual_summary.json":
            raise OSError("simulated writer failure")
        original(path, value)

    monkeypatch.setattr(runner, "_write_json", fail_summary)
    with pytest.raises(OSError, match="simulated writer"):
        runner.run_c1_attrition(
            tmp_path, output, replay=lambda _root, fold: {"fold": fold, "metrics": _metrics(fold)},
            audit=lambda trace: _audit([key for key in keys if int(key.split(":", 1)[0]) == trace["fold"]]),
            snapshot=lambda _root: {}, assert_unchanged=lambda _before, _root: None,
            git_sha=lambda _root: "a" * 40,
        )
    assert not output.exists()
    assert not list(output.parent.glob(".e03_5_c1_attrition.staging-*"))


def test_policy_hash_normalizes_readonly_mappings():
    """Attested read-only policy metadata must hash by values, never its repr/address."""
    from src.pipeline.challenger.c1_attrition_run import _value_sha256

    assert _value_sha256(MappingProxyType({"threshold": .5, "cap": 4})) == _value_sha256({"cap": 4, "threshold": .5})


def test_runner_rejects_symlinked_immutable_output(tmp_path: Path):
    """A reparse-point output cannot be used to escape the evidence boundary."""
    from src.pipeline.challenger.c1_attrition_run import run_c1_attrition

    keys = [f"{index % 5}:sid-{index}:0:1" for index in range(11)]
    _fn_csv(tmp_path, keys)
    output = tmp_path / "outputs/challenger_70/e03_5_c1_attrition"
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        output.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable on this Windows test host: {exc}")
    with pytest.raises(RuntimeError, match="immutable"):
        run_c1_attrition(
            tmp_path, output, replay=lambda _root, fold: {"fold": fold, "metrics": _metrics(fold)},
            audit=lambda trace: _audit([key for key in keys if int(key.split(":", 1)[0]) == trace["fold"]]),
            snapshot=lambda _root: {}, assert_unchanged=lambda _before, _root: None,
            git_sha=lambda _root: "a" * 40,
        )


def test_runner_never_trains_or_scores_after_injected_replay(tmp_path: Path):
    """The runner consumes an already-scored trace; fit/predict hooks are forbidden."""
    from src.pipeline.challenger.c1_attrition_run import run_c1_attrition

    keys = [f"{index % 5}:sid-{index}:0:1" for index in range(11)]
    _fn_csv(tmp_path, keys)
    output = tmp_path / "outputs/challenger_70/e03_5_c1_attrition"

    class FrozenTrace:
        def __init__(self, fold: int):
            self.fold, self.metrics, self.policy = fold, _metrics(fold), MappingProxyType({"threshold": .5})

        def fit(self, *_args, **_kwargs):  # pragma: no cover - must remain unreachable
            raise AssertionError("runner attempted fit")

        def predict_proba(self, *_args, **_kwargs):  # pragma: no cover - must remain unreachable
            raise AssertionError("runner attempted predict_proba")

    run_c1_attrition(
        tmp_path, output, replay=lambda _root, fold: FrozenTrace(fold),
        audit=lambda trace: _audit([key for key in keys if int(key.split(":", 1)[0]) == trace.fold]),
        snapshot=lambda _root: {}, assert_unchanged=lambda _before, _root: None,
        git_sha=lambda _root: "a" * 40,
    )
    assert output.is_dir()
