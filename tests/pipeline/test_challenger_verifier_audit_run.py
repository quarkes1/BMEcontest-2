"""Sealed-output contracts for the E04 verifier audit writer."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest


_FOLD_COUNTS = (3, 1, 10, 6, 11)
_METRICS = ((22, 30, 39),) * 4 + ((26, 33, 41),)


def _e01(root: Path) -> list[str]:
    """Create the approved 27 C1 / 4 C2 raw-reachable E01 identity set."""
    rows: list[dict[str, object]] = []
    keys: list[str] = []
    ordinal = 0
    for fold, count in enumerate(_FOLD_COUNTS):
        for _ in range(count):
            stage = "FN-C2" if ordinal >= 27 else "FN-C1"
            key = f"{fold}:sid-{ordinal}:0:1"
            keys.append(key)
            rows.append({"truth_key": key, "outer_fold": fold, "fn_stage": stage,
                         "union_raw_reachable": "True"})
            ordinal += 1
    path = root / "outputs/challenger_70/error_atlas/fn.csv"
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for relative in (
        "docs/superpowers/specs/2026-09-21-amendment-a2-e04-verifier-failure-audit-design.md",
        "docs/superpowers/specs/2026-09-21-amendment-a1-e03-5-c1-attrition-design.md",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("sealed test evidence\n", encoding="utf-8")
    return keys


def _audit(fold: int, keys: list[str]) -> dict[str, object]:
    witnesses = [{"truth_key": key, "raw_index": 0, "fold": fold, "margin": -.1,
                  "diagnostic_only_uses_gt": True} for key in keys]
    pairs = [dict(row, group="reachable_fn_candidate") for row in witnesses]
    return {"fold": fold, "reachable_fn_candidates": pairs, "tp_reference": (), "fp_reference": (),
            "best_iou_witnesses": [dict(row, witness_definition="best_iou") for row in witnesses],
            "best_score_witnesses": [dict(row, witness_definition="best_score") for row in witnesses],
            "analysis_rows": (), "examples": witnesses[:1]}


def _summary(_audits: object) -> dict[str, object]:
    return {name: {"diagnostic_only": True} for name in (
        "score_margin_analysis", "lr_lgbm_disagreement", "context_contribution",
        "separability_by_witness", "mechanisms",
    )}


def test_runner_installs_exact_atomic_e04_surface_and_preserves_registry(tmp_path: Path):
    """Missing any specified artifact or writing the registry breaks the sealed audit boundary."""
    from src.pipeline.challenger.verifier_audit_run import OUTPUT_FILES, run_verifier_audit

    keys = _e01(tmp_path)
    registry = tmp_path / "outputs/challenger_70/experiment_registry.jsonl"
    registry.write_text('{"historical": true}\n', encoding="utf-8")
    output = tmp_path / "outputs/challenger_70/e04_verifier_audit"
    installed = run_verifier_audit(
        tmp_path, output, replay=lambda _root, fold: {"fold": fold, "metrics": _METRICS[fold], "policy": {}},
        audit=lambda trace: _audit(trace["fold"], [key for key in keys if key.startswith(f"{trace['fold']}:")]),
        summarize=_summary, snapshot=lambda _root: {"protected": ("file", "frozen")},
        assert_unchanged=lambda _before, _root: None, git_sha=lambda _root: "a" * 40,
    )
    assert installed == output
    assert tuple(sorted(path.name for path in output.iterdir())) == tuple(sorted(OUTPUT_FILES))
    assert (output / "candidate_examples").is_dir()
    assert registry.read_text(encoding="utf-8") == '{"historical": true}\n'
    assert "outer_f1" not in (output / "report.md").read_text(encoding="utf-8")


def test_runner_rejects_wrong_e01_identity_and_immutable_or_symlink_output(tmp_path: Path):
    """Changing the 31 raw-reachable E01 targets or replacing evidence must abort before install."""
    from src.pipeline.challenger.verifier_audit_run import run_verifier_audit

    keys = _e01(tmp_path)
    output = tmp_path / "outputs/challenger_70/e04_verifier_audit"
    common = dict(replay=lambda _root, fold: {"fold": fold, "metrics": _METRICS[fold]}, summarize=_summary,
                  snapshot=lambda _root: {}, assert_unchanged=lambda _before, _root: None,
                  git_sha=lambda _root: "a" * 40)
    with pytest.raises(RuntimeError, match="E01"):
        run_verifier_audit(tmp_path, output, audit=lambda trace: _audit(trace["fold"], []), **common)
    output.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="immutable"):
        run_verifier_audit(tmp_path, output, audit=lambda trace: _audit(trace["fold"], []), **common)


def test_runner_cleans_staging_on_snapshot_or_writer_failure(tmp_path: Path, monkeypatch):
    """A protected-input change or writer fault cannot leave installed or staging evidence behind."""
    import src.pipeline.challenger.verifier_audit_run as runner

    keys = _e01(tmp_path)
    output = tmp_path / "outputs/challenger_70/e04_verifier_audit"
    with pytest.raises(RuntimeError, match="protected release"):
        runner.run_verifier_audit(
            tmp_path, output, replay=lambda _root, fold: {"fold": fold, "metrics": _METRICS[fold]},
            audit=lambda trace: _audit(trace["fold"], [key for key in keys if key.startswith(f"{trace['fold']}:")]),
            summarize=_summary, snapshot=lambda _root: {"protected": ("file", "frozen")},
            assert_unchanged=lambda _before, _root: (_ for _ in ()).throw(RuntimeError("protected release input changed")),
            git_sha=lambda _root: "a" * 40,
        )
    assert not output.exists()
    assert not list(output.parent.glob(".e04_verifier_audit.staging-*"))


def test_runner_never_trains_or_scores_after_injected_replay(tmp_path: Path):
    """The runner only transports frozen audit values; fit and scoring hooks remain unreachable."""
    from src.pipeline.challenger.verifier_audit_run import run_verifier_audit

    keys = _e01(tmp_path)
    output = tmp_path / "outputs/challenger_70/e04_verifier_audit"

    class FrozenTrace:
        def __init__(self, fold: int): self.fold, self.metrics, self.policy = fold, _METRICS[fold], {}
        def fit(self, *_args, **_kwargs): raise AssertionError("runner attempted fit")
        def predict_proba(self, *_args, **_kwargs): raise AssertionError("runner attempted scoring")

    run_verifier_audit(
        tmp_path, output, replay=lambda _root, fold: FrozenTrace(fold),
        audit=lambda trace: _audit(trace.fold, [key for key in keys if key.startswith(f"{trace.fold}:")]),
        summarize=_summary, snapshot=lambda _root: {}, assert_unchanged=lambda _before, _root: None,
        git_sha=lambda _root: "a" * 40,
    )
    assert output.is_dir()


def test_runner_rejects_more_than_two_or_unsafe_candidate_examples_per_fold(tmp_path: Path):
    """Unbounded or path-derived examples would violate the data-only sealed artifact contract."""
    from src.pipeline.challenger.verifier_audit_run import run_verifier_audit

    keys = _e01(tmp_path)
    output = tmp_path / "outputs/challenger_70/e04_verifier_audit"

    def overfull(trace):
        result = _audit(trace["fold"], [key for key in keys if key.startswith(f"{trace['fold']}:")])
        if trace["fold"] == 0:
            result["examples"] = [
                {"truth_key": "../unsafe", "raw_index": index, "diagnostic_only_uses_gt": True}
                for index in range(3)
            ]
        return result

    with pytest.raises(RuntimeError, match="at most two"):
        run_verifier_audit(
            tmp_path, output, replay=lambda _root, fold: {"fold": fold, "metrics": _METRICS[fold]},
            audit=overfull, summarize=_summary, snapshot=lambda _root: {},
            assert_unchanged=lambda _before, _root: None, git_sha=lambda _root: "a" * 40,
        )
    assert not output.exists()


def test_report_describes_mechanisms_model_spaces_and_fold_support_without_recommendation():
    """A report that omits descriptive evidence or recommends a challenger would misstate E04's scope."""
    from src.pipeline.challenger.verifier_audit_run import _report

    summary = {
        "mechanisms": {
            "best_iou": {"pooled": {"counts": {"A": 1, "B": 2, "C": 3, "D": 4, "B_model_disagreement": 5, "unsupported": 6, "unresolved": 7}},
                         "per_fold": {"0": {"denominator": 1}}},
            "best_score": {"pooled": {"counts": {"A": 2, "B": 3, "C": 4, "D": 5, "B_model_disagreement": 6, "unsupported": 7, "unresolved": 8}},
                           "per_fold": {"0": {"denominator": 1}}},
        },
        "lr_lgbm_disagreement": {"pooled": {"best_score_fn": {"flagged": 1}}},
        "context_contribution": {"pooled": {"best_score_fn": {"lr_context": {"median": -.2}, "lgbm_context": {"median": -.1}}}},
        "separability_by_witness": {"best_score": {"base_00": {"direction": "stable_descriptive"}}},
    }
    report = _report(tuple(f"0:key-{index}" for index in range(31)), summary)
    assert "27 C1 and 4 C2" in report
    assert "Mechanism counts" in report and "LR/LGBM" in report and "Context" in report
    assert "fold support" in report and "recommend" not in report.lower()
    assert "best-IoU" in report and "best-score" in report
    for count in ("A=1", "B=2", "C=3", "D=4", "B_model_disagreement=5", "unsupported=6", "unresolved=7",
                  "A=2", "B=3", "C=4", "D=5", "B_model_disagreement=6", "unsupported=7", "unresolved=8"):
        assert count in report


def test_candidate_example_filename_is_stable_and_cannot_embed_truth_key_path_segments():
    """Truth identifiers are evidence data, never path syntax inside the installed artifact tree."""
    from src.pipeline.challenger.verifier_audit_run import _example_filename

    name = _example_filename(0, "../sid/with:separators", 12)
    assert name == _example_filename(0, "../sid/with:separators", 12)
    assert name.startswith("fold-0_") and name.endswith(".json")
    assert "/" not in name and "\\" not in name and ".." not in name


def test_runner_rejects_string_gt_flag_and_never_serializes_it_as_a_boolean(tmp_path: Path):
    """Truth-aware provenance must be the canonical bool, not a truthy string supplied by a row."""
    from src.pipeline.challenger.verifier_audit_run import run_verifier_audit

    keys = _e01(tmp_path)
    output = tmp_path / "outputs/challenger_70/e04_verifier_audit"

    def string_flag(trace):
        result = _audit(trace["fold"], [key for key in keys if key.startswith(f"{trace['fold']}:")])
        if trace["fold"] == 0:
            result["examples"][0]["diagnostic_only_uses_gt"] = "False"
        return result

    with pytest.raises(RuntimeError, match="GT-aware"):
        run_verifier_audit(
            tmp_path, output, replay=lambda _root, fold: {"fold": fold, "metrics": _METRICS[fold]},
            audit=string_flag, summarize=_summary, snapshot=lambda _root: {},
            assert_unchanged=lambda _before, _root: None, git_sha=lambda _root: "a" * 40,
        )
    assert not output.exists()


def test_runner_never_performs_a_fallible_snapshot_check_after_atomic_install(tmp_path: Path):
    """Post-install cleanup could delete a concurrent replacement, so all guards must precede install."""
    from src.pipeline.challenger.verifier_audit_run import run_verifier_audit

    keys = _e01(tmp_path)
    output = tmp_path / "outputs/challenger_70/e04_verifier_audit"
    calls = 0

    def unchanged(_before, _root):
        nonlocal calls
        calls += 1
        if calls > 3:
            raise AssertionError("runner performed a snapshot check after installation")

    run_verifier_audit(
        tmp_path, output, replay=lambda _root, fold: {"fold": fold, "metrics": _METRICS[fold]},
        audit=lambda trace: _audit(trace["fold"], [key for key in keys if key.startswith(f"{trace['fold']}:")]),
        summarize=_summary, snapshot=lambda _root: {}, assert_unchanged=unchanged,
        git_sha=lambda _root: "a" * 40,
    )
    assert calls == 3
    assert output.is_dir()
    assert not list(output.parent.glob(".e04_verifier_audit.staging-*"))


def test_no_replace_rename_preserves_a_target_created_immediately_before_install(tmp_path: Path, monkeypatch):
    """A TOCTOU target creation must make the sealed install fail without replacement or deletion."""
    import src.pipeline.challenger.verifier_audit_run as runner

    keys = _e01(tmp_path)
    output = tmp_path / "outputs/challenger_70/e04_verifier_audit"

    original_rename = runner.os.rename

    def racing_rename(stage, target):
        target.mkdir()
        (target / "external.txt").write_text("not ours", encoding="utf-8")
        original_rename(stage, target)

    monkeypatch.setattr(runner.os, "rename", racing_rename)
    with pytest.raises(FileExistsError):
        runner.run_verifier_audit(
            tmp_path, output, replay=lambda _root, fold: {"fold": fold, "metrics": _METRICS[fold]},
            audit=lambda trace: _audit(trace["fold"], [key for key in keys if key.startswith(f"{trace['fold']}:")]),
            summarize=_summary, snapshot=lambda _root: {}, assert_unchanged=lambda _before, _root: None,
            git_sha=lambda _root: "a" * 40,
        )
    assert (output / "external.txt").read_text(encoding="utf-8") == "not ours"
    assert not list(output.parent.glob(".e04_verifier_audit.staging-*"))


def test_runner_fails_closed_without_windows_no_replace_rename(tmp_path: Path, monkeypatch):
    """Portability must not silently downgrade immutable installation to replacement semantics."""
    import src.pipeline.challenger.verifier_audit_run as runner

    keys = _e01(tmp_path)
    output = tmp_path / "outputs/challenger_70/e04_verifier_audit"
    monkeypatch.setattr(
        runner, "_require_windows_no_replace",
        lambda: (_ for _ in ()).throw(RuntimeError("Windows no-replace rename is required")),
        raising=False,
    )
    with pytest.raises(RuntimeError, match="Windows"):
        runner.run_verifier_audit(
            tmp_path, output, replay=lambda _root, fold: {"fold": fold, "metrics": _METRICS[fold]},
            audit=lambda trace: _audit(trace["fold"], [key for key in keys if key.startswith(f"{trace['fold']}:")]),
            summarize=_summary, snapshot=lambda _root: {}, assert_unchanged=lambda _before, _root: None,
            git_sha=lambda _root: "a" * 40,
        )
    assert not output.exists()
