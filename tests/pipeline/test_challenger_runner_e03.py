"""Behavioural guards for the preregistered E03 inner-only gate."""

from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from src.pipeline.event_stack import EventMetrics


def _metric(tp: int, fp: int, *, true: int = 120) -> EventMetrics:
    pred = tp + fp
    return EventMetrics(tp, pred, true, tp / pred if pred else 0.0,
                        tp / true if true else 0.0,
                        2 * tp / (true + pred) if true + pred else 0.0)


def test_gate_requires_all_four_preregistered_predicates():
    """Dropping FP, TP, fold-stability, or strict-F1 condition must block E03."""
    from src.pipeline.challenger.e03 import e03_gate_passes

    control = [_metric(20, 4)] * 5
    assert e03_gate_passes(control, [_metric(17, 0)] * 5) is False  # loses >2 TP
    assert e03_gate_passes(control, [_metric(19, 1)] * 5) is False  # FP drop <4 pooled
    assert e03_gate_passes(control, [_metric(20, 0)] * 2 + [_metric(18, 0)] * 3) is False
    assert e03_gate_passes(control, [_metric(20, 0)] * 5) is True


def test_inner_weight_builder_rejects_non_outer_train_payload():
    """Accepting an outer FoldDataset would expose E03 mining to validation data."""
    from src.pipeline.challenger.e03 import build_e03_inner_weights, registered_e03_config

    with pytest.raises(TypeError, match="OuterTrainOnlyData"):
        build_e03_inner_weights(object(), registered_e03_config(0, workers=1))


def test_plan_records_v1_constants_and_e02_derivation(monkeypatch, tmp_path: Path):
    """A plan missing the immutable mining surface cannot support a valid gate."""
    from src.pipeline.challenger import e03

    root = tmp_path
    output = root / "outputs" / "challenger_70" / "experiments" / "E03-001"
    monkeypatch.setattr(e03, "_require_e03_environment", lambda *_args, **kwargs: kwargs.get("code_sha") or "a" * 40)
    monkeypatch.setattr(e03, "_frozen_outer_anchor", lambda _root: {"run_key": "160afaf81debf1ee", "tp": 114, "eligible": 153, "pred": 197, "fp": 83, "f1": 0.6514285714285715})
    plan_path = e03.write_e03_preregistration(root, output, code_sha="a" * 40, experiment_id="E03-001")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    assert plan["experiment_id"] == "E03-001"
    assert plan["derived_from"] == "E02-005"
    assert plan["feature_version"] == "v1"
    assert plan["allowed_hyperparameter_grid"]["hard_negative_weight"] == [2.0]
    assert len(plan["mining_spec_sha256"]) == 64


def test_plan_contains_verified_read_only_incumbent_anchor(monkeypatch, tmp_path: Path):
    """Confusing the release outer anchor with a reselected inner arm misstates E03 evidence."""
    from src.pipeline.challenger import e03
    output = tmp_path / "outputs" / "challenger_70" / "experiments" / "E03-001"
    monkeypatch.setattr(e03, "_require_e03_environment", lambda *_args, **kwargs: kwargs.get("code_sha") or "a" * 40)
    monkeypatch.setattr(e03, "_frozen_outer_anchor", lambda _root: {"run_key": "160afaf81debf1ee", "tp": 114, "eligible": 153, "pred": 197, "fp": 83, "f1": 0.6514285714285715})
    plan = json.loads(e03.write_e03_preregistration(tmp_path, output, code_sha="a" * 40, experiment_id="E03-001").read_text())
    assert plan["incumbent_outer_anchor"] == {"run_key": "160afaf81debf1ee", "tp": 114, "eligible": 153, "pred": 197, "fp": 83, "f1": 0.6514285714285715}
    assert plan["comparison_arms"] == ["threshold_only", "hard_negative_v1"]


def test_new_e03_experiment_id_requires_lineage_and_uses_its_own_output(monkeypatch, tmp_path: Path):
    """A post-abort retry must be a new immutable experiment, never E03-001 reuse."""
    from src.pipeline.challenger import e03
    output = tmp_path / "outputs" / "challenger_70" / "experiments" / "E03-002"
    monkeypatch.setattr(e03, "_require_e03_environment", lambda *_args, **kwargs: kwargs.get("code_sha") or "b" * 40)
    monkeypatch.setattr(e03, "_frozen_outer_anchor", lambda _root: {"run_key": "160afaf81debf1ee", "tp": 114, "eligible": 153, "pred": 197, "fp": 83, "f1": 0.6514285714285715})
    monkeypatch.setattr(e03, "_require_terminal_parent_abort", lambda *_args, **_kwargs: None)
    with pytest.raises(ValueError, match="derived_from"):
        e03.write_e03_preregistration(tmp_path, output, code_sha="b" * 40, experiment_id="E03-002")
    plan = json.loads(e03.write_e03_preregistration(tmp_path, output, code_sha="b" * 40, experiment_id="E03-002", derived_from="E03-001").read_text())
    assert plan["experiment_id"] == "E03-002"
    assert plan["derived_from"] == "E03-001"


def test_preflight_abort_freezes_no_score_terminal_evidence(monkeypatch, tmp_path: Path):
    """A preregistered failure records no metric or training claim and cannot be overwritten."""
    from src.pipeline.challenger import e03
    output = tmp_path / "outputs" / "challenger_70" / "experiments" / "E03-001"
    output.mkdir(parents=True)
    (output / "plan.json").write_text(json.dumps({"experiment_id": "E03-001", "code_sha": "a" * 40}), encoding="utf-8")
    monkeypatch.setattr(e03, "_require_e03_environment", lambda *_args, **_kwargs: "b" * 40)
    monkeypatch.setattr(e03, "_git_sha", lambda _root: "b" * 40)

    target = e03.record_e03_preflight_abort(tmp_path, output, failure="missing raw session is canonical cache state", experiment_id="E03-001")
    evidence = json.loads(target.read_text())
    assert evidence["scores_recorded"] is False
    assert evidence["training_started"] is False
    assert evidence["cache_written"] is False
    assert evidence["historical_terminal"] is True
    assert evidence["replay_permitted"] is False
    assert "n_tp" not in evidence
    with pytest.raises(RuntimeError, match="immutable"):
        e03.record_e03_preflight_abort(tmp_path, output, failure="different", experiment_id="E03-001")


def test_e03_plan_validation_rejects_copied_id_or_missing_retry_lineage():
    """An E03-001 plan copied to E03-002 must fail before identity or data access."""
    from src.pipeline.challenger import e03

    plan = e03._plan_payload("a" * 40, experiment_id="E03-001", derived_from="E02-005")
    with pytest.raises(RuntimeError, match="experiment_id"):
        e03.validate_e03_plan(plan, experiment_id="E03-002")
    retry = e03._plan_payload("a" * 40, experiment_id="E03-002", derived_from="E03-001")
    retry["derived_from"] = "E02-005"
    with pytest.raises(RuntimeError, match="lineage"):
        e03.validate_e03_plan(retry, experiment_id="E03-002")


def test_e03_plan_validation_rejects_protocol_drift_before_execution():
    """Changing v1 width/mining/config contract is not a valid E03 retry."""
    from src.pipeline.challenger import e03

    plan = e03._plan_payload("a" * 40, experiment_id="E03-002", derived_from="E03-001")
    plan["feature_version"] = "v2"
    with pytest.raises(RuntimeError, match="protocol"):
        e03.validate_e03_plan(plan, experiment_id="E03-002")


def test_e03_plan_validation_accepts_real_json_roundtrip_for_retry():
    """Plans are persisted as JSON, so tuple/list representation must not cause a false abort."""
    from src.pipeline.challenger import e03

    persisted = json.loads(json.dumps(e03._plan_payload("a" * 40, experiment_id="E03-002", derived_from="E03-001")))
    assert e03.validate_e03_plan(persisted, experiment_id="E03-002") == "a" * 40


def test_e03_experiment_chain_registers_e03_005_only_as_e03_004_child():
    """The semantic repair gets one new immutable child, never a reused E03-004."""
    from src.pipeline.challenger import e03

    with pytest.raises(ValueError, match="registered"):
        e03._plan_payload("a" * 40, experiment_id="E03-006", derived_from="E03-005")
    with pytest.raises(ValueError, match="derived_from"):
        e03._plan_payload("a" * 40, experiment_id="E03-003", derived_from="E03-001")
    assert e03._plan_payload("a" * 40, experiment_id="E03-003", derived_from="E03-002")["derived_from"] == "E03-002"
    assert e03._plan_payload("a" * 40, experiment_id="E03-005", derived_from="E03-004")["derived_from"] == "E03-004"


def test_e03_retry_rationale_names_its_direct_aborted_parent():
    """A retry must not claim lineage from an older failure in the chain."""
    from src.pipeline.challenger import e03

    second = e03._plan_payload("a" * 40, experiment_id="E03-002", derived_from="E03-001")
    third = e03._plan_payload("a" * 40, experiment_id="E03-003", derived_from="E03-002")
    assert "E03-001" in second["derivation_rationale"]
    assert "E03-002" in third["derivation_rationale"]
    assert "E03-001" not in third["derivation_rationale"]


def test_e03_003_preregistration_requires_matching_parent_terminal_abort(monkeypatch, tmp_path: Path):
    """E03-003 cannot be registered unless E03-002 ended preflight with no work done."""
    from src.pipeline.challenger import e03

    output = tmp_path / "outputs" / "challenger_70" / "experiments" / "E03-003"
    monkeypatch.setattr(e03, "_require_e03_environment", lambda *_args, **kwargs: kwargs.get("code_sha") or "c" * 40)
    monkeypatch.setattr(e03, "_frozen_outer_anchor", lambda _root: {"run_key": "160afaf81debf1ee", "tp": 114, "eligible": 153, "pred": 197, "fp": 83, "f1": 0.6514285714285715})
    with pytest.raises(RuntimeError, match="parent terminal abort"):
        e03.write_e03_preregistration(tmp_path, output, code_sha="c" * 40, experiment_id="E03-003", derived_from="E03-002")


@pytest.mark.parametrize(
    ("artifact_update", "registry_update"),
    [
        ({"plan_code_sha": "b" * 40}, {}),
        ({}, {"abort_code_sha": "d" * 40}),
        ({"failure": "artifact reason"}, {"failure": "registry reason"}),
        ({"training_started": True}, {"training_started": True}),
    ],
)
def test_e03_003_rejects_parent_artifact_registry_cross_record_mismatch(
    tmp_path: Path, artifact_update: dict[str, object], registry_update: dict[str, object],
):
    """A stitched parent record cannot authorize a retry when its two terminal sources disagree."""
    from src.pipeline.challenger.e03 import _require_terminal_parent_abort

    terminal = {
        "experiment_id": "E03-002", "status": "aborted_preflight", "scores_recorded": False,
        "training_started": False, "cache_written": False, "historical_terminal": True,
        "replay_permitted": False, "plan_code_sha": "a" * 40, "abort_code_sha": "c" * 40,
        "failure": "canonical preflight failure",
    }
    artifact = dict(terminal)
    artifact.update(artifact_update)
    registry = {"code_sha": terminal["plan_code_sha"], **terminal}
    registry.update(registry_update)
    evidence_path = tmp_path / "outputs" / "challenger_70" / "experiments" / "E03-002" / "preflight_abort.json"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text(json.dumps(artifact), encoding="utf-8")
    registry_path = tmp_path / "outputs" / "challenger_70" / "experiment_registry.jsonl"
    registry_path.write_text(json.dumps(registry) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="parent terminal abort"):
        _require_terminal_parent_abort(tmp_path, experiment_id="E03-003")


def test_e03_003_gate_checks_parent_terminal_abort_before_code_identity(monkeypatch, tmp_path: Path):
    """A hand-authored E03-003 plan must not bypass its aborted-parent provenance guard."""
    from src.pipeline.challenger import e03

    output = tmp_path / "outputs" / "challenger_70" / "experiments" / "E03-003"
    output.mkdir(parents=True)
    output.joinpath("plan.json").write_text(
        json.dumps(e03._plan_payload("c" * 40, experiment_id="E03-003", derived_from="E03-002")),
        encoding="utf-8",
    )
    monkeypatch.setattr(e03, "_require_e03_environment", lambda *_args, **_kwargs: "c" * 40)
    monkeypatch.setattr(e03, "verify_e03_code_identity", lambda *_args, **_kwargs: pytest.fail("identity must follow parent guard"))

    with pytest.raises(RuntimeError, match="parent terminal abort"):
        e03.run_e03_inner_gate(tmp_path, output, workers=1, experiment_id="E03-003")


def test_cache_identity_binds_namespace_config_code_and_input_fingerprints():
    """Omitting any source input signature could reuse a cache after train data changed."""
    from src.pipeline.challenger.e03 import e03_cache_key, registered_e03_config
    config = registered_e03_config(0, workers=1)
    first = e03_cache_key(config, code_sha="a" * 40, input_fingerprints=[{"path": "a", "missing": True}])
    changed = e03_cache_key(config, code_sha="a" * 40, input_fingerprints=[{"path": "b", "missing": True}])
    assert first != changed


def test_terminal_registry_is_atomic_append_only_and_rejects_duplicate_id(tmp_path: Path):
    """A second terminal status would overwrite the registered E03 result history."""
    from src.pipeline.challenger.e03 import append_e03_terminal_registry
    append_e03_terminal_registry(tmp_path, {"status": "stopped_inner_gate", "passed": False}, code_sha="a" * 40, experiment_id="E03-001")
    with pytest.raises(RuntimeError, match="already exists"):
        append_e03_terminal_registry(tmp_path, {"status": "complete", "passed": True}, code_sha="a" * 40, experiment_id="E03-001")


def test_terminal_registry_preserves_prior_rows_byte_for_byte(tmp_path: Path):
    """Atomic append must never reserialize older experiment evidence."""
    from src.pipeline.challenger.e03 import append_e03_terminal_registry
    registry = tmp_path / "outputs" / "challenger_70" / "experiment_registry.jsonl"
    registry.parent.mkdir(parents=True)
    prior = '{"code_sha":"old","experiment_id":"E00","status":"complete"}\n'
    registry.write_text(prior, encoding="utf-8", newline="\n")
    append_e03_terminal_registry(tmp_path, {"status": "stopped_inner_gate", "passed": False}, code_sha="a" * 40, experiment_id="E03-001")
    assert registry.read_text(encoding="utf-8").startswith(prior)


def test_terminal_registry_refuses_a_concurrent_e03_lock(tmp_path: Path):
    """A stale/concurrent writer must fail closed instead of racing duplicate checks."""
    from src.pipeline.challenger.e03 import append_e03_terminal_registry
    lock = tmp_path / "outputs" / "challenger_70" / ".experiment_registry.E03-001.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("other process", encoding="utf-8")
    with pytest.raises(RuntimeError, match="concurrent"):
        append_e03_terminal_registry(tmp_path, {"status": "stopped_inner_gate", "passed": False}, code_sha="a" * 40, experiment_id="E03-001")


def test_train_only_input_fingerprints_exclude_validation_artifacts(tmp_path: Path):
    """The E03 cache must bind every train-only source without opening val files."""
    from src.pipeline.challenger.e03 import e03_train_only_input_fingerprints, registered_e03_config

    source = tmp_path / "cache"
    for relative in (
        "slide/fold0_train.npz", "slide/fold0_meal_train.npz", "slide/fold0_no_meal_train.npz",
        "micro15/fold0_train.npz", "micro15/fold0_meal_train.npz", "micro15/fold0_no_meal_train.npz",
        "splits/fold0.json",
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    index = tmp_path / "data" / "sensor_index.csv"
    meals = tmp_path / "data" / "meals.csv"
    index.parent.mkdir(parents=True)
    index.write_text("index", encoding="utf-8")
    meals.write_text("meals", encoding="utf-8")
    # The presence of a validation file must not make it into E03's snapshot.
    (source / "slide" / "fold0_val.npz").write_bytes(b"outer validation")
    (source / "splits" / "fold0.json").write_text(
        json.dumps({"train_sessions": ["train-a"], "val_sessions": ["val-z"]}), encoding="utf-8",
    )
    sessions = source / "sessions"
    sessions.mkdir()
    (sessions / "train-a.npz").write_bytes(b"train raw")
    (sessions / "val-z.npz").write_bytes(b"outer raw")

    rows = e03_train_only_input_fingerprints(
        tmp_path, registered_e03_config(0, workers=1), index_path=index, meals_path=meals,
    )
    paths = {Path(str(row["path"])).name for row in rows}
    assert "fold0_val.npz" not in paths
    assert "val-z.npz" not in paths
    assert {"fold0_train.npz", "fold0_meal_train.npz", "fold0_no_meal_train.npz", "fold0.json", "sensor_index.csv", "meals.csv"} <= paths
    train = source / "slide" / "fold0_train.npz"
    assert next(row for row in rows if row["path"] == str(train.resolve()))["sha256"] == hashlib.sha256(train.read_bytes()).hexdigest()
    raw = sessions / "train-a.npz"
    before = e03_train_only_input_fingerprints(tmp_path, registered_e03_config(0, workers=1), index_path=index, meals_path=meals)
    raw.write_bytes(b"changed train raw")
    after = e03_train_only_input_fingerprints(tmp_path, registered_e03_config(0, workers=1), index_path=index, meals_path=meals)
    assert before != after


def test_outer_subject_guard_uses_only_subject_ids_and_rejects_overlap(tmp_path: Path):
    """Outer IDs guard training provenance, but validation windows/truths stay inaccessible."""
    from src.pipeline.challenger.e03 import assert_e03_outer_subject_disjoint

    assert_e03_outer_subject_disjoint({"train-1", "train-2"}, {"outer-1"})
    with pytest.raises(RuntimeError, match="outer-validation subject overlap"):
        assert_e03_outer_subject_disjoint({"train-1", "outer-1"}, {"outer-1"})


def test_missing_train_raw_session_uses_canonical_missing_fingerprint(tmp_path: Path):
    """Missing raw caches are legal in micro metadata and must not block E03 preflight."""
    from src.pipeline.challenger.e03 import e03_train_only_input_fingerprints, registered_e03_config

    cache = tmp_path / "cache"
    for folder, names in {"slide": ("train", "meal_train", "no_meal_train"), "micro15": ("train", "meal_train", "no_meal_train")}.items():
        for name in names:
            path = cache / folder / f"fold0_{name}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"input")
    manifest = cache / "splits" / "fold0.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"train_sessions": ["present", "missing"], "val_sessions": []}), encoding="utf-8")
    sessions = cache / "sessions"
    sessions.mkdir()
    (sessions / "present.npz").write_bytes(b"present raw")
    index = tmp_path / "index.csv"; index.write_text("index", encoding="utf-8")
    meals = tmp_path / "meals.csv"; meals.write_text("meals", encoding="utf-8")

    rows = e03_train_only_input_fingerprints(tmp_path, registered_e03_config(0, workers=1), index_path=index, meals_path=meals)
    by_path = {Path(str(row["path"])).name: row for row in rows}
    assert by_path["missing.npz"] == {"role": "train_session_raw", "path": str((sessions / "missing.npz").resolve()), "missing": True}
    assert {"size", "mtime_ns", "sha256"} <= set(by_path["present.npz"])


def test_outer_validation_sentinel_rejects_loader_or_outer_subject_access():
    """E03's train-only boundary has no route to a held-out loader or subjects."""
    from src.pipeline.challenger.e03 import assert_e03_outer_validation_inaccessible

    class SafeSource:
        def load_outer_train_only(self, _config):
            return object()

    guarded = assert_e03_outer_validation_inaccessible(SafeSource())
    with pytest.raises(RuntimeError, match="outer-validation"):
        guarded.load_outer_fold(object())


def test_e03_003_identity_uses_the_requested_experiment_output_not_e03_001(tmp_path: Path):
    """Regression: the third retry once silently defaulted its identity path to E03-001."""
    from src.pipeline.challenger import e03

    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True).stdout.strip()

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    git("config", "user.email", "e03-test@example.invalid")
    git("config", "user.name", "E03 test")
    output = tmp_path / "outputs" / "challenger_70" / "experiments" / "E03-003"
    output.mkdir(parents=True)
    output.joinpath("plan.json").write_text("{}\n", encoding="utf-8")
    git("add", "outputs/challenger_70/experiments/E03-003/plan.json")
    git("commit", "-qm", "e03 plan")
    code_sha = git("rev-parse", "HEAD")

    assert e03.verify_e03_code_identity(tmp_path, output, code_sha, experiment_id="E03-003") == code_sha


def test_e03_004_is_the_only_next_retry_and_requires_e03_003_terminal_parent(tmp_path: Path):
    """The fourth attempt has no alternate lineage and cannot skip the third terminal record."""
    from src.pipeline.challenger import e03

    assert e03._plan_payload("a" * 40, experiment_id="E03-004", derived_from="E03-003")["derived_from"] == "E03-003"
    with pytest.raises(ValueError, match="derived_from"):
        e03._plan_payload("a" * 40, experiment_id="E03-004", derived_from="E03-002")
    with pytest.raises(RuntimeError, match="parent terminal abort"):
        e03._require_terminal_parent_abort(tmp_path, experiment_id="E03-004")

    terminal = {
        "experiment_id": "E03-003", "status": "aborted_preflight", "scores_recorded": False,
        "training_started": False, "cache_written": False, "historical_terminal": True,
        "replay_permitted": False, "plan_code_sha": "a" * 40, "abort_code_sha": "b" * 40,
        "failure": "identity output defaulted to E03-001 before data access",
    }
    artifact = tmp_path / "outputs" / "challenger_70" / "experiments" / "E03-003" / "preflight_abort.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps(terminal), encoding="utf-8")
    registry = tmp_path / "outputs" / "challenger_70" / "experiment_registry.jsonl"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(json.dumps({"code_sha": "a" * 40, **terminal}) + "\n", encoding="utf-8")
    e03._require_terminal_parent_abort(tmp_path, experiment_id="E03-004")


def test_e03_005_requires_matching_e03_004_inner_implementation_failure(tmp_path: Path):
    """The real nested E03-004 diagnostic must match its terminal registry row."""
    from src.pipeline.challenger import e03

    diagnostic = {
        "experiment_id": "E03-004", "status": "failed_inner_implementation",
        "code_sha": "a" * 40,
        "scores_recorded": False, "training_started": True,
        "failure": "hard_negative._validate_features rejected canonical Context-v1 NaNs after fold-0 logistic inner-OOF scores completed in memory.",
        "cache_inspection": {
            "cache_written": False,
            "cache_root_exists_at_failure_freeze": False,
        },
        "partial_execution": {
            "fold_reached": 0,
            "logistic_inner_oof_completed_in_memory": True,
            "hard_negative_mining_completed": False,
            "comparison_arms_completed": False,
            "terminal_metrics_recorded": False,
        },
    }
    terminal = {
        "experiment_id": "E03-004", "status": "failed_inner_implementation",
        "code_sha": "a" * 40, "scores_recorded": False, "training_started": True,
        "cache_written": False, "partial_inner_oof_logistic_completed": True,
        "terminal_metrics_recorded": False, "replay_permitted": False,
        "failure": "canonical Context-v1 NaNs were rejected by E03 miner after fold-0 logistic inner-OOF completed in memory; diagnostic frozen in E03-004/failure_diagnostic.json",
    }
    evidence = tmp_path / "outputs" / "challenger_70" / "experiments" / "E03-004" / "failure_diagnostic.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text(json.dumps(diagnostic), encoding="utf-8")
    registry = tmp_path / "outputs" / "challenger_70" / "experiment_registry.jsonl"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(json.dumps({"code_sha": "a" * 40, **terminal}) + "\n", encoding="utf-8")

    e03._require_terminal_parent_abort(tmp_path, experiment_id="E03-005")
    terminal["code_sha"] = "b" * 40
    registry.write_text(json.dumps(terminal) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="records disagree"):
        e03._require_terminal_parent_abort(tmp_path, experiment_id="E03-005")

    terminal["code_sha"] = "a" * 40
    registry.write_text(json.dumps(terminal) + "\n", encoding="utf-8")
    diagnostic["partial_execution"]["terminal_metrics_recorded"] = True
    evidence.write_text(json.dumps(diagnostic), encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact is invalid"):
        e03._require_terminal_parent_abort(tmp_path, experiment_id="E03-005")

    diagnostic["partial_execution"]["terminal_metrics_recorded"] = False
    evidence.write_text(json.dumps(diagnostic), encoding="utf-8")
    terminal["failure"] = "unrelated failure"
    registry.write_text(json.dumps(terminal) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="records disagree"):
        e03._require_terminal_parent_abort(tmp_path, experiment_id="E03-005")


def test_e03_004_preflight_roundtrips_real_plan_without_loading_or_fitting(monkeypatch, tmp_path: Path):
    """Preflight validates the committed E03-004 chain but creates no result/cache or fit side effect."""
    from src.pipeline.challenger import e03

    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True).stdout.strip()

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    git("config", "user.email", "e03-test@example.invalid")
    git("config", "user.name", "E03 test")
    release = tmp_path / "release" / "event_stack_incumbent.json"
    release.parent.mkdir(parents=True)
    release.write_text(json.dumps({"run_key": "160afaf81debf1ee", "f1": 0.6514285714285715}), encoding="utf-8")
    summary = tmp_path / "models" / "event_stack" / "160afaf81debf1ee" / "promotion_summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({"outer_metrics": {"n_tp": 114, "n_true": 153, "n_pred": 197, "f1": 0.6514285714285715}}), encoding="utf-8")
    for name in ("inference", "submission", "visual", "event_stack"):
        (tmp_path / "dist" / name).mkdir(parents=True)
    index = tmp_path / "data" / "sensor_index.csv"
    index.parent.mkdir()
    index.write_text("session_id,externalid\ntrain-a,subject-train\nval-a,subject-val\n", encoding="utf-8")
    meals = tmp_path / "data" / "meals.csv"; meals.write_text("meal\n", encoding="utf-8")
    for fold in range(5):
        for folder in ("slide", "micro15"):
            for suffix in ("train", "meal_train", "no_meal_train"):
                path = tmp_path / "cache" / folder / f"fold{fold}_{suffix}.npz"
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"train-only")
        split = tmp_path / "cache" / "splits" / f"fold{fold}.json"
        split.parent.mkdir(parents=True, exist_ok=True)
        split.write_text(json.dumps({"train_sessions": ["train-a"], "val_sessions": ["val-a"]}), encoding="utf-8")
    raw = tmp_path / "cache" / "sessions" / "train-a.npz"; raw.parent.mkdir(parents=True); raw.write_bytes(b"raw")
    parent_terminal = {
        "experiment_id": "E03-003", "status": "aborted_preflight", "scores_recorded": False,
        "training_started": False, "cache_written": False, "historical_terminal": True,
        "replay_permitted": False, "plan_code_sha": "a" * 40, "abort_code_sha": "b" * 40,
        "failure": "historical preflight failure",
    }
    parent_abort = tmp_path / "outputs" / "challenger_70" / "experiments" / "E03-003" / "preflight_abort.json"
    parent_abort.parent.mkdir(parents=True); parent_abort.write_text(json.dumps(parent_terminal), encoding="utf-8")
    registry = tmp_path / "outputs" / "challenger_70" / "experiment_registry.jsonl"
    registry.parent.mkdir(parents=True, exist_ok=True); registry.write_text(json.dumps({"code_sha": "a" * 40, **parent_terminal}) + "\n", encoding="utf-8")
    git("add", "."); git("commit", "-qm", "frozen E03 preflight inputs")
    code_sha = git("rev-parse", "HEAD")
    output = tmp_path / "outputs" / "challenger_70" / "experiments" / "E03-004"
    output.mkdir()
    output.joinpath("plan.json").write_text(json.dumps(e03._plan_payload(code_sha, root=tmp_path, experiment_id="E03-004", derived_from="E03-003")), encoding="utf-8")
    git("add", "outputs/challenger_70/experiments/E03-004/plan.json"); git("commit", "-qm", "preregister E03-004")

    class NoTrainingSource:
        calls = 0
        def load_outer_train_only(self, _config):
            self.calls += 1
            raise AssertionError("preflight must not load training data")

    source = NoTrainingSource()
    fit_calls: list[str] = []
    monkeypatch.setattr(e03, "crossfit_predict_proba", lambda *_args, **_kwargs: fit_calls.append("crossfit"))
    monkeypatch.setattr(e03._runner, "_window_estimator", lambda *_args, **_kwargs: fit_calls.append("window-estimator"))
    report = e03.run_e03_preflight(tmp_path, output, workers=1, experiment_id="E03-004", source_factory=lambda _root: source, index_path=index, meals_path=meals)

    assert report["phase"] == "preflight"
    assert report["outer_validation_accessible"] is False
    assert report["training_started"] is False
    assert report["scoring_started"] is False
    assert source.calls == 0
    assert fit_calls == []
    assert not (output / "inner_gate.json").exists()
    assert not (tmp_path / "cache" / "challenger_70" / "e03").exists()
    assert len(report["folds"]) == 5


def test_e03_cli_is_runnable_directly_from_the_repository_root():
    """The documented direct-script CLI must reach argparse before importing E03 code."""
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "scripts/run_challenger_e03.py", "--help"],
        cwd=root, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "preflight" in result.stdout
