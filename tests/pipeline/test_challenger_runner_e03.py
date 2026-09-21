"""Behavioural guards for the preregistered E03 inner-only gate."""

from __future__ import annotations

import json
import hashlib
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
    plan_path = e03.write_e03_preregistration(root, output, code_sha="a" * 40)
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
    plan = json.loads(e03.write_e03_preregistration(tmp_path, output, code_sha="a" * 40).read_text())
    assert plan["incumbent_outer_anchor"] == {"run_key": "160afaf81debf1ee", "tp": 114, "eligible": 153, "pred": 197, "fp": 83, "f1": 0.6514285714285715}
    assert plan["comparison_arms"] == ["threshold_only", "hard_negative_v1"]


def test_new_e03_experiment_id_requires_lineage_and_uses_its_own_output(monkeypatch, tmp_path: Path):
    """A post-abort retry must be a new immutable experiment, never E03-001 reuse."""
    from src.pipeline.challenger import e03
    output = tmp_path / "outputs" / "challenger_70" / "experiments" / "E03-002"
    monkeypatch.setattr(e03, "_require_e03_environment", lambda *_args, **kwargs: kwargs.get("code_sha") or "b" * 40)
    monkeypatch.setattr(e03, "_frozen_outer_anchor", lambda _root: {"run_key": "160afaf81debf1ee", "tp": 114, "eligible": 153, "pred": 197, "fp": 83, "f1": 0.6514285714285715})
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

    target = e03.record_e03_preflight_abort(tmp_path, output, failure="missing raw session is canonical cache state")
    evidence = json.loads(target.read_text())
    assert evidence["scores_recorded"] is False
    assert evidence["training_started"] is False
    assert evidence["cache_written"] is False
    assert evidence["historical_terminal"] is True
    assert evidence["replay_permitted"] is False
    assert "n_tp" not in evidence
    with pytest.raises(RuntimeError, match="immutable"):
        e03.record_e03_preflight_abort(tmp_path, output, failure="different")


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


def test_e03_experiment_chain_rejects_unregistered_id_or_wrong_e03_003_lineage():
    """Only the explicitly registered retry chain can create a new plan."""
    from src.pipeline.challenger import e03

    with pytest.raises(ValueError, match="registered"):
        e03._plan_payload("a" * 40, experiment_id="E03-004", derived_from="E03-003")
    with pytest.raises(ValueError, match="derived_from"):
        e03._plan_payload("a" * 40, experiment_id="E03-003", derived_from="E03-001")
    assert e03._plan_payload("a" * 40, experiment_id="E03-003", derived_from="E03-002")["derived_from"] == "E03-002"


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
    append_e03_terminal_registry(tmp_path, {"status": "stopped_inner_gate", "passed": False}, code_sha="a" * 40)
    with pytest.raises(RuntimeError, match="already exists"):
        append_e03_terminal_registry(tmp_path, {"status": "complete", "passed": True}, code_sha="a" * 40)


def test_terminal_registry_preserves_prior_rows_byte_for_byte(tmp_path: Path):
    """Atomic append must never reserialize older experiment evidence."""
    from src.pipeline.challenger.e03 import append_e03_terminal_registry
    registry = tmp_path / "outputs" / "challenger_70" / "experiment_registry.jsonl"
    registry.parent.mkdir(parents=True)
    prior = '{"code_sha":"old","experiment_id":"E00","status":"complete"}\n'
    registry.write_text(prior, encoding="utf-8", newline="\n")
    append_e03_terminal_registry(tmp_path, {"status": "stopped_inner_gate", "passed": False}, code_sha="a" * 40)
    assert registry.read_text(encoding="utf-8").startswith(prior)


def test_terminal_registry_refuses_a_concurrent_e03_lock(tmp_path: Path):
    """A stale/concurrent writer must fail closed instead of racing duplicate checks."""
    from src.pipeline.challenger.e03 import append_e03_terminal_registry
    lock = tmp_path / "outputs" / "challenger_70" / ".experiment_registry.E03-001.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("other process", encoding="utf-8")
    with pytest.raises(RuntimeError, match="concurrent"):
        append_e03_terminal_registry(tmp_path, {"status": "stopped_inner_gate", "passed": False}, code_sha="a" * 40)


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
