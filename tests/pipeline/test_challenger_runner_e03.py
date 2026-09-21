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


def test_outer_validation_sentinel_rejects_loader_or_outer_subject_access():
    """E03's train-only boundary has no route to a held-out loader or subjects."""
    from src.pipeline.challenger.e03 import assert_e03_outer_validation_inaccessible

    class SafeSource:
        def load_outer_train_only(self, _config):
            return object()

    guarded = assert_e03_outer_validation_inaccessible(SafeSource())
    with pytest.raises(RuntimeError, match="outer-validation"):
        guarded.load_outer_fold(object())
