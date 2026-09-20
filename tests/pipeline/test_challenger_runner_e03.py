"""Behavioural guards for the preregistered E03 inner-only gate."""

from __future__ import annotations

import json
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
    plan_path = e03.write_e03_preregistration(root, output, code_sha="a" * 40)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    assert plan["experiment_id"] == "E03-001"
    assert plan["derived_from"] == "E02-005"
    assert plan["feature_version"] == "v1"
    assert plan["allowed_hyperparameter_grid"]["hard_negative_weight"] == [2.0]
    assert len(plan["mining_spec_sha256"]) == 64

