"""Boundary contracts for the preregistered E02 Context-v2 gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_registered_config_is_the_fixed_incumbent_surface():
    from src.pipeline.challenger.context_v2 import registered_e02_config

    config = registered_e02_config(2, workers=5)
    assert config.outer_fold == 2
    assert config.context_features_version == "v2"
    assert config.inner_splits == 4
    assert config.seed == 20260908
    assert config.workers == 5
    assert config.no_tcn is True
    assert config.micro_enabled is True
    assert config.micro_gravity_align is True
    assert config.candidate_control_enabled is True
    assert config.admission_minimum_recall == 0.80
    assert config.verifier_c_grid == (0.1,)
    assert config.external_fd_weight_grid == (0.0,)


def test_cache_key_binds_namespace_config_code_numpy_and_schema(monkeypatch):
    from src.pipeline.challenger.context_v2 import e02_cache_key, registered_e02_config

    config = registered_e02_config(0, workers=5)
    key = e02_cache_key(config, code_sha="a" * 40, numpy_version="2.4.6")
    assert key != e02_cache_key(config, code_sha="b" * 40, numpy_version="2.4.6")
    assert key != e02_cache_key(config, code_sha="a" * 40, numpy_version="2.4.5")
    assert len(key) == 64


def test_preregistration_is_immutable_and_has_fixed_schema(monkeypatch, tmp_path: Path):
    from src.pipeline.challenger import context_v2

    output = tmp_path / "outputs" / "challenger_70" / "experiments" / "E02"
    monkeypatch.setattr(context_v2, "_require_e02_environment", lambda *_args, **kwargs: kwargs.get("code_sha") or "a" * 40)
    plan_path = context_v2.write_e02_preregistration(tmp_path, output, code_sha="a" * 40)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    schema = json.loads((output / "context_v2_schema.json").read_text(encoding="utf-8"))
    assert plan["base_run"] == "160afaf81debf1ee"
    assert plan["code_sha"] == "a" * 40
    assert plan["numpy_version"] == "2.4.6"
    assert plan["selection_protocol"] == "outer-train inner-OOF only"
    assert plan["allowed_hyperparameter_grid"]["admission_minimum_recall"] == 0.80
    assert schema["width"] == 25
    assert len(schema["columns"]) == 25
    assert context_v2.write_e02_preregistration(tmp_path, output, code_sha="a" * 40) == plan_path
    with pytest.raises(RuntimeError, match="immutable"):
        context_v2.write_e02_preregistration(tmp_path, output, code_sha="b" * 40)


def test_derived_retry_has_new_identity_without_mutating_prior_attempt(monkeypatch, tmp_path: Path):
    from src.pipeline.challenger import context_v2

    monkeypatch.setattr(context_v2, "_require_e02_environment", lambda *_args, **kwargs: kwargs.get("code_sha") or "b" * 40)
    first = tmp_path / "outputs" / "challenger_70" / "experiments" / "E02"
    retry = tmp_path / "outputs" / "challenger_70" / "experiments" / "E02-002"
    context_v2.write_e02_preregistration(tmp_path, first, code_sha="a" * 40)
    context_v2.write_e02_preregistration(tmp_path, retry, code_sha="b" * 40, experiment_id="E02-002", derived_from="E02-001")
    assert json.loads((first / "plan.json").read_text(encoding="utf-8"))["code_sha"] == "a" * 40
    plan = json.loads((retry / "plan.json").read_text(encoding="utf-8"))
    assert plan["experiment_id"] == "E02-002"
    assert plan["derived_from"] == "E02-001"


def test_wrong_numpy_rejects_before_output_or_cache_write(monkeypatch, tmp_path: Path):
    from src.pipeline.challenger import context_v2

    monkeypatch.setattr(context_v2.np, "__version__", "2.4.5")
    output = tmp_path / "outputs" / "challenger_70" / "experiments" / "E02"
    with pytest.raises(RuntimeError, match="NumPy 2.4.6"):
        context_v2.write_e02_preregistration(tmp_path, output, code_sha="a" * 40)
    assert not output.exists()
    assert not (tmp_path / "cache").exists()


def test_gate_requires_every_registered_predicate():
    from src.pipeline.challenger.context_v2 import gate_passes

    report = {
        "v1_parity": True,
        "v1": {"n_tp": 20, "n_pred": 30, "f1": 0.70},
        "v2": {"n_tp": 18, "n_pred": 25, "f1": 0.72},
        "folds": [{"v1_f1": .1, "v2_f1": .1}] * 3 + [{"v1_f1": .2, "v2_f1": .1}] * 2,
    }
    assert gate_passes(report) is True
    report["v2"] = {"n_tp": 17, "n_pred": 23, "f1": .73}
    assert gate_passes(report) is False
    report["v2"] = {"n_tp": 18, "n_pred": 30, "f1": .72}
    assert gate_passes(report) is False
    report["v2"] = {"n_tp": 18, "n_pred": 25, "f1": .70}
    assert gate_passes(report) is False


def test_inner_gate_never_reads_validation_or_incumbent_cache(monkeypatch, tmp_path: Path):
    from src.pipeline.challenger import context_v2

    class Source:
        def load_outer_fold(self, _config):
            raise AssertionError("outer validation must not be loaded")

        def load_outer_train_only(self, config):
            return object()

    monkeypatch.setattr(context_v2, "_require_e02_environment", lambda *_args, **_kwargs: "a" * 40)
    monkeypatch.setattr(context_v2, "write_e02_preregistration", lambda *_args, **_kwargs: tmp_path / "plan.json")
    monkeypatch.setattr(context_v2, "protected_snapshot", lambda *_args: {})
    monkeypatch.setattr(context_v2, "assert_snapshot_unchanged", lambda *_args: None)
    monkeypatch.setattr(context_v2, "_frozen_v1_inner", lambda *_args: {"n_tp": 10, "n_pred": 15, "n_true": 20, "f1": .5714})
    seen = []
    def inner(data, config):
        assert data is not None
        seen.append(config.context_features_version)
        return type("Result", (), {"inner_metrics": type("Metrics", (), {"n_tp": 2, "n_pred": 3, "n_true": 4, "f1": .5})(), "verifier_feature_count": 116 if config.context_features_version == "v1" else 141})()
    monkeypatch.setattr(context_v2, "run_outer_train_inner_oof", inner)
    report = context_v2.run_e02_inner_gate(tmp_path, tmp_path / "outputs" / "challenger_70" / "experiments" / "E02", workers=1, source_factory=lambda _: Source())
    assert seen == ["v1", "v2"] * 5
    assert report["phase"] == "inner-gate"
