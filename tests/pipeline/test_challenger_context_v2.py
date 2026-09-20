"""Boundary contracts for the preregistered E02 Context-v2 gate."""

from __future__ import annotations

import json
import os
from pathlib import Path
import py_compile
import subprocess

import pytest


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True,
    ).stdout.strip()


def _commit(root: Path, message: str) -> str:
    _git(root, "add", ".")
    _git(root, "commit", "-m", message)
    return _git(root, "rev-parse", "HEAD")


def _identity_repo(tmp_path: Path) -> tuple[Path, Path, str]:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test")
    (root / "src").mkdir()
    (root / ".gitignore").write_text("src/generated.py\n", encoding="utf-8")
    (root / "src" / "implementation.py").write_text("VALUE = 1\n", encoding="utf-8")
    code_sha = _commit(root, "code")
    output = root / "outputs" / "challenger_70" / "experiments" / "E02-005"
    output.mkdir(parents=True)
    (output / "plan.json").write_text(json.dumps({"code_sha": code_sha}), encoding="utf-8")
    (output / "context_v2_schema.json").write_text("{}", encoding="utf-8")
    return root, output, code_sha


def test_preregistered_code_identity_allows_only_its_immutable_evidence_descendant(tmp_path: Path):
    from src.pipeline.challenger.context_v2 import verify_e02_code_identity

    root, output, code_sha = _identity_repo(tmp_path)
    _commit(root, "data: preregister E02-005")

    assert verify_e02_code_identity(root, output, code_sha) == code_sha


def test_new_plan_binds_current_source_commit_before_its_evidence_descendant(tmp_path: Path):
    from src.pipeline.challenger.context_v2 import verify_e02_code_identity

    root, output, code_sha = _identity_repo(tmp_path)
    assert json.loads((output / "plan.json").read_text(encoding="utf-8"))["code_sha"] == code_sha
    _commit(root, "data: preregister E02-005")

    assert verify_e02_code_identity(root, output, code_sha) == code_sha


@pytest.mark.parametrize("path", [
    "src/implementation.py",
    "outputs/challenger_70/error_atlas/report.md",
])
def test_preregistered_code_identity_rejects_non_evidence_descendant_changes(tmp_path: Path, path: str):
    from src.pipeline.challenger.context_v2 import verify_e02_code_identity

    root, output, code_sha = _identity_repo(tmp_path)
    _commit(root, "data: preregister E02-005")
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("changed\n", encoding="utf-8")
    _commit(root, "unexpected change")

    with pytest.raises(RuntimeError, match="non-evidence"):
        verify_e02_code_identity(root, output, code_sha)


def test_preregistered_code_identity_rejects_nonancestor_and_dirty_code(tmp_path: Path):
    from src.pipeline.challenger.context_v2 import verify_e02_code_identity

    root, output, code_sha = _identity_repo(tmp_path)
    _commit(root, "data: preregister E02-005")
    with pytest.raises(RuntimeError, match="not an ancestor"):
        verify_e02_code_identity(root, output, "a" * 40)

    (root / "src" / "implementation.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="dirty non-evidence"):
        verify_e02_code_identity(root, output, code_sha)
    _git(root, "checkout", "--", "src/implementation.py")
    (root / "new_module.py").write_text("VALUE = 3\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="dirty non-evidence"):
        verify_e02_code_identity(root, output, code_sha)


def test_preregistered_code_identity_rejects_ignored_source_file(tmp_path: Path):
    from src.pipeline.challenger.context_v2 import verify_e02_code_identity

    root, output, code_sha = _identity_repo(tmp_path)
    _commit(root, "data: preregister E02-005")
    (root / "src" / "generated.py").write_text("VALUE = 3\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="dirty non-evidence"):
        verify_e02_code_identity(root, output, code_sha)


def test_preregistered_code_identity_allows_only_normal_runtime_detritus(tmp_path: Path):
    from src.pipeline.challenger.context_v2 import verify_e02_code_identity

    root, output, code_sha = _identity_repo(tmp_path)
    _commit(root, "data: preregister E02-005")
    py_compile.compile(str(root / "src" / "implementation.py"), doraise=True)
    pytest_cache = root / "tests" / ".pytest_cache" / "v" / "cache"
    pytest_cache.mkdir(parents=True)
    (pytest_cache / "nodeids").write_text("[]\n", encoding="utf-8")

    assert verify_e02_code_identity(root, output, code_sha) == code_sha


def test_preregistered_code_identity_rejects_source_rename_into_evidence(tmp_path: Path):
    from src.pipeline.challenger.context_v2 import verify_e02_code_identity

    root, output, code_sha = _identity_repo(tmp_path)
    _commit(root, "data: preregister E02-005")
    _git(root, "mv", "src/implementation.py", "outputs/challenger_70/experiments/E02-005/inner_gate.json")
    _commit(root, "move source into evidence")

    with pytest.raises(RuntimeError, match="non-evidence"):
        verify_e02_code_identity(root, output, code_sha)


def test_preregistered_code_identity_rejects_symlink_and_unapproved_evidence_path(tmp_path: Path):
    from src.pipeline.challenger.context_v2 import verify_e02_code_identity

    root, output, code_sha = _identity_repo(tmp_path)
    _commit(root, "data: preregister E02-005")
    (output / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="dirty non-evidence"):
        verify_e02_code_identity(root, output, code_sha)
    (output / "unexpected.json").unlink()

    target = output / "plan.json"
    replacement = output / "replacement.json"
    replacement.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
    target.unlink()
    try:
        os.symlink(replacement.name, target)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(RuntimeError, match="symlink"):
        verify_e02_code_identity(root, output, code_sha)


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
    monkeypatch.setattr(context_v2, "verify_e02_code_identity", lambda _root, _output, code_sha: code_sha)
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


def test_reopened_preregistration_reuses_verified_plan_code_sha(monkeypatch, tmp_path: Path):
    from src.pipeline.challenger import context_v2

    output = tmp_path / "outputs" / "challenger_70" / "experiments" / "E02"
    monkeypatch.setattr(context_v2, "_require_e02_environment", lambda *_args, **kwargs: kwargs.get("code_sha") or "b" * 40)
    context_v2.write_e02_preregistration(tmp_path, output, code_sha="a" * 40)
    seen: list[str] = []
    monkeypatch.setattr(
        context_v2, "verify_e02_code_identity",
        lambda _root, _output, code_sha: seen.append(code_sha) or code_sha,
    )

    assert context_v2.write_e02_preregistration(tmp_path, output) == output / "plan.json"
    assert seen == ["a" * 40]


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
