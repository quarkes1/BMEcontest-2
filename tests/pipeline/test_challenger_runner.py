"""Fail-closed orchestration contracts for the E00/E01 runner."""

from pathlib import Path

import pytest


def test_nonincumbent_registry_is_rejected_before_output_creation(tmp_path: Path):
    from scripts.run_challenger_e00_e01 import validate_registry

    release = tmp_path / "release"
    release.mkdir()
    (release / "event_stack_incumbent.json").write_text(
        '{"run_key":"not-the-incumbent","f1":0.1}', encoding="utf-8"
    )
    output = tmp_path / "outputs" / "challenger_70"
    with pytest.raises(RuntimeError, match="incumbent"):
        validate_registry(tmp_path, "160afaf81debf1ee")
    assert not output.exists()


def test_runner_does_not_call_training_entrypoint(monkeypatch):
    import src.pipeline.runner as pipeline_runner

    def forbidden(*args, **kwargs):
        raise AssertionError("run_outer_fold must not be called by challenger replay")

    monkeypatch.setattr(pipeline_runner, "run_outer_fold", forbidden)
    from src.pipeline.challenger import replay

    assert not hasattr(replay, "run_outer_fold")


def test_atomic_writer_replaces_only_complete_output(tmp_path: Path):
    from scripts.run_challenger_e00_e01 import atomic_install

    target = tmp_path / "result.json"
    target.write_text("old", encoding="utf-8")
    atomic_install(target, b"new")
    assert target.read_text(encoding="utf-8") == "new"


def test_runner_rejects_output_outside_challenger_root(tmp_path: Path):
    from scripts.run_challenger_e00_e01 import run_challenger

    with pytest.raises(ValueError, match="exactly outputs/challenger_70"):
        run_challenger(tmp_path, tmp_path / "outputs" / "crossfit")
