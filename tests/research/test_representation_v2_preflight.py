"""P0 must account for the full training topology before fitting."""

from scripts.research.representation_v2.preflight import projected_fits, required_seconds_per_step
from scripts.research.representation_v2 import run


def test_projected_fits_include_nested_stack_and_source():
    projection = projected_fits(outer_wrappers=5, inner_folds=4, arms=4, meta_folds=3)
    assert projection == {
        "p2_target_fits": 80,
        "p3_primary_additional_fits": 60,
        "p3_confirmation_fits": 80,
        "fd_source_fits": 1,
        "total_target_steps_if_all_gates_pass": 440_000,
        "total_steps_including_fd": 450_000,
    }


def test_budget_includes_every_retry_in_the_same_cap():
    assert required_seconds_per_step(24, 450_000) == 0.192


def test_preflight_writes_reviewable_artifacts_without_training(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "audit", lambda _root: {
        "status": "P0_READY", "base_git_sha": "abc", "protected_tree_sha256": "def",
        "budget": {"total_steps_including_fd": 450_000},
    })
    result = run.preflight(tmp_path, tmp_path / "out")
    assert result["status"] == "P0_READY"
    assert (tmp_path / "out/plan.json").is_file()
    assert (tmp_path / "out/evidence_review.json").is_file()
    assert (tmp_path / "out/state.json").is_file()
    assert not (tmp_path / "out/checkpoints").exists()
