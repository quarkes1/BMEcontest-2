"""Contracts for replaying frozen outer-fold evidence without fitting."""

from __future__ import annotations

from pathlib import Path
from dataclasses import asdict

import pytest

from src.pipeline.event_stack import EventRef


def test_official_match_indices_preserves_stable_tie_order():
    """Equal-IoU pairs retain the input Cartesian order used by official scoring."""

    from src.pipeline.challenger.replay import official_match_indices

    pairs = official_match_indices(
        [EventRef("s", 0, 4), EventRef("s", 0, 4)], [EventRef("s", 0, 4)]
    )

    assert [(pair.prediction_index, pair.truth_index) for pair in pairs] == [(0, 0)]


def test_official_match_indices_accepts_iou_boundary_and_rejects_other_sessions():
    """The .25 boundary is inclusive, while session identity is mandatory."""

    from src.pipeline.challenger.replay import official_match_indices

    pairs = official_match_indices(
        [EventRef("left", 0, 4), EventRef("right", 0, 4)],
        [EventRef("left", 3, 4), EventRef("other", 0, 4)],
    )

    assert [(pair.prediction_index, pair.truth_index, pair.iou) for pair in pairs] == [
        (0, 0, 0.25)
    ]


def test_summary_run_configs_rejects_duplicate_or_disagreeing_fold_payloads():
    """A replay cannot borrow a threshold/config hash from a different fold."""

    from src.pipeline.challenger.replay import validate_summary_run_configs

    configs = {fold: {"outer_fold": fold, "seed": 7} for fold in range(5)}
    summary = {"run_configs": [dict(configs[fold]) for fold in range(5)]}

    assert validate_summary_run_configs(summary, configs) == configs

    duplicate = {"run_configs": [dict(configs[0]) for _ in range(5)]}
    with pytest.raises(RuntimeError, match="unique.*0 through 4"):
        validate_summary_run_configs(duplicate, configs)

    disagreement = {"run_configs": [dict(configs[fold]) for fold in range(5)]}
    disagreement["run_configs"][3]["seed"] = 8
    with pytest.raises(RuntimeError, match="disagrees"):
        validate_summary_run_configs(disagreement, configs)


def test_replay_config_rejects_research_only_context_v2():
    """Sealed incumbent replay never becomes a v2 runtime by accident."""

    from src.pipeline.challenger.replay import _config
    from src.pipeline.runner import RunConfig

    config = RunConfig(
        outer_fold=0, micro_enabled=True, candidate_control_enabled=True,
        context_features_version="v2", admission_minimum_recall=0.80,
    )
    with pytest.raises(RuntimeError, match="requires human promotion/runtime implementation"):
        _config(asdict(config))


@pytest.mark.integration
def test_fold_zero_replays_frozen_metric():
    """Deleting scoring or substituting a trained path cannot reproduce fold-zero evidence."""

    from src.pipeline.challenger.replay import replay_outer_fold

    trace = replay_outer_fold(Path("."), 0)

    assert (trace.metrics.n_tp, trace.metrics.n_true, trace.metrics.n_pred) == (16, 23, 35)
    assert (len(trace.raw_union), len(trace.admitted_indices)) == (713, 42)
