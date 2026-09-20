"""Contracts for replaying frozen outer-fold evidence without fitting."""

from __future__ import annotations

from pathlib import Path

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


@pytest.mark.integration
def test_fold_zero_replays_frozen_metric():
    """Deleting scoring or substituting a trained path cannot reproduce fold-zero evidence."""

    from src.pipeline.challenger.replay import replay_outer_fold

    trace = replay_outer_fold(Path("."), 0)

    assert (trace.metrics.n_tp, trace.metrics.n_true, trace.metrics.n_pred) == (16, 23, 35)
    assert (len(trace.raw_union), len(trace.admitted_indices)) == (713, 42)
