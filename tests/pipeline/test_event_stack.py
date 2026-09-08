import numpy as np

from src.pipeline.event_stack import (
    EventRef,
    compute_event_metrics,
    select_event_threshold,
)


def test_compute_event_metrics_matches_once_per_truth():
    truths = [EventRef("s1", 100, 200)]
    predictions = [EventRef("s1", 100, 200), EventRef("s1", 110, 190)]

    metrics = compute_event_metrics(predictions, truths, iou_threshold=0.25)

    assert (metrics.n_tp, metrics.n_pred, metrics.n_true) == (1, 2, 1)
    assert metrics.f1 == 2 / 3


def test_select_event_threshold_prefers_stricter_equal_f1_choice():
    candidates = [
        EventRef("s1", 100, 200),
        EventRef("s2", 100, 200),
        EventRef("s3", 100, 200),
        EventRef("s4", 100, 200),
    ]
    truths = [EventRef("s1", 100, 200), EventRef("s2", 100, 200)]

    selected = select_event_threshold(
        candidates,
        np.array([0.9, 0.7, 0.7, 0.7]),
        truths,
    )

    assert selected.threshold == 0.9
    assert selected.metrics.n_tp == selected.metrics.n_pred == 1
