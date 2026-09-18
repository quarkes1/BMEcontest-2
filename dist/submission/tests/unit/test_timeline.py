import numpy as np


def _session_with_timestamps(values):
    from src.data.loader import SessionData

    timestamps = np.asarray(values, dtype=np.int64)
    count = len(timestamps)
    return SessionData(
        acc=np.ones((3, count), dtype=np.float32), gyro=np.ones((3, count), dtype=np.float32),
        ppg=np.zeros((44, count), dtype=np.float32), t_acc=timestamps,
        t_ppg=np.full(count, -1, dtype=np.int64), imu_valid=np.ones(count, dtype=bool),
        ppg_valid=np.zeros(count, dtype=bool), meta={},
    )


def test_gap_creates_separate_spans():
    from src.pipeline.preprocessing.timeline import valid_imu_spans

    session = _session_with_timestamps([0, 50, 100, 10_000, 10_050])
    assert [(span.start_ms, span.end_ms) for span in valid_imu_spans(session)] == [(0, 100), (10_000, 10_050)]


def test_regression_starts_new_span_without_changing_ordered_case():
    from src.pipeline.preprocessing.timeline import timeline_regressions, valid_imu_spans

    ordered = _session_with_timestamps([0, 50, 100, 150])
    assert timeline_regressions(ordered) == 0
    assert [(s.start_ms, s.end_ms) for s in valid_imu_spans(ordered)] == [(0, 150)]

    regressed = _session_with_timestamps([0, 50, 100, 25, 75, 125])
    assert timeline_regressions(regressed) == 1
    spans = valid_imu_spans(regressed)
    assert [(s.start_ms, s.end_ms) for s in spans] == [(0, 100), (25, 125)]
    assert [s.row_indices.tolist() for s in spans] == [[0, 1, 2], [3, 4, 5]]


def test_window_starts_stay_within_span_and_coverage():
    from src.pipeline.preprocessing.timeline import TimelineSpan, window_starts

    span = TimelineSpan(start_ms=0, end_ms=1_000, timestamps_ms=np.arange(0, 1_001, 100))
    np.testing.assert_array_equal(window_starts(span, 500, 250, 0.8), np.array([0, 250, 500]))
