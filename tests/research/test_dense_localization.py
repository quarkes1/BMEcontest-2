import numpy as np
import pytest
from scripts.research.representation_v2.transformer.timeline import SessionTokens


def timeline(n=30):
    return SessionTokens('sid', np.arange(n)*5000+2500, np.zeros(n, int),
                         np.ones(n), np.ones(n, bool), np.zeros((n, 16)), 0, n*5000)


def test_two_hits_merge_at_fixed_threshold_and_single_hit_drops():
    from scripts.research.representation_v2.localization.probe import proposals
    t = timeline(); p = np.zeros(30); p[[0, 2, 29]] = .5
    events = proposals(t, p)
    assert [(e.sid, e.start_ms, e.end_ms) for e in events] == [('sid', 0, 15000)]


@pytest.mark.parametrize('break_type', ['invalid', 'span', 'gap'])
def test_never_merge_through_acquisition_break(break_type):
    from scripts.research.representation_v2.localization.probe import proposals
    t = timeline(4); p = np.array([1., 1., 1., 1.])
    if break_type == 'invalid':
        t.valid[2] = False
    elif break_type == 'span':
        t.span_id[2:] = 1
    else:
        t.center_ms[2:] += 20000
        object.__setattr__(t, 'end_ms', 40000)
    events = proposals(t, p)
    assert events[0].end_ms == 10000
    assert len(events) == (1 if break_type == 'invalid' else 2)


def test_support_clipped_and_long_hit_gap_splits():
    from scripts.research.representation_v2.localization.probe import proposals
    t = timeline(30); object.__setattr__(t, 'start_ms', 1000)
    object.__setattr__(t, 'end_ms', 149000)
    p = np.zeros(30); p[[0, 1, 28, 29]] = 1
    assert [(e.start_ms, e.end_ms) for e in proposals(t, p)] == [(1000, 10000), (140000, 149000)]


def test_invalid_probability_rejected():
    from scripts.research.representation_v2.localization.probe import proposals
    for p in ([np.nan]*30, [1.1]*30, [0.]*29):
        with pytest.raises(ValueError):
            proposals(timeline(), p)


def test_unsealed_report_refused_before_truth_read(tmp_path):
    from scripts.research.representation_v2.localization.probe import report
    with pytest.raises(RuntimeError, match='committed'):
        report(tmp_path)


def test_reachable_count_is_not_one_to_one_tp():
    from scripts.research.representation_v2.localization.probe import reach
    from scripts.research.representation_v2.validation_metrics import matched_indices
    from src.pipeline.event_stack import EventRef
    events = [EventRef('s', 0, 10000)]
    truths = [EventRef('s', 0, 6000), EventRef('s', 4000, 10000)]
    assert reach(events, truths) == {0, 1}
    assert len(matched_indices(events, truths)[1]) == 1
