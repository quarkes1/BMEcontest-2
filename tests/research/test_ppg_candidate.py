import numpy as np
import pytest


def window(start, end, **kw):
    return dict(sid='s', start_ms=start, end_ms=end, valid=True, pair_pass=True,
                motion=2., frequency=1.2, rms3_dc=.01, rms5_dc=.02, **kw)


def test_union_and_median():
    from scripts.research.representation_v2.ppg.candidate import union_length, weighted_median
    assert union_length([(0, 20), (10, 30)]) == 30
    assert weighted_median([1, 3], [1, 1]) == 1
    assert np.isnan(weighted_median([np.nan], [1]))


@pytest.mark.parametrize('n,length,available', [(6, 5000, True), (5, 6000, False), (6, 4990, False)])
def test_window_count_and_coverage_gates(n, length, available):
    from scripts.research.representation_v2.ppg.candidate import aggregate_candidate
    event = dict(sid='s', start_ms=2000000, end_ms=2060000)
    windows = [window(2000000+i*length, 2000000+(i+1)*length) for i in range(n)]
    r = aggregate_candidate(event, windows)
    assert bool(np.isfinite(r['phases'][1]['proxies'][0])) == available
    assert np.isnan(r['p']).all()  # no pre baseline


def test_same_sid_full_denominator_and_anchor():
    from scripts.research.representation_v2.ppg.candidate import aggregate_candidate
    event = dict(sid='s', start_ms=2000000, end_ms=2060000)
    w = window(2000000, 2030000)
    other = {**w, 'sid': 'other', 'end_ms': 2060000}
    r = aggregate_candidate(event, [w, w, other])
    assert r['q'][3] == .5
    assert aggregate_candidate(event, [w], -560)['q'][3] == pytest.approx(29440/60000)


def test_featurewise_missing_and_difference():
    from scripts.research.representation_v2.ppg.candidate import aggregate_candidate
    event = dict(sid='s', start_ms=2000000, end_ms=2180000)
    pre = [window(800000+i*30000, 830000+i*30000) for i in range(30)]
    during = [{**window(2000000+i*30000, 2030000+i*30000), 'frequency': 1.4,
               'rms3_dc': np.nan} for i in range(6)]
    r = aggregate_candidate(event, pre+during)
    assert r['p'][0] == pytest.approx(.2)
    assert np.isnan(r['p'][1]) and r['p'][2] == 0
    assert np.isnan(r['p'][3:]).all()
