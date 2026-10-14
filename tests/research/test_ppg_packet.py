import numpy as np
import pytest


def rows(t, n=15):
    return np.tile([t, 3., 5.], (n, 1))


def test_chunk_boundary_and_identical_samples():
    from scripts.research.representation_v2.ppg.packet import iter_packets
    x = rows(1000)
    p = list(iter_packets([x[:7], x[7:]]))
    assert len(p) == 1 and p[0]['valid']
    assert p[0]['values'].shape == (15, 2)
    p = list(iter_packets([x[:7], np.zeros((30, 3)), x[7:]]))
    assert len(p) == 1 and p[0]['valid']


@pytest.mark.parametrize('gap,broken', [(449, True), (450, False), (750, False), (751, True), (-1, True)])
def test_gap_boundaries(gap, broken):
    from scripts.research.representation_v2.ppg.packet import iter_packets
    p = list(iter_packets([np.vstack([rows(1000), rows(1000+gap)])]))
    assert p[1]['break_before'] == broken


@pytest.mark.parametrize('n', [14, 16])
def test_incomplete_packet(n):
    from scripts.research.representation_v2.ppg.packet import iter_packets
    assert not list(iter_packets([rows(1000, n)]))[0]['valid']


def test_absence_neutral_but_invalid_time_splits():
    from scripts.research.representation_v2.ppg.packet import iter_packets
    x = rows(1000)
    p = list(iter_packets([np.vstack([x[:7], [[0, 0, 0]], x[7:]])]))
    assert len(p) == 1 and p[0]['valid']
    p = list(iter_packets([np.vstack([x[:7], [[np.nan, 3, 5]], x[7:]])]))
    assert not any(v['valid'] for v in p)


def test_bad_values_not_dropped_and_repeated_timestamp():
    from scripts.research.representation_v2.ppg.packet import iter_packets
    x = rows(1000, 16); x[4, 1] = np.nan
    p = list(iter_packets([np.vstack([x, rows(1600), rows(1000)])]))
    assert not p[0]['valid'] and len(p[0]['values']) == 16
    assert p[-1]['break_before'] and 'repeated' in p[-1]['reason']
