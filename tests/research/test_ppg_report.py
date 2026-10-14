import numpy as np
import pytest


def test_insufficient_group_and_constant_correlation():
    from scripts.research.representation_v2.ppg.report import describe, correlation
    r = describe(np.array([1, 2, np.nan]))
    assert r['n_valid'] == 2 and r['status'] == 'insufficient'
    assert correlation(np.ones(6), np.arange(6))['rho'] is None
    assert describe(np.arange(5))['median'] == 2


def test_score_boundary():
    from scripts.research.representation_v2.ppg.report import score_bin
    assert [score_bin(v) for v in [0, .249, .25, .5, .75, 1]] == [0, 0, 1, 2, 3, 3]
    with pytest.raises(ValueError):
        score_bin(float('nan'))


def test_acceptance_does_not_guess_from_overlap():
    from scripts.research.representation_v2.ppg.report import selection_status
    e = dict(sid='s', start_ms=0, end_ms=10)
    assert selection_status(e, [e]) == 'accepted_exact'
    assert selection_status(e, [{**e, 'end_ms': 11}]) == 'unknown_overlap'
    assert selection_status(e, []) == 'not_exactly_selected'
    assert selection_status(e, None) == 'unknown_unavailable'


def test_report_requires_sealed_state(tmp_path):
    from scripts.research.representation_v2.ppg.audit import write_json
    from scripts.research.representation_v2.ppg.contract import OUTPUT
    from scripts.research.representation_v2.ppg.report import build_report
    out = tmp_path/OUTPUT; out.mkdir(parents=True)
    write_json(out/'state.json', {'status': 'REGISTERED'})
    with pytest.raises(RuntimeError, match='state'):
        build_report(tmp_path)


def test_direction_ignores_small_groups():
    from scripts.research.representation_v2.ppg.report import compare_groups
    x = np.column_stack([np.arange(10), np.ones(10)])
    y = np.array([True]*5+[False]*5)
    r = compare_groups(x, y)
    assert r[0]['median_true_minus_false'] == -5
    assert r[1]['median_true_minus_false'] == 0
    assert compare_groups(x[:6], y[:6])[0]['median_true_minus_false'] is None
