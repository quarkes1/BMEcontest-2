import numpy as np
import pytest


def test_offset_gradient_matches_finite_difference():
    from scripts.research.representation_v2.ppg_residual.model import objective
    x = np.array([[1, -.5], [1, .3], [1, 1.2]])
    y = np.array([0, 1, 1]); prior = np.array([-.7, .2, .8]); w = np.array([1., 2., 1.])
    beta = np.array([.1, -.2])
    loss, grad = objective(beta, x, y, prior, w)
    for j in range(2):
        delta = np.eye(2)[j]*1e-6
        numeric = (objective(beta+delta, x, y, prior, w)[0]-objective(beta-delta, x, y, prior, w)[0])/2e-6
        assert grad[j] == pytest.approx(numeric, abs=1e-6)


def test_zero_correction_and_missing_fallback():
    from scripts.research.representation_v2.ppg_residual.model import predict
    model = dict(center=[0., 0.], scale=[1., 1.], beta=[0., 0., 0.], active=True)
    base = np.array([.2, .8, 0., 1.])
    p = predict(model, np.ones((4, 2)), base, np.array([True, True, False, False]))
    assert np.allclose(p, base, atol=1e-15)
    assert np.array_equal(p[2:], base[2:])


def test_fit_uses_training_moments_only_and_finite_outputs():
    from scripts.research.representation_v2.ppg_residual.model import fit, predict
    x = np.column_stack([np.arange(40), np.full(40, np.nan)])
    y = np.array([0]*30+[1]*10); base = np.full(40, .5); groups = np.array(['A']*20+['B']*20)
    m = fit(x, y, base, groups)
    assert m['center'][0] == 19.5 and m['center'][1] == 0
    center = m['center'].copy()
    p = predict(m, np.array([[1e10, np.nan]]), np.array([.5]), np.array([True]))
    assert np.isfinite(p).all() and m['center'] == center


def test_weak_support_returns_inactive_prior():
    from scripts.research.representation_v2.ppg_residual.model import fit, predict
    y = np.array([1]*9+[0]*31)
    m = fit(np.zeros((40, 2)), y, np.full(40, .4), np.array(['A']*40))
    assert not m['active']
    assert np.array_equal(predict(m, np.zeros((2, 2)), np.array([.2, .7]), np.array([True, True])), [.2, .7])


def test_heldout_partition_rejected_before_fit():
    from scripts.research.representation_v2.ppg_residual.probe import check_partition
    with pytest.raises(ValueError):
        check_partition({'A'}, {'A'}, set())
    with pytest.raises(ValueError):
        check_partition({'A'}, {'B'}, {'A'})
    check_partition({'A'}, {'B'}, {'C'})


def test_unused_neural_source_is_not_a_prior_ancestor():
    from scripts.research.representation_v2.ppg_residual.probe import base_graph
    ref = {'graph': {'base': {'fit_subjects': ['A'], 'parents': ['candidate']},
          'candidate': {'fit_subjects': [], 'parents': ['producer']},
          'producer': {'fit_subjects': ['A'], 'parents': []},
          'source': {'fit_subjects': ['FD-X'], 'parents': []}}}
    graph = base_graph(ref)
    assert 'source' not in graph and graph['prior_score']['parents'] == ['base']


def test_health_gate_before_any_data(tmp_path):
    from scripts.research.representation_v2.ppg_residual import probe
    with pytest.raises(RuntimeError, match='committed'):
        probe.worker(tmp_path, {}, 'health')


def test_policy_seal_covers_final_resource_budget(tmp_path):
    from scripts.research.representation_v2.ppg_residual.probe import seal_policy
    from scripts.research.representation_v2.ppg.audit import write_json, verify_hashes
    from scripts.research.representation_v2.ppg.contract import read_json
    for k in range(3):
        write_json(tmp_path/f'wrapper{k}_policies.json', {'fixed': True})
    write_json(tmp_path/'policy_resources.json', {'elapsed_seconds': 123})
    seal_policy(tmp_path)
    seal = read_json(tmp_path/'policy_seal.json')
    assert 'policy_resources.json' in seal
    verify_hashes(tmp_path, seal)
    write_json(tmp_path/'policy_resources.json', {'elapsed_seconds': 0})
    with pytest.raises(RuntimeError):
        verify_hashes(tmp_path, seal)


def test_gt_migration_tracks_one_to_one_gain_and_loss():
    from scripts.research.representation_v2.ppg_residual.probe import gt_migration
    from src.pipeline.event_stack import EventRef
    truths = [EventRef('s', 0, 1000), EventRef('s', 2000, 3000)]
    row = gt_migration({'b': [truths[0]], 'q': [truths[0]], 'p': [truths[1], truths[1]]}, truths)
    assert row['p']['gained_indices'] == [1]
    assert row['p']['lost_indices'] == [0]
    assert row['p']['matched_indices'] == [1]
