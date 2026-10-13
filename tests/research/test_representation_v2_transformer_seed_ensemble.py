"""Fixed-seed, partition-contained ensemble contracts."""
import importlib

import numpy as np
import pytest


def mod():
    name = 'scripts.research.representation_v2.transformer.seed_ensemble'
    assert importlib.util.find_spec(name) is not None, 'seed ensemble not implemented'
    return importlib.import_module(name)


def test_averages_probabilities_not_best_seed():
    assert mod().average_three([[.1, .9], [.2, .3], [.9, .3]]).tolist() == pytest.approx([.4, .5])


@pytest.mark.parametrize('values', [[[.1], [.2]], [[.1], [.2], [.3, .4]], [[.1], [.2], [np.nan]], [[.1], [.2], [1.1]]])
def test_rejects_partial_misaligned_or_invalid_members(values):
    with pytest.raises(ValueError):
        mod().average_three(values)


def test_reproduction_guard_rejects_reordered_candidates():
    m = mod()
    reference = {'candidates': [{'x': 1}, {'x': 2}], 'labels': [1, 0], 'groups': ['a', 'b'], 'scores': {'b': [.8, .2]}}
    m.verify_reproduction(reference, [{'x': 1}, {'x': 2}], [1, 0], ['a', 'b'], [.8, .2])
    with pytest.raises(ValueError):
        m.verify_reproduction(reference, [{'x': 2}, {'x': 1}], [1, 0], ['a', 'b'], [.8, .2])
    with pytest.raises(ValueError):
        m.verify_reproduction(reference, reference['candidates'], [1, 0], ['a', 'b'], [.7, .2])


def test_predeclared_seed_offsets_distinguish_partitions():
    m = mod()
    assert m.seeds_for(0) == (20261012, 20361012, 20461012)
    assert m.seeds_for(11) == (20261023, 20361023, 20461023)
    with pytest.raises(ValueError):
        m.seeds_for(12)


def test_primary_gate_does_not_accept_secondary_or_single_regression():
    m = mod()
    assert m.ensemble_gate(.72, .76, .77, [True, True, False])['passed']
    assert not m.ensemble_gate(.72, .76, .75, [True, True, True])['passed']
    assert not m.ensemble_gate(.72, .76, .77, [True, False, False])['passed']


def test_cache_snapshot_anchors_both_header_and_arrays_and_rejects_partial(tmp_path):
    m = mod()
    (tmp_path/'x.json').write_bytes(b'header')
    (tmp_path/'x.npz').write_bytes(b'array')
    snapshot = m.cache_snapshot(tmp_path)
    assert set(snapshot) == {'x.json', 'x.npz'}
    (tmp_path/'x.npz').write_bytes(b'changed')
    assert snapshot['x.npz'] != m.cache_snapshot(tmp_path)['x.npz']
    (tmp_path/'orphan.json').write_bytes(b'partial')
    with pytest.raises(ValueError):
        m.cache_snapshot(tmp_path)


def test_policy_partitions_cover_subjects_once():
    m = mod()
    m.policy_coverage([['a'], ['b'], ['c']], {'a', 'b', 'c'})
    for parts in ([['a'], ['b']], [['a'], ['a', 'b'], ['c']]):
        with pytest.raises(ValueError):
            m.policy_coverage(parts, {'a', 'b', 'c'})


def test_wrapper_directions_report_both_arms():
    m = mod()
    original = [{'events': {'c': {'f1': .7}, 't': {'f1': .8}}}]*3
    new = [{'events': {'c': {'f1': .71}, 't': {'f1': .79}}},
           {'events': {'c': {'f1': .70}, 't': {'f1': .82}}},
           {'events': {'c': {'f1': .69}, 't': {'f1': .80}}}]
    assert m.wrapper_directions(new, original) == {'c': [True, True, False], 't': [False, True, True]}
