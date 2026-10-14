import numpy as np
import pytest


def test_complete_cache_requires_all_windows():
    from scripts.research.representation_v2.ppg_fusion.cache import complete_cache
    counts = {'complete_valid_packets': 108, 'tail_packets': 8}
    assert complete_cache(counts, 2)
    assert not complete_cache(counts, 1)


def test_unknown_cache_subject_is_not_missing_ppg():
    from scripts.research.representation_v2.ppg_fusion.cache import candidate_features
    from types import SimpleNamespace
    c = SimpleNamespace(event=SimpleNamespace(sid='unknown', start_ms=0, end_ms=10))
    with pytest.raises(ValueError):
        candidate_features([c], {}, {'s': 'A'}, {'A'})


def test_unavailable_scores_fall_back_exactly():
    from scripts.research.representation_v2.ppg_fusion.probe import fallback_scores
    base = np.array([.123456789, .5, .9])
    prediction = np.array([.3, .8, .1])
    result = fallback_scores(base, prediction, np.array([False, True, False]))
    assert np.array_equal(result[[0, 2]], base[[0, 2]])
    assert result[1] == .8


def test_both_branches_have_same_minimum_support():
    from scripts.research.representation_v2.ppg_fusion.probe import sufficient_support
    assert sufficient_support(np.array([1]*10+[0]*30), np.ones(40, bool))
    assert not sufficient_support(np.array([1]*9+[0]*31), np.ones(40, bool))


def test_reproduction_disagreement_is_fatal():
    from scripts.research.representation_v2.ppg_fusion.probe import reproduce
    candidate = {'event': {'sid': 's', 'start_ms': 0, 'end_ms': 10}}
    ref = {'candidates': [candidate], 'groups': ['A'], 'labels': [1], 'scores': {'b': [.4]}}
    reproduce(ref, [candidate], np.array(['A']), np.array([1]), np.array([.4]))
    with pytest.raises(RuntimeError):
        reproduce(ref, [candidate], np.array(['A']), np.array([1]), np.array([.5]))


def test_reproduction_normalizes_json_tuple_roundtrip():
    from scripts.research.representation_v2.ppg_fusion.probe import reproduce
    c = {'event': {'sid': 's', 'start_ms': 0, 'end_ms': 10}, 'macro': {'probabilities': (.1, .2)}}
    r = {'event': c['event'], 'macro': {'probabilities': [.1, .2]}}
    ref = dict(candidates=[r], groups=['A'], labels=[1], scores={'b': [.4]})
    reproduce(ref, [c], np.array(['A']), np.array([1]), np.array([.4]))


def test_health_gate_precedes_runtime_loading(tmp_path, monkeypatch):
    from scripts.research.representation_v2.ppg_fusion import probe
    def forbidden(*args):
        raise AssertionError('registration loaded before policy gate')
    monkeypatch.setattr(probe, 'registration', forbidden)
    with pytest.raises(RuntimeError, match='committed'):
        probe.worker(tmp_path, {}, 'health')


def test_gate_needs_ppg_increment_over_quality():
    from scripts.research.representation_v2.ppg_fusion.probe import gate
    base = dict(f1=.72, n_tp=66, fp=20)
    p = dict(f1=.75, n_tp=70, fp=18)
    assert gate(base, dict(f1=.74), p, [True, True, False])['passed']
    assert not gate(base, dict(f1=.749), p, [True]*3)['passed']


def test_short_event_matching_import_and_counts():
    from scripts.research.representation_v2.ppg_fusion.probe import short_recall
    from src.pipeline.event_stack import EventRef
    e = EventRef(sid='s', start_ms=0, end_ms=599999)
    assert short_recall([e], [e]) == {'gt': 1, 'detected': 1}
    e = EventRef(sid='s', start_ms=0, end_ms=600000)
    assert short_recall([e], [e]) == {'gt': 0, 'detected': 0}


def test_resource_gate_covers_post_write_size():
    from scripts.research.representation_v2.ppg_fusion.cache import resource_gate
    resource_gate(10, 2**20, 2**20, 30*2**30)
    with pytest.raises(RuntimeError):
        resource_gate(10, 2**20, 129*2**20, 30*2**30)


def test_config_tuple_json_equivalence_without_parameter_change():
    from scripts.research.representation_v2.ppg_fusion.probe import same_config
    assert same_config({'grid': (.2, .35)}, {'grid': [.2, .35]})
    assert not same_config({'grid': (.2, .35)}, {'grid': [.2, .4]})
