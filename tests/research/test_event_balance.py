from types import SimpleNamespace
import numpy as np
import pytest
from src.pipeline.event_stack import EventRef


def fixture():
    truths = [EventRef('a', 0, 1000), EventRef('a', 2000, 3000), EventRef('b', 0, 1000)]
    events = [truths[0], truths[0], truths[1], EventRef('a', 5000, 6000),
              truths[2], EventRef('b', 5000, 6000), EventRef('b', 7000, 8000)]
    return [SimpleNamespace(event=e) for e in events], np.array([1, 1, 1, 0, 1, 0, 0]), np.array(['A']*4+['B']*3), truths, {'a': 'A', 'b': 'B'}


def test_subject_event_weights_preserve_global_class_mass():
    from scripts.research.representation_v2.event_balance.model import sample_weights
    cs, y, groups, truths, owners = fixture()
    w = sample_weights(cs, y, groups, truths, owners)
    assert w.tolist() == pytest.approx([.5, .5, 1, 1.5, 2, .75, .75])
    assert w[y == 1].sum() == pytest.approx(4)
    assert w[y == 0].sum() == pytest.approx(3)
    assert w.mean() == pytest.approx(1)


@pytest.mark.parametrize('bad', ['label', 'owner', 'group'])
def test_misaligned_training_identity_rejected(bad):
    from scripts.research.representation_v2.event_balance.model import sample_weights
    cs, y, groups, truths, owners = fixture()
    if bad == 'label': y[0] = 0
    elif bad == 'owner': owners.pop('b')
    else: groups[0] = 'Z'
    with pytest.raises(ValueError): sample_weights(cs, y, groups, truths, owners)


def test_subject_without_positive_is_not_given_fictional_weight():
    from scripts.research.representation_v2.event_balance.model import sample_weights
    cs, y, groups, truths, owners = fixture()
    cs.append(SimpleNamespace(event=EventRef('c', 0, 1000)))
    w = sample_weights(cs, np.r_[y, 0], np.concatenate([groups, ['C']]), truths, {**owners, 'c': 'C'})
    assert np.isfinite(w).all() and w[-1] == pytest.approx(4/3)


def test_unit_weights_reproduce_existing_verifier():
    from scripts.research.representation_v2.event_balance.model import probabilities
    from scripts.research.representation_v2.event_probe import verifier_probabilities
    rng = np.random.default_rng(42); x = rng.normal(size=(100, 116)); z = x[:5].copy()
    y = np.arange(100)%2
    expected = verifier_probabilities(x, y, z, seed=10)
    assert np.allclose(probabilities(x, y, z, np.ones(100), seed=10), expected, atol=1e-6, rtol=1e-5)


def test_uncommitted_policy_stops_before_runtime(tmp_path):
    from scripts.research.representation_v2.event_balance.probe import worker
    with pytest.raises(RuntimeError, match='committed'):
        worker(tmp_path, {}, 'health')
