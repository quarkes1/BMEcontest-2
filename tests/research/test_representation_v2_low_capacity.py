import numpy as np
import torch
import pytest

from scripts.research.representation_v2.dataset import TokenBag
from scripts.research.representation_v2.low_capacity import subject_weights, pool_ordered, fit_readout


def test_subject_weight_total_is_equal_without_balancing_classes():
    np.testing.assert_allclose(subject_weights(['a', 'a', 'a', 'b']), [2/3, 2/3, 2/3, 2])


def test_pool_keeps_order_and_ignores_invalid_tokens():
    bag = TokenBag(('s', 0, 20000), np.zeros((4, 6, 125), np.float32),
                   np.ones((4, 125), bool), np.array([1, 1, 0, 1], bool),
                   np.array([2500, 7500, 12500, 17500]), 0, 20000, np.ones(4))
    z = np.repeat(np.array([1, 2, 999, 4])[:, None], 16, axis=1)
    row = pool_ordered(z, bag)
    assert row.shape == (133,)
    np.testing.assert_array_equal(row[[0, 32, 64, 96]], [1, 2, 0, 4])
    np.testing.assert_array_equal(row[128:132], np.ones(4))
    assert row[-1] == pytest.approx(np.log1p(20))


def test_readout_freezes_train_scaling_and_learns_simple_signal():
    x = np.array([[-2.], [-1.], [1.], [2.]])
    fitted = fit_readout(x, np.array([0, 0, 1, 1]), ['a', 'b', 'a', 'b'])
    before = fitted['scaler'].mean_.copy()
    p = fitted['model'].predict_proba(fitted['scaler'].transform(np.array([[-1.], [1.]])))[:, 1]
    assert p[0] < .5 < p[1]
    np.testing.assert_array_equal(before, fitted['scaler'].mean_)
