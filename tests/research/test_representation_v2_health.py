import numpy as np
import pytest

from scripts.research.representation_v2.health import logit_diagnostics, validate_health_partition


def test_extreme_logits_are_not_clipped_for_loss():
    result = logit_diagnostics(np.array([0, 1]), np.array([100., -100.]))
    assert result['raw_logit_bce'] == pytest.approx(100.)
    assert result['positive_below_001'] == 1
    assert result['negative_above_099'] == 1
    with pytest.raises(ValueError):
        logit_diagnostics(np.array([0]), np.array([np.inf]))


def test_health_partition_excludes_original_inner_validation():
    validate_health_partition(('a', 'b', 'c'), ('v',), ('a', 'b'), ('c',))
    with pytest.raises(ValueError):
        validate_health_partition(('a', 'b', 'c'), ('v',), ('a', 'b'), ('v',))
    with pytest.raises(ValueError):
        validate_health_partition(('a', 'b', 'c'), ('v',), ('a', 'b'), ('b', 'c'))
