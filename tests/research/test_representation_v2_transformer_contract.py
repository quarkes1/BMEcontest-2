"""A6 registration must reject missing fits, ambiguous identities and stale caches."""
from copy import deepcopy

import pytest


def identities():
    return {
        't24': [f't{i}' for i in range(24)], 'v7': [f'v{i}' for i in range(7)],
        'outer': ['o'], 'source_train': [f'fd{i}' for i in range(27)],
        'source_val': [f'fd{i}' for i in range(27, 34)],
    }


def test_registered_counts_reject_missing_refit():
    from scripts.research.representation_v2.transformer.contract import validate_fit_counts
    expected = dict(producer_fits=120, base_fits=24, neural_fits=36, steps=36000)
    validate_fit_counts(expected)
    with pytest.raises(ValueError):
        validate_fit_counts({**expected, 'producer_fits': 119})


@pytest.mark.parametrize('change', ['empty', 'duplicate', 'overlap', 'short_t24', 'short_source'])
def test_identity_partitions_fail_closed(change):
    from scripts.research.representation_v2.transformer.contract import validate_identities
    value = identities()
    if change == 'empty': value['outer'] = []
    if change == 'duplicate': value['v7'][1] = value['v7'][0]
    if change == 'overlap': value['outer'] = ['t0']
    if change == 'short_t24': value['t24'].pop()
    if change == 'short_source': value['source_train'].pop()
    with pytest.raises(ValueError):
        validate_identities(value)


def test_namespace_does_not_prove_source_target_disjointness():
    from scripts.research.representation_v2.transformer.contract import validate_identities
    with pytest.raises(ValueError, match='source.*target'):
        validate_identities(identities())
    validate_identities(identities(), source_target_evidence={'confirmed_disjoint': True,
                        'basis': 'dataset custodian confirms independent cohorts'})


def test_cache_identity_binds_encoder_fit_subjects_time_and_preprocessing():
    from scripts.research.representation_v2.transformer.contract import cache_key
    meta = dict(schema='a6-token-v1', raw_sha256='a'*64, encoder_sha256='b'*64,
                normalization='unitfree', fit_subjects=['source1'],
                start_ms=0, end_ms=10000, sid='s', preprocessing={'rate': 25})
    original = cache_key(meta)
    assert cache_key(dict(reversed(list(meta.items())))) == original
    for field, value in [('encoder_sha256', 'c'*64), ('fit_subjects', ['target']),
                         ('start_ms', 5), ('normalization', 'other')]:
        changed = deepcopy(meta); changed[field] = value
        assert cache_key(changed) != original
    with pytest.raises(ValueError):
        cache_key({'schema': 'a6-token-v1'})


def test_config_is_immutable_and_rejects_silent_budget_change():
    from dataclasses import FrozenInstanceError
    from scripts.research.representation_v2.transformer.contract import T1Config
    config = T1Config()
    with pytest.raises(FrozenInstanceError):
        config.steps = 2
    with pytest.raises(ValueError):
        T1Config(steps=999)
