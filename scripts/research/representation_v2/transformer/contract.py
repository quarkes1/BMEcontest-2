"""Immutable experiment identity, cache provenance and exact fit budgets."""
from dataclasses import dataclass
import hashlib
import json

IDENTITY = 'RV2-A6-T1-001'
OUTPUT = 'outputs/challenger_70/representation_v2/amendment_a6_transformer/t1'
ARMS = ('b', 'p', 'c', 't')
SOURCE_SHA = 'c6276837b1366117075e7e8a7504e084926cbfc5d80b98b863e442255995efc4'
SOURCE_TARGET_EVIDENCE = {
    'confirmed_disjoint': True,
    'basis': 'User confirmed independent target and FD-I participant cohorts, 2026-10-12',
    'limitation': 'Anonymous public identifiers cannot independently establish cross-dataset identity.',
}


@dataclass(frozen=True)
class T1Config:
    steps: int = 1000
    batch_size: int = 16
    seed: int = 20261012
    chunk_tokens: int = 360
    context_ms: int = 600000
    cache_bytes: int = 2 * 1024**3
    evidence_bytes: int = 256 * 1024**2
    wall_seconds: int = 36 * 3600
    fit_seconds: int = 1200

    def __post_init__(self):
        expected = (1000, 16, 20261012, 360, 600000, 2*1024**3,
                    256*1024**2, 36*3600, 1200)
        if tuple(self.__dict__.values()) != expected:
            raise ValueError('registered T1 configuration differs')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def validate_identities(identities, *, source_target_evidence=None):
    required = {'t24', 'v7', 'outer', 'source_train', 'source_val'}
    if set(identities) != required:
        raise ValueError('identity fields differ')
    seen = set()
    for key, values in identities.items():
        if (not isinstance(values, (list, tuple)) or not values
                or any(not isinstance(s, str) or not s for s in values)
                or len(set(values)) != len(values) or seen & set(values)):
            raise ValueError('identity overlap, empty or duplicate population: ' + key)
        seen.update(values)
    for key, size in [('t24', 24), ('v7', 7), ('source_train', 27), ('source_val', 7)]:
        if len(identities[key]) != size:
            raise ValueError('registered population differs: ' + key)
    evidence = source_target_evidence or {}
    if evidence.get('confirmed_disjoint') is not True or not evidence.get('basis'):
        raise ValueError('source/target participant disjointness not confirmed')


def cache_key(meta):
    required = {'schema', 'raw_sha256', 'encoder_sha256', 'normalization',
                'fit_subjects', 'start_ms', 'end_ms', 'sid', 'preprocessing'}
    if not required <= set(meta) or any(meta[k] is None for k in required):
        raise ValueError('incomplete token cache provenance')
    for name in ('raw_sha256', 'encoder_sha256'):
        if not isinstance(meta[name], str) or len(meta[name]) != 64:
            raise ValueError('invalid cache checksum')
    if not meta['sid'] or meta['end_ms'] <= meta['start_ms']:
        raise ValueError('invalid cache session bounds')
    return digest(meta)


def validate_fit_counts(counts):
    expected = dict(producer_fits=120, base_fits=24, neural_fits=36, steps=36000)
    if counts != expected:
        raise ValueError('registered fit budget differs')
