import json
import pytest


def fixture_inventory(tmp_path, subject='A', sid='s'):
    from scripts.research.representation_v2.ppg.contract import build_inventory
    raw = tmp_path / 'raw'
    (raw / 's').mkdir(parents=True)
    (raw / 's' / 'collect_data.csv').write_text('header\n')
    manifest = {'identities': {'t24': ['A'], 'v7': ['B'], 'outer': ['C']},
                'subject_by_session': {sid: subject}}
    payload = {'candidates': [{'event': {'sid': sid, 'start_ms': 1, 'end_ms': 2}}],
               'labels': None, 'scores': None}
    return build_inventory(manifest, {'a': payload, 'b': payload}, raw)


def test_occurrences_not_deduplicated(tmp_path):
    inv = fixture_inventory(tmp_path)
    assert len(inv['keys']) == 1 and len(inv['occurrences']) == 2
    assert [o['partition'] for o in inv['occurrences']] == ['a', 'b']
    assert all(o['key_index'] == 0 for o in inv['occurrences'])


@pytest.mark.parametrize('subject,sid', [('B', 's'), ('A', '../s'), ('C', 's')])
def test_forbidden_identity_or_path(tmp_path, subject, sid):
    with pytest.raises(ValueError):
        fixture_inventory(tmp_path, subject, sid)


def test_missing_extra_file_and_counts(tmp_path):
    from scripts.research.representation_v2.ppg.contract import validate_inventory, build_inventory
    inv = fixture_inventory(tmp_path)
    validate_inventory(inv, expected=(1, 2, 1))
    with pytest.raises(ValueError):
        validate_inventory(inv)
    manifest = {'identities': {'t24': ['A'], 'v7': [], 'outer': []},
                'subject_by_session': {'s': 'A'}}
    data = {'a': {'candidates': [{'event': dict(sid='s', start_ms=1, end_ms=2)}]}}
    path = tmp_path / 'raw' / 's' / 'collect_data.csv'
    path.unlink()
    with pytest.raises(ValueError):
        build_inventory(manifest, data, tmp_path / 'raw')
    path.write_text('x')
    path.with_name('collect_data2.csv').write_text('x')
    with pytest.raises(ValueError):
        build_inventory(manifest, data, tmp_path / 'raw')


def test_overlapping_subjects_fail_before_file_access(tmp_path):
    from scripts.research.representation_v2.ppg.contract import build_inventory
    with pytest.raises(ValueError):
        build_inventory({'identities': {'t24': ['A'], 'v7': ['A'], 'outer': []}}, {}, tmp_path)
