"""A5: reject contaminated/degenerate partitions, not merely different seeds."""
import pytest


def subjects():
    return tuple(f'p{i:02d}' for i in range(24))


def history():
    s = subjects()
    return tuple((tuple(x for x in s if x not in s[k*8:(k+1)*8]), s[k*8:(k+1)*8]) for k in range(3))


def test_registered_partitions_are_deterministic_balanced_and_label_free():
    from scripts.research.representation_v2 import validation_probe as vp
    repeats = vp.registered_partitions(subjects(), history())
    assert repeats == vp.registered_partitions(tuple(reversed(subjects())), history())
    assert len(repeats) == 2
    for folds in repeats:
        assert sorted(s for _, h in folds for s in h) == list(subjects())
        assert all(len(t) == 16 and len(h) == 8 and not set(t) & set(h) for t, h in folds)


@pytest.mark.parametrize('s', [(), subjects()[:-1], subjects()[:-1]+('p00',), subjects()[:-1]+('',)])
def test_invalid_population_is_not_silently_adjusted(s):
    from scripts.research.representation_v2 import validation_probe as vp
    with pytest.raises(ValueError):
        vp.registered_partitions(s, history())


def test_equivalent_historical_partition_rejected_even_if_folds_reordered():
    from scripts.research.representation_v2 import validation_probe as vp
    repeats = vp.registered_partitions(subjects(), history())
    with pytest.raises(ValueError, match='equivalent'):
        vp.registered_partitions(subjects(), tuple(reversed(repeats[0])))


def test_identity_overlap_is_rejected():
    from scripts.research.representation_v2 import validation_probe as vp
    with pytest.raises(ValueError, match='overlap'):
        vp.validate_identities({'t24': list(subjects()), 'v7': ['p00'], 'outer': ['x'],
                                'source_train': ['FD-R-1'], 'source_val': ['FD-R-2']})


def test_manifest_hash_change_blocks_loading(tmp_path):
    from scripts.research.representation_v2 import validation_probe as vp
    out = tmp_path / vp.OUTPUT
    vp.ep.write_json(out/'manifest.json', {'id': vp.IDENTITY})
    vp.ep.write_json(out/'plan.json', {'id': vp.IDENTITY, 'manifest_sha256': '0'*64})
    with pytest.raises(RuntimeError, match='manifest'):
        vp.load_manifest(tmp_path)
