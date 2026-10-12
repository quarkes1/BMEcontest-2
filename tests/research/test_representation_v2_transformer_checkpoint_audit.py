"""Frozen audit must neither silently rebuild caches nor relabel invalid motion."""
import importlib
import json

import numpy as np
import pytest
import torch

from scripts.research.representation_v2.transformer.cache import file_hash
from scripts.research.representation_v2.transformer.heads import build_head


def audit():
    name = 'scripts.research.representation_v2.transformer.checkpoint_audit'
    assert importlib.util.find_spec(name) is not None, 'frozen checkpoint audit not implemented'
    return importlib.import_module(name)


def token_fixture(tmp_path):
    path = tmp_path / 'tokens.npz'
    np.savez_compressed(path, center_ms=np.array([2500, 7500]), span_id=np.array([0, 0]),
                        coverage=np.ones(2), valid=np.ones(2, bool),
                        embeddings=np.ones((2, 16), np.float32), start_ms=0, end_ms=10000)
    metadata = {'sid': 'allowed', 'raw_sha256': 'raw', 'source_checkpoint': 'source'}
    header = tmp_path / 'tokens.json'
    header.write_text(json.dumps({'metadata': metadata, 'sha256': file_hash(path)}))
    return header, path


def test_readonly_cache_preserves_arrays_and_rejects_unknown_identity(tmp_path):
    mod = audit()
    header, path = token_fixture(tmp_path)
    before = file_hash(path)
    tokens = mod.read_tokens(header, allowed_sids={'allowed'}, raw_hashes={'allowed': 'raw'}, source_sha='source')
    assert tokens.center_ms.tolist() == [2500, 7500]
    assert file_hash(path) == before
    with pytest.raises(ValueError):
        mod.read_tokens(header, allowed_sids={'other'}, raw_hashes={'allowed': 'raw'}, source_sha='source')


def test_corrupt_cache_is_rejected_not_rebuilt(tmp_path):
    mod = audit()
    header, path = token_fixture(tmp_path)
    path.write_bytes(b'corrupted')
    with pytest.raises(ValueError):
        mod.read_tokens(header, allowed_sids={'allowed'}, raw_hashes={'allowed': 'raw'}, source_sha='source')
    assert path.read_bytes() == b'corrupted'


def test_missing_cache_fails_without_creating_files(tmp_path):
    mod = audit()
    with pytest.raises(FileNotFoundError):
        mod.read_tokens(tmp_path / 'missing.json', allowed_sids={'allowed'}, raw_hashes={}, source_sha='source')
    assert not list(tmp_path.iterdir())


def test_profile_excludes_invalid_motion_but_keeps_candidate_metrics():
    mod = audit()
    row = mod.profile([1, 1, 0], [.8, .7, .1], [.6, .9, .2], [True, False, True])
    assert row['positive_motion']['n'] == 1
    assert row['positive_motion']['median'] == pytest.approx(.6)
    assert row['negative_motion']['n'] == 1
    assert row['candidate_ap'] == pytest.approx(1.)
    assert row['invalid_count'] == 1
    assert not any('f1' in key.lower() for key in row)


def test_profile_rejects_misaligned_or_nonfinite_scores():
    mod = audit()
    for motion in ([.2], [.2, np.nan]):
        with pytest.raises(ValueError):
            mod.profile([1, 0], [.8, .1], motion, [True, True])


def test_real_checkpoint_prediction_preserves_order_fallback_and_weights(tmp_path):
    mod = audit()
    model = build_head('c')
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.head.bias.fill_(float(np.log(3)))
    path = tmp_path / 'c.pt'
    torch.save({'state_dict': model.state_dict(), 'center': torch.zeros(16), 'scale': torch.ones(16)}, path)
    seq = {'sid': 's', 'subject': 'p', 'start_ms': 0, 'chunks': [
        {'region': 1, 'indices': np.array([0]), 'embeddings': np.ones((1, 16), np.float32),
         'coverage': np.ones(1), 'times': np.array([2500]), 'valid': np.array([True])}]}
    before = file_hash(path)
    motion, fused = mod.score_checkpoint(path, 'c', [seq, seq], [.8, .2], [True, False], 'cpu')
    assert motion.tolist() == pytest.approx([.75, .75], abs=1e-6)
    assert fused.tolist() == pytest.approx([.7875, .2], abs=1e-6)
    assert file_hash(path) == before
