"""Frozen encoding must enforce raw ownership, physical gaps and cache integrity."""
from pathlib import Path

import numpy as np
import pytest
import torch


class TinyEncoder(torch.nn.Module):
    def forward(self,x,mask):
        return x.mean((1,2))[:,None].expand(-1,16)


def raw_fixture(root, gap=False):
    p = root/'cache/sessions'; p.mkdir(parents=True)
    t = np.arange(500,dtype=np.int64)*40
    if gap: t[60:] += 1000
    motion=np.stack([np.sin(np.arange(500)*(.12+i*.03)) for i in range(3)]).astype(np.float32)
    np.savez(p/'s.npz',t_acc=t,acc=motion,
             gyro=motion*.3,imu_valid=np.ones(500,bool),row_rate=25.)


def test_disallowed_session_is_rejected_before_io(tmp_path):
    from scripts.research.representation_v2.transformer.cache import encode_session
    with pytest.raises(ValueError,match='not allowed'):
        encode_session(tmp_path,'heldout',{'train'},None,'cpu')
    with pytest.raises(ValueError):
        encode_session(tmp_path,'../s',{'../s'},None,'cpu')


def test_gap_does_not_combine_spans_to_reach_coverage(tmp_path):
    from scripts.research.representation_v2.transformer.cache import encode_session
    raw_fixture(tmp_path,gap=True)
    encoder=TinyEncoder().requires_grad_(False).eval()
    tokens=encode_session(tmp_path,'s',{'s'},encoder,'cpu')
    assert tokens.coverage[0] < .8 and not tokens.valid[0]
    assert tokens.valid[1] and tokens.span_id[0] != tokens.span_id[1]
    assert tokens.coverage[-1] < .8
    assert np.isfinite(tokens.embeddings).all()


def test_encoding_preserves_encoder_and_uses_full_day_grid(tmp_path):
    from scripts.research.representation_v2.transformer.cache import encode_session, encoder_hash
    raw_fixture(tmp_path)
    encoder=TinyEncoder().requires_grad_(False).eval()
    before=encoder_hash(encoder)
    tokens=encode_session(tmp_path,'s',{'s'},encoder,'cpu')
    assert len(tokens.center_ms)==4 and tokens.valid.all()
    np.testing.assert_array_equal(tokens.center_ms,[2500,7500,12500,17500])
    assert encoder_hash(encoder)==before


def test_corrupt_cache_and_wrong_encoder_fail_closed(tmp_path):
    from scripts.research.representation_v2.transformer.cache import load_or_encode, file_hash, encoder_hash
    raw_fixture(tmp_path); encoder=TinyEncoder().requires_grad_(False).eval()
    meta=dict(schema='a6-token-v1',raw_sha256=file_hash(tmp_path/'cache/sessions/s.npz'),
              encoder_sha256=encoder_hash(encoder),normalization='unitfree',fit_subjects=['fd'],
              start_ms=0,end_ms=20000,sid='s',preprocessing={'rate':25})
    kwargs=dict(root=tmp_path,sid='s',allowed_sids={'s'},encoder=encoder,device='cpu',
                cache_root=tmp_path/'derived',metadata=meta)
    original=load_or_encode(**kwargs)
    assert len(load_or_encode(**kwargs).center_ms)==len(original.center_ms)
    with pytest.raises(ValueError,match='encoder'):
        load_or_encode(**{**kwargs,'metadata':{**meta,'encoder_sha256':'x'*64}})
    next((tmp_path/'derived').glob('*.npz')).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):
        load_or_encode(**kwargs)


def test_cache_quota_failure_leaves_no_partial_files(tmp_path):
    from scripts.research.representation_v2.transformer.cache import load_or_encode, file_hash, encoder_hash
    raw_fixture(tmp_path); encoder=TinyEncoder().requires_grad_(False).eval()
    meta=dict(schema='a6-token-v1',raw_sha256=file_hash(tmp_path/'cache/sessions/s.npz'),
              encoder_sha256=encoder_hash(encoder),normalization='unitfree',fit_subjects=['fd'],
              start_ms=0,end_ms=20000,sid='s',preprocessing={'rate':25})
    with pytest.raises(RuntimeError,match='quota'):
        load_or_encode(root=tmp_path,sid='s',allowed_sids={'s'},encoder=encoder,device='cpu',
                       cache_root=tmp_path/'derived',metadata=meta,quota_bytes=1)
    assert not list((tmp_path/'derived').iterdir())


def test_t0_audit_never_reads_raw_bytes(tmp_path,monkeypatch):
    from scripts.research.representation_v2.transformer import cache
    raw_fixture(tmp_path)
    def forbidden_read(*args,**kwargs): raise AssertionError('T0 read raw bytes')
    monkeypatch.setattr(cache,'file_hash',forbidden_read)
    result=cache.audit_data(tmp_path,{'identities':{'t24':['person']},'subject_by_session':{'s':'person'}})
    assert result['total_raw_bytes']>0 and result['raw_bytes_read'] is False
