"""Readouts must share subject-balanced samples and train-only token moments."""
import numpy as np
import pytest


def sequence(sid='s',subject='train',value=1.):
    return dict(sid=sid,subject=subject,start_ms=0,chunks=[dict(region=1,
                indices=np.arange(3),embeddings=np.full((3,16),value,np.float32),
                coverage=np.ones(3,np.float32),times=np.array([2500,7500,12500]),valid=np.ones(3,bool))])


def test_invalid_motion_preserves_incumbent_score():
    from scripts.research.representation_v2.transformer.training import fuse_scores
    np.testing.assert_allclose(fuse_scores(np.array([.8,.2]),np.array([.4,.9]),np.array([True,False])),[.7,.2])
    with pytest.raises(ValueError): fuse_scores(np.array([.8]),np.array([np.nan]),np.array([False]))


def test_moments_deduplicate_tokens_and_reject_score_subject():
    from scripts.research.representation_v2.transformer.training import fit_moments
    center,scale=fit_moments([sequence(),sequence(),sequence('s2','other',3.)],{'train','other'})
    np.testing.assert_allclose(center,np.full(16,2.))
    np.testing.assert_allclose(scale,np.ones(16))
    with pytest.raises(ValueError,match='subject'):
        fit_moments([sequence(subject='heldout')],{'train'})


def test_paired_sampler_and_two_step_fits_are_auditable():
    from scripts.research.representation_v2.transformer.training import fit_head,predict_head
    seq=[sequence('s0','a',0.),sequence('s1','a',1.),sequence('s2','b',2.),sequence('s3','b',3.)]
    fits=[fit_head(a,seq,np.array([0,1,0,1]),np.array(['a','a','b','b']),seed=1,device='cpu',steps=2)
          for a in ('p','c','t')]
    assert len({f['sampling_sha256'] for f in fits})==1
    assert all(f['steps']==2 and f['clip_norm']==1. for f in fits)
    for fit in fits:
        assert np.isfinite(predict_head(fit,seq)).all()
    with pytest.raises(ValueError,match='binary'):
        fit_head('p',seq,np.zeros(4),np.array(['a','a','b','b']),seed=1,device='cpu',steps=2)
