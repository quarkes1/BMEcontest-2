import numpy as np
import pytest
from scripts.research.representation_v2.dataset import TokenBag
from scripts.research.representation_v2.joint_probe import pool_joint, joint_features
from scripts.research.representation_v2.joint_probe import encode_joint, joint_probabilities
import torch
from copy import deepcopy
import json
from scripts.research.representation_v2 import joint_probe as jp


def bag(valid):
    n=len(valid)
    return TokenBag(('s',0,20000),np.zeros((n,6,125),np.float32),
                    np.ones((n,125),bool),np.array(valid,bool),
                    np.arange(n)*5000+2500,0,20000,np.ones(4))


def test_joint_pool_excludes_invalid_and_has_fixed_order():
    z=np.repeat(np.array([1,999,3,5])[:,None],16,axis=1)
    x=pool_joint(z,bag([1,0,1,1]))
    assert x.shape==(38,)
    np.testing.assert_allclose(x[:16],3)
    np.testing.assert_allclose(x[16:32],np.std([1,3,5]))
    np.testing.assert_array_equal(x[32:36],np.ones(4))
    np.testing.assert_array_equal(x[-2:],[.75,1])


def test_joint_invalid_is_missing_not_synthetic_embedding():
    x=pool_joint(np.zeros((4,16)),bag([0,0,0,0]))
    assert np.isnan(x[:32]).all()
    assert np.isfinite(x[32:]).all()
    np.testing.assert_array_equal(x[-2:],[0,0])
    one=pool_joint(np.ones((1,16)),bag([1]))
    np.testing.assert_array_equal(one[16:32],np.zeros(16))


def test_joint_width_and_inf_rejected_without_mutating_base():
    base=np.zeros((2,116)); original=base.copy()
    motion=np.zeros((2,38)); motion[0,:32]=np.nan
    x=joint_features(base,motion)
    assert x.shape==(2,154)
    np.testing.assert_array_equal(base,original)
    for b,m in [(base[:,:115],motion),(base,motion[:1]),(base,np.full((2,38),np.inf))]:
        with pytest.raises(ValueError):joint_features(b,m)


def test_empty_tokens_produce_missing_moments_and_zero_quality():
    x=pool_joint(np.zeros((0,16)),bag([]))
    assert np.isnan(x[:32]).all()
    np.testing.assert_array_equal(x[32:],np.zeros(6))


def test_encoder_is_frozen_and_only_sees_valid_tokens():
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight=torch.nn.Parameter(torch.ones(16)); self.seen=0
        def forward(self,x,mask):
            assert not torch.is_grad_enabled()
            self.seen+=len(x)
            return self.weight[None].expand(len(x),-1)
    encoder=Encoder()
    with pytest.raises(ValueError,match='frozen'):encode_joint(encoder,[bag([1])])
    encoder.requires_grad_(False)
    before=encoder.weight.detach().clone()
    result=encode_joint(encoder,[bag([1,0,1,0]),bag([0,0,0,0])],chunk=1)
    assert encoder.seen==2 and result.shape==(2,38)
    torch.testing.assert_close(encoder.weight,before)
    assert np.isnan(result[1,:32]).all()
    assert encode_joint(encoder,[]).shape==(0,38)


def test_real_joint_verifier_handles_all_missing_motion_columns():
    rng=np.random.default_rng(42)
    x=rng.normal(size=(100,154)); y=(x[:,0]>0).astype(int)
    x[:,116:148]=np.nan
    v=x[:4].copy(); v[:,116:148]=5
    p=joint_probabilities(x,y,v,seed=1)
    assert p.shape==(4,) and np.isfinite(p).all()
    with pytest.raises(ValueError):joint_probabilities(x[:,:116],y,v,seed=1)


@pytest.mark.parametrize('key',['candidate_metrics','subject_events'])
def test_baseline_replay_rejects_changed_diagnostics(monkeypatch,key):
    old={'events':{'b':{'n_tp':1}},'predictions':{'b':[]},
         'candidate_metrics':{'b':{'ap':.8}},'subject_events':[{'subject':'s','events':{'b':{'n_tp':1}}}]}
    monkeypatch.setattr(jp.subprocess,'check_output',lambda *a,**kw:json.dumps(old).encode())
    changed=deepcopy(old)
    if key=='candidate_metrics':changed[key]['b']['ap']=.9
    else:changed[key][0]['events']['b']['n_tp']=2
    with pytest.raises(RuntimeError):jp.baseline_replay('.', 'fold0_report.json',changed)
