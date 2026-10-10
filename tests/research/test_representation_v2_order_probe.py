from dataclasses import replace

import numpy as np
import pytest
import torch

from scripts.research.representation_v2.dataset import TokenBag
from scripts.research.representation_v2 import order_probe as probe


def bag(n=4):
    return TokenBag(('s', 0, max(n, 1)*5000), np.zeros((n,6,125),np.float32),
                    np.ones((n,125),bool), np.ones(n,bool),
                    np.arange(n)*5000+2500,0,max(n,1)*5000,np.ones(n,np.float32))


def test_shuffle_preserves_invalid_positions_and_vector_multiset():
    b=replace(bag(8),token_mask=np.array([1,1,0,1,1,0,1,1],bool))
    z=np.arange(128,dtype=np.float32).reshape(8,16)
    shuffled=probe.shuffle_tokens(z,b)
    np.testing.assert_array_equal(shuffled[~b.token_mask],z[~b.token_mask])
    assert sorted(map(tuple,shuffled[b.token_mask]))==sorted(map(tuple,z[b.token_mask]))
    assert not np.array_equal(shuffled,z)
    np.testing.assert_array_equal(shuffled,probe.shuffle_tokens(z,b))
    np.testing.assert_array_equal(b.token_center_ms,np.arange(8)*5000+2500)


def test_normalizer_is_subject_candidate_token_balanced_not_length_weighted():
    bags=(bag(1),bag(3),bag(1))
    zs=[np.full((1,16),0.),np.full((3,16),2.),np.full((1,16),5.)]
    center,scale=probe.fit_moments(zs,bags,np.array(['a','a','b']))
    np.testing.assert_allclose(center,3.)
    np.testing.assert_allclose(scale,np.sqrt(4.5))


def test_bins_keep_missing_geometry_and_handle_zero_tokens():
    b=replace(bag(),token_mask=np.array([1,0,0,1],bool))
    x,m=probe.bin_tokens(np.ones((4,16)),b,np.zeros(16),np.ones(16))
    assert x.shape==(64,36) and m.sum()==2
    assert m[8] and m[56] and not m[24]
    assert np.all(x[~m]==0)
    e,em=probe.bin_tokens(np.zeros((0,16)),bag(0),np.zeros(16),np.ones(16))
    assert not em.any() and not e.any()


def test_readout_masks_invalid_bins_and_can_distinguish_order():
    torch.manual_seed(1)
    model=probe.TemporalReadout()
    x=torch.randn(2,64,36); mask=torch.zeros(2,64,dtype=torch.bool);mask[:,:6]=True
    y=model(x,mask)
    poison=x.clone();poison[:,6:]=1000
    torch.testing.assert_close(y,model(poison,mask))
    reverse=x.clone();reverse[:,:6]=x[:,:6].flip(1)
    assert not torch.allclose(y,model(reverse,mask),atol=1e-7)
    assert torch.isfinite(model(x,torch.zeros_like(mask))).all()


def test_training_is_reproducible_and_does_not_modify_features():
    rng=np.random.default_rng(2);x=rng.normal(size=(12,64,36)).astype(np.float32)
    mask=np.ones((12,64),bool);y=np.arange(12)%2;g=np.array(['a']*6+['b']*6)
    before=x.copy()
    a,h=probe.fit_temporal(x,mask,y,g,seed=2,device='cpu',steps=3)
    b,_=probe.fit_temporal(x,mask,y,g,seed=2,device='cpu',steps=3)
    assert h[-1]['step']==3
    for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
    np.testing.assert_array_equal(x,before)


def test_encoder_freeze_is_enforced_and_invalid_tokens_are_not_encoded():
    from scripts.research.representation_v2.models import MotionBranch
    encoder=MotionBranch().token_encoder
    with pytest.raises(ValueError,match='frozen'):
        probe.encode_tokens(encoder,[bag()],device='cpu')
    for p in encoder.parameters():p.requires_grad_(False)
    before={k:v.clone() for k,v in encoder.state_dict().items()}
    b=replace(bag(),token_mask=np.array([1,0,1,0],bool))
    z=probe.encode_tokens(encoder,[b,bag(0)],device='cpu')
    assert z[0].shape==(4,16) and np.all(z[0][[1,3]]==0) and z[1].shape==(0,16)
    for k,v in encoder.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_readout_cpu_cuda_parity():
    torch.manual_seed(4);a=probe.TemporalReadout().eval();b=probe.TemporalReadout().cuda().eval()
    b.load_state_dict(a.state_dict());x=torch.randn(3,64,36);m=torch.rand(3,64)>.3
    torch.testing.assert_close(a(x,m),b(x.cuda(),m.cuda()).cpu(),rtol=1e-4,atol=1e-5)


def test_gate_requires_order_ablation_and_convergence():
    def row():
        return {'metrics':{a:{'all':{'average_precision':ap,'raw_logit_bce':.3}}
                           for a,ap in [('l',.5),('r',.4),('f',.55),('fs',.52),('prior',.1)]},
                'checks':{k:True for k in ('provenance','finite','frozen','optimizer','resource')}}
    rows=[row() for _ in range(3)]
    assert probe.order_gate(rows)['passed']
    rows[0]['checks']['optimizer']=False
    assert not probe.order_gate(rows)['passed']
    rows=[row() for _ in range(3)]
    for r in rows:r['metrics']['fs']['all']['average_precision']=.55
    assert not probe.order_gate(rows)['passed']
