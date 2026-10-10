from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from scripts.research.representation_v2 import adapt_probe as ap
from scripts.research.representation_v2.dataset import TokenBag
from scripts.research.representation_v2.low_capacity import pool_ordered
from scripts.research.representation_v2.models import MotionBranch


def bag(seed=1):
    rng=np.random.default_rng(seed)
    return TokenBag(('s',0,20000),rng.normal(size=(4,6,125)).astype(np.float32),np.ones((4,125),bool),
                    np.array([1,1,0,1],bool),np.arange(4)*5000+2500,0,20000,np.ones(4,np.float32))


def test_differentiable_pool_matches_s1_and_excludes_invalid_tokens():
    b=bag();z=torch.arange(64,dtype=torch.float32).reshape(4,16).requires_grad_()
    result=ap.pool_tensor(z,b)
    np.testing.assert_allclose(result.detach().numpy(),pool_ordered(z.detach().numpy(),b),atol=1e-6)
    result.sum().backward()
    assert torch.isfinite(z.grad).all() and not z.grad[2].any() and z.grad[0].any()


def test_joint_ordered_features_have_explicit_missing_latent_not_missing_quality():
    b=replace(bag(),token_mask=np.zeros(4,bool))
    x=ap.pool_joint_ordered(np.zeros((4,16)),b)
    assert x.shape==(135,) and np.isnan(x[:128]).all() and np.isfinite(x[128:]).all()
    assert x[-2:].tolist()==[0.,0.]
    with pytest.raises(ValueError):ap.ordered_joint_features(np.zeros((1,116)),np.zeros((1,38)))


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))])
def test_adaptation_updates_only_copied_encoder_and_target_head(device):
    torch.set_num_threads(2)
    e=MotionBranch().token_encoder;h=torch.nn.Linear(16,1)
    original=deepcopy(e.state_dict());source_head=deepcopy(h.state_dict())
    bs=tuple(bag(i) for i in range(8));y=np.array([0,1]*4);g=np.array(['a']*4+['b']*4)
    rng=np.random.default_rng(3)
    bank={'values':rng.normal(size=(8,6,125)).astype(np.float32),'labels':y,'groups':np.array([0]*4+[1]*4)}
    trained,report=ap.adapt_encoder(e,h,bs,y,g,bank,train_subjects={'a','b'},seed=2,device=device,steps=2)
    assert report['steps']==2 and report['fit_subjects']==['a','b']
    assert any(not torch.equal(v.cpu(),original[k]) for k,v in trained.state_dict().items())
    for k,v in e.state_dict().items():torch.testing.assert_close(v,original[k],atol=0,rtol=0)
    for k,v in h.state_dict().items():torch.testing.assert_close(v,source_head[k],atol=0,rtol=0)
    assert all(not p.requires_grad for p in trained.parameters())
    with pytest.raises(ValueError,match='subject'):
        ap.adapt_encoder(e,h,bs,y,g,bank,train_subjects={'a'},seed=2,device='cpu',steps=1)


def test_real_ordered_verifier_keeps_width_and_handles_missing_latent():
    rng=np.random.default_rng(7);x=rng.normal(size=(80,251));y=(x[:,0]>0).astype(int)
    x[:8,116:244]=np.nan
    p=ap.ordered_joint_probabilities(x,y,x[:4],seed=3)
    assert p.shape==(4,) and np.isfinite(p).all()
    with pytest.raises(ValueError):ap.ordered_joint_probabilities(x[:,:154],y,x[:4],seed=3)


def test_engine_variant_sets_work_budget_and_rejects_unsafe_output():
    default=ap.jp.experiment_contract()
    assert default['max_wall_seconds']==10800 and default['neural_optimizer_steps']==0
    variant=ap.jp.experiment_contract({'output':'adapt','label':'S8','max_wall_seconds':21600,
                                     'max_work_seconds':10800,'neural_optimizer_steps':12000})
    assert variant['max_wall_seconds']==21600 and variant['neural_optimizer_steps']==12000
    with pytest.raises(ValueError):ap.jp.experiment_contract({'output':'../release'})
    with pytest.raises(ValueError):ap.jp.experiment_contract({'max_work_seconds':-1})
