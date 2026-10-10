from copy import deepcopy

import numpy as np
import pytest
import torch

from scripts.research.representation_v2 import optimization_audit as oa
from scripts.research.representation_v2.models import MotionBranch
from scripts.research.representation_v2.dataset import TokenBag


def bag(seed):
    rng=np.random.default_rng(seed)
    return TokenBag(('s',0,20000),rng.normal(size=(4,6,125)).astype(np.float32),np.ones((4,125),bool),
                    np.ones(4,bool),np.arange(4)*5000+2500,0,20000,np.ones(4,np.float32))


def test_saved_readout_applies_training_scaler_and_intercept():
    payload={'mean':[1.,2.],'scale':[2.,4.],'coefficient':[[3.,-2.]],'intercept':[.5]}
    np.testing.assert_allclose(oa.saved_logits(np.array([[3.,6.],[1.,2.]]),payload),[1.5,.5])
    with pytest.raises(ValueError):oa.saved_logits(np.zeros((1,3)),payload)


def test_gradient_direction_distinguishes_conflict_and_zero_norm():
    assert oa.gradient_relation(torch.tensor([1.,0.]),torch.tensor([-2.,0.]))=={'target_norm':1.,'source_norm':2.,'cosine':-1.,'weighted_source_to_target':.5}
    assert oa.gradient_relation(torch.zeros(2),torch.ones(2))['cosine'] is None


def test_target_gradient_reaches_encoder_without_weight_update():
    torch.set_num_threads(2);torch.manual_seed(3)
    encoder=MotionBranch().token_encoder;head=torch.nn.Linear(133,1)
    before=deepcopy(encoder.state_dict())
    payload={'mean':[0.]*133,'scale':[1.]*133}
    gradient=oa.target_gradient(encoder,head,[bag(1),bag(2)],[0,1],payload,'cpu')
    assert torch.isfinite(gradient).all() and gradient.norm()>0
    for key,value in encoder.state_dict().items():torch.testing.assert_close(value,before[key],atol=0,rtol=0)
