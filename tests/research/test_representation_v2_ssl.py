import numpy as np
import torch

from scripts.research.representation_v2.ssl import rotate_yaw, vicreg_loss, build_token_bank
from scripts.research.representation_v2.dataset import TokenBag


def test_yaw_uses_same_rotation_for_acc_and_gyro_preserves_z():
    x = torch.tensor([[[1.], [0.], [3.], [2.], [0.], [6.]]])
    y = rotate_yaw(x, torch.tensor([torch.pi/2]))
    torch.testing.assert_close(y[0, :3, 0], torch.tensor([0., 1., 3.]), atol=1e-6, rtol=0)
    torch.testing.assert_close(y[:, 3:], 2*y[:, :3])


def test_vicreg_penalizes_collapse_and_has_finite_gradient():
    collapsed = torch.zeros(8, 4, requires_grad=True)
    loss, parts = vicreg_loss(collapsed, collapsed)
    assert parts['variance'] > .9 and parts['invariance'] == 0
    loss.backward()
    assert torch.isfinite(collapsed.grad).all()
    # Orthogonal centered balanced columns have unit variance and no covariance.
    z = torch.tensor([[1,1], [1,-1], [-1,1], [-1,-1]], dtype=torch.float32)
    good, _ = vicreg_loss(z, z)
    assert good < loss


def test_bank_deduplicates_same_session_time_and_excludes_invalid():
    def bag(start, centers, valid):
        return TokenBag(('s', start, 15000), np.ones((len(valid),6,125),np.float32)*start,
                        np.ones((len(valid),125),bool), np.array(valid,bool), np.array(centers),
                        start,15000,np.ones(len(valid)))
    bank = build_token_bank([bag(5000,[7500,12500],[1,0]),bag(0,[2500,7500],[1,1])], ['a','a'], seed=1)
    assert len(bank['values']) == 2
    assert np.all(bank['values'] == 0)  # lexicographically first candidate supplies duplicate.
    assert set(bank['subjects']) == {'a'}


def test_ssl_smoke_preserves_initial_encoder_and_completes_fixed_steps():
    from scripts.research.representation_v2.models import MotionBranch
    from scripts.research.representation_v2.ssl import fit_ssl
    torch.set_num_threads(2)
    encoder = MotionBranch().token_encoder
    original = {k:v.clone() for k,v in encoder.state_dict().items()}
    bank = {'values':np.random.default_rng(3).normal(size=(16,6,125)).astype(np.float32),
            'masks':np.ones((16,125),bool),'subjects':np.array(['a','b']*8)}
    result = fit_ssl(encoder, bank, np.zeros(6,np.float32), np.ones(6,np.float32), steps=2,device='cpu')
    assert result['steps']==2 and result['history'][-1]['step']==2
    assert np.isfinite(result['history'][-1]['loss'])
    assert all(torch.equal(original[k],v) for k,v in encoder.state_dict().items())
    assert any(not torch.equal(original[k],v) for k,v in result['encoder'].state_dict().items())
