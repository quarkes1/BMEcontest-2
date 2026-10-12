"""All temporal heads share safe masking, region pooling and physical positions."""
import numpy as np
import pytest
import torch


def batch(empty=False):
    torch.manual_seed(4)
    return dict(x=torch.randn(2,7,17),times=torch.arange(7)[None,:].repeat(2,1)*5.,
                mask=torch.zeros(2,7,dtype=torch.bool) if empty else torch.ones(2,7,dtype=torch.bool),
                owner=torch.tensor([0,0]),region=torch.tensor([0,1]),
                coverage=torch.tensor([[0.,0.,0.]]) if empty else torch.tensor([[1.,1.,0.]]),
                count=1)


@pytest.mark.parametrize('arm',['p','c','t'])
def test_heads_ignore_padding_and_all_empty_is_finite(arm):
    from scripts.research.representation_v2.transformer.heads import build_head
    torch.set_num_threads(2)
    model=build_head(arm).eval(); original=batch()
    expected=model(original)
    padded={**original,'x':torch.cat([original['x'],torch.randn(2,3,17)*100],1),
            'times':torch.cat([original['times'],torch.zeros(2,3)],1),
            'mask':torch.cat([original['mask'],torch.zeros(2,3,dtype=torch.bool)],1)}
    torch.testing.assert_close(model(padded),expected,atol=1e-5,rtol=1e-4)
    assert torch.isfinite(model(batch(True))).all()
    model.train(); loss=model(original).sum(); loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_transformer_cpu_cuda_scores_agree_off_boundary():
    from scripts.research.representation_v2.transformer.heads import build_head
    torch.set_num_threads(2); model=build_head('t').eval(); data=batch()
    with torch.inference_mode():
        cpu=torch.sigmoid(model(data))
        gpu=torch.sigmoid(model.cuda()({k:v.cuda() if torch.is_tensor(v) else v for k,v in data.items()})).cpu()
    torch.testing.assert_close(cpu,gpu,atol=1e-5,rtol=1e-4)
    stable=(cpu-.5).abs()>1e-5
    assert torch.equal(cpu[stable]>=.5,gpu[stable]>=.5)
