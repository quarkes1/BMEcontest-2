"""Protect exact upstream semantics and source-only fitted statistics."""
from pathlib import Path
import math
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from scripts.research.representation_v2 import foundation_source as m


def test_limu_matches_independent_functional_upstream_graph():
    torch.manual_seed(5)
    model=m.LimuEncoder().eval(); state=model.state_dict(); x=torch.randn(3,20,6)
    def linear(x,name):return F.linear(x,state[name+'.weight'],state[name+'.bias'])
    def norm(x,name):
        return state[name+'.gamma']*(x-x.mean(-1,keepdim=True))/torch.sqrt(
            ((x-x.mean(-1,keepdim=True))**2).mean(-1,keepdim=True)+1e-12)+state[name+'.beta']
    h=norm(norm(linear(x,'embed.lin'),'embed.norm')+state['embed.pos_embed.weight'],'embed.norm')
    for _ in range(4):
        q,k,v=[linear(h,'attn.proj_'+s).reshape(3,20,4,18).transpose(1,2) for s in 'qkv']
        h=((q@k.transpose(-2,-1)/math.sqrt(18)).softmax(-1)@v).transpose(1,2).reshape(3,20,72)
        h=norm(h+linear(h,'proj'),'norm1')
        z=linear(h,'pwff.fc1'); z=z*.5*(1+torch.erf(z/math.sqrt(2)))
        h=norm(h+linear(z,'pwff.fc2'),'norm2')
    torch.testing.assert_close(model(x),h,rtol=1e-5,atol=1e-6)
    assert h.shape==(3,20,72)


@pytest.mark.parametrize('shape',[(2,19,6),(2,20,7)])
def test_encoder_rejects_incompatible_input(shape):
    with pytest.raises(ValueError):m.LimuEncoder()(torch.zeros(shape))


def test_bridge_keeps_five_seconds_and_triad_order():
    from scipy.signal import resample_poly
    x=np.stack([np.tile(np.arange(125,dtype=np.float32),(6,1))])
    result=m.bridge_tokens(x)
    assert result.shape==(1,5,20,6)
    expected=resample_poly(x,4,5,axis=2,padtype='line').transpose(0,2,1)
    np.testing.assert_allclose(result.reshape(1,100,6),expected)
    np.testing.assert_allclose(result[0,:,0,0],[0,25,50,75,100],atol=.1)


def test_bridge_rejects_nan_and_wrong_width():
    with pytest.raises(ValueError):m.bridge_tokens(np.zeros((2,6,124)))
    with pytest.raises(ValueError):m.bridge_tokens(np.full((2,6,125),np.nan))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cpu_cuda_encoder_outputs_close():
    torch.manual_seed(7); model=m.LimuEncoder().eval(); x=torch.randn(2,20,6)
    expected=model(x).detach(); actual=model.cuda()(x.cuda()).detach().cpu()
    torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)


def test_strict_checkpoint_loader_roundtrip_and_unchanged_state(tmp_path):
    model=m.LimuEncoder()
    state={'transformer.'+k:v.clone() for k,v in model.state_dict().items()}
    state.update(m.head_template())
    path=tmp_path/'model.pt';torch.save(state,path)
    loaded=m.load_pretrained(path,expected_sha=m.u._file_sha(path))
    for k,v in model.state_dict().items():torch.testing.assert_close(v,loaded.state_dict()[k])
    assert all(not p.requires_grad for p in loaded.parameters())
    with pytest.raises(ValueError):m.load_pretrained(path,expected_sha='0'*64)


@pytest.mark.parametrize('damage',['key','shape','dtype','nan'])
def test_strict_loader_rejects_malformed_checkpoint(tmp_path,damage):
    state={'transformer.'+k:v.clone() for k,v in m.LimuEncoder().state_dict().items()}
    state.update(m.head_template()); key='transformer.embed.lin.weight'
    if damage=='key':state['extra']=torch.zeros(1)
    if damage=='shape':state[key]=torch.zeros(1)
    if damage=='dtype':state[key]=state[key].double()
    if damage=='nan':state[key][0,0]=float('nan')
    path=tmp_path/'bad.pt';torch.save(state,path)
    with pytest.raises(ValueError):m.load_pretrained(path,expected_sha=m.u._file_sha(path))


def metrics():
    return {'r':{'average_precision':.83,'raw_logit_bce':.26,'prevalence':.21},
            'd':{'average_precision':.84,'raw_logit_bce':.25,'prevalence':.21},
            'f':{'average_precision':.86,'raw_logit_bce':.26,'prevalence':.21}}


def checks():return dict.fromkeys(('finite','isolation','frozen','resource','protected'),True)


def test_gate_accepts_registered_source_improvement():
    assert m.foundation_gate(metrics(),checks())['passed']


@pytest.mark.parametrize('arm,key,value',[('f','average_precision',.849),('f','raw_logit_bce',.28),
                                         ('f','average_precision',float('nan'))])
def test_gate_rejects_bad_improvement(arm,key,value):
    rows=metrics();rows[arm][key]=value
    assert not m.foundation_gate(rows,checks())['passed']


def test_gate_requires_complete_integrity_checks():
    c=checks();del c['isolation']
    assert not m.foundation_gate(metrics(),c)['passed']
