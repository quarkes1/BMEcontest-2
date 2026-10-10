"""S14 frozen LIMU-BERT-X source-only unit-free transfer research.

Encoder equations adapted from WANDS-HKUST/LIMU-BERT_Experience (MIT),
Copyright (c) 2025 WANDS HKUST. Full upstream notice retained with experiment.
No remote code is imported or executed. This is NOT physical-unit calibration.
"""
from pathlib import Path
import hashlib
import math

import numpy as np
from scipy.signal import resample_poly
import torch
from torch import nn

from scripts.research.representation_v2 import unitfree as u

UPSTREAM_COMMIT='bdfc3d0f734befaeb9375f5f7b8a7342f0ddfdcb'
WEIGHT_SHA='65697e7017020f36429a221382253f1fc370fdfd493cfb71667f95846c18516c'
WEIGHT_BYTES=226783
OUTPUT='outputs/challenger_70/representation_v2/amendment_a4_foundation_source'


class LayerNorm(nn.Module):
    def __init__(self):
        super().__init__();self.gamma=nn.Parameter(torch.ones(72));self.beta=nn.Parameter(torch.zeros(72))

    def forward(self,x):
        centered=x-x.mean(-1,keepdim=True)
        return self.gamma*centered/torch.sqrt(centered.square().mean(-1,keepdim=True)+1e-12)+self.beta


class Embeddings(nn.Module):
    def __init__(self):
        super().__init__();self.lin=nn.Linear(6,72);self.pos_embed=nn.Embedding(20,72);self.norm=LayerNorm()

    def forward(self,x):
        return self.norm(self.norm(self.lin(x))+self.pos_embed(torch.arange(20,device=x.device)))


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_q=nn.Linear(72,72);self.proj_k=nn.Linear(72,72);self.proj_v=nn.Linear(72,72)

    def forward(self,x):
        q,k,v=[layer(x).reshape(len(x),20,4,18).transpose(1,2) for layer in
               (self.proj_q,self.proj_k,self.proj_v)]
        return ((q@k.transpose(-2,-1)/math.sqrt(18)).softmax(-1)@v).transpose(1,2).reshape(len(x),20,72)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__();self.fc1=nn.Linear(72,144);self.fc2=nn.Linear(144,72)

    def forward(self,x):
        h=self.fc1(x);return self.fc2(h*.5*(1+torch.erf(h/math.sqrt(2))))


class LimuEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.embed=Embeddings();self.attn=Attention();self.proj=nn.Linear(72,72)
        self.norm1=LayerNorm();self.pwff=FeedForward();self.norm2=LayerNorm()

    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(20,6) or not torch.isfinite(x).all():
            raise ValueError('LIMU input requires finite N x20 x6')
        h=self.embed(x)
        for _ in range(4):
            h=self.attn(h)
            h=self.norm1(h+self.proj(h))
            h=self.norm2(h+self.pwff(h))
        return h


def head_template():
    """Known upstream reconstruction tensors; validated, never used for classification."""
    return {k:torch.zeros(shape) for k,shape in {
        'fc.weight':(72,72),'fc.bias':(72,),'linear.weight':(72,72),'linear.bias':(72,),
        'norm.gamma':(72,),'norm.beta':(72,),'decoder.weight':(6,72),'decoder.bias':(6,)}.items()}


def load_pretrained(path,*,expected_sha=WEIGHT_SHA):
    path=Path(path)
    if path.stat().st_size>1024**2 or u._file_sha(path)!=expected_sha:
        raise ValueError('external checkpoint size/SHA mismatch')
    state=torch.load(path,map_location='cpu',weights_only=True)
    with torch.random.fork_rng(devices=[]):model=LimuEncoder()
    expected={'transformer.'+k:v for k,v in model.state_dict().items()};expected.update(head_template())
    if not isinstance(state,dict) or set(state)!=set(expected):raise ValueError('external state keys mismatch')
    for key,template in expected.items():
        value=state[key]
        if (not isinstance(value,torch.Tensor) or value.shape!=template.shape or value.dtype!=torch.float32
                or not torch.isfinite(value).all()):raise ValueError('invalid external tensor')
    model.load_state_dict({k:state['transformer.'+k] for k in model.state_dict()},strict=True)
    return model.eval().requires_grad_(False)


def bridge_tokens(values):
    x=np.asarray(values)
    if x.ndim!=3 or x.shape[1:]!=(6,125) or not np.isfinite(x).all():
        raise ValueError('finite unit-free N x6 x125 required')
    return resample_poly(x,4,5,axis=2,padtype='line').transpose(0,2,1).reshape(len(x),5,20,6).astype(np.float32)


def foundation_gate(metrics,checks):
    passed=all(checks.get(k) is True for k in ('finite','isolation','frozen','resource','protected'))
    try:
        r,d,f=[metrics[k] for k in ('r','d','f')]
        for row in (r,d,f):
            ap,bce,prior=[row[k] for k in ('average_precision','raw_logit_bce','prevalence')]
            passed &= bool(np.isfinite([ap,bce,prior]).all() and 0<=ap<=1 and bce>=0 and 0<=prior<=1)
        passed &= (f['average_precision']>=max(r['average_precision']+.01,d['average_precision']+.01,
                      .8440475046760654+.01,f['prevalence']+.1)-1e-12
                   and f['raw_logit_bce']<=min(r['raw_logit_bce']+.02,d['raw_logit_bce']+.02,
                                             .25704393608246073+.02)+1e-12)
    except (KeyError,TypeError,ValueError):passed=False
    return {'passed':bool(passed),'status':'S14_SUPPORTS_TARGET_PROBE' if passed else
            'S14_NO_SUPPORTED_FOUNDATION_TRANSFER'}
