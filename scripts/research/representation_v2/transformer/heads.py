"""Paired regional pooling, TCN and Transformer candidate readouts."""
import math

import torch
from torch import nn


class CandidateHead(nn.Module):
    def __init__(self,arm):
        super().__init__()
        if arm not in ('p','c','t'): raise ValueError('unknown temporal arm')
        self.arm=arm
        self.stem=nn.Sequential(nn.Linear(17,64),nn.LayerNorm(64),nn.GELU())
        if arm=='c':
            self.temporal=nn.ModuleList(nn.Conv1d(64,64,3,padding=d,dilation=d) for d in (1,2))
            self.dropout=nn.Dropout(.1)
        elif arm=='t':
            layer=nn.TransformerEncoderLayer(64,4,128,.1,batch_first=True,norm_first=True,
                                             activation='gelu')
            self.temporal=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.head=nn.Linear(387,1)

    def forward(self,batch):
        x,mask=batch['x'],batch['mask']
        n=batch['count']; device=self.head.weight.device
        sums=torch.zeros((n*3,64),device=device)
        squares=torch.zeros_like(sums); counts=torch.zeros((n*3,1),device=device)
        selected=torch.nonzero(mask.any(1),as_tuple=False).flatten()
        if len(selected):
            m=mask[selected]; h=self.stem(x[selected])
            seconds=batch['times'][selected]/300.
            frequency=torch.exp(torch.arange(0,64,2,device=device,dtype=h.dtype)*(-math.log(10000.)/64))
            phase=seconds[:,:,None]*frequency
            position=torch.stack((torch.sin(phase),torch.cos(phase)),dim=-1).flatten(-2)
            h=(h+position)*m[:,:,None]
            if self.arm=='c':
                for convolution in self.temporal:
                    h=(h+self.dropout(torch.nn.functional.gelu(convolution(h.transpose(1,2)).transpose(1,2))))*m[:,:,None]
            elif self.arm=='t':
                h=self.temporal(h,src_key_padding_mask=~m)*m[:,:,None]
            owners=batch['owner'][selected]*3+batch['region'][selected]
            sums=sums.index_add(0,owners,h.sum(1))
            squares=squares.index_add(0,owners,h.square().sum(1))
            counts=counts.index_add(0,owners,m.sum(1,keepdim=True).to(h.dtype))
        mean=sums/counts.clamp_min(1)
        variance=(squares/counts.clamp_min(1)-mean.square()).clamp_min(0)
        # sqrt(0) has an infinite derivative: preserve exact zero with a safe branch.
        std=torch.where(variance>0,variance.clamp_min(1e-12).sqrt(),torch.zeros_like(variance))
        features=torch.cat((torch.cat((mean,std),1).reshape(n,384),batch['coverage']),1)
        return self.head(features).squeeze(-1)


def build_head(arm):
    return CandidateHead(arm)
