"""Gap-safe full-timeline feature TCN and weak meal-bag supervision."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.pipeline.event_stack import EventRef

BODY=256
HALO=31


@dataclass(frozen=True)
class Span:
    sid: str
    subject: str
    windows: tuple
    features: np.ndarray


@dataclass(frozen=True)
class Bag:
    subject: str
    label: int
    refs: np.ndarray
    interval: tuple


def build_spans(features,windows,mapping):
    x=np.asarray(features,dtype=np.float32)
    if x.shape!=(len(windows),47) or np.isinf(x).any(): raise ValueError('invalid micro features')
    keys=[(w.sid,int(w.start_ms),int(w.end_ms)) for w in windows]
    if len(set(keys))!=len(keys): raise ValueError('duplicate micro windows')
    if any(s not in mapping or b-a!=15000 for s,a,b in keys): raise ValueError('unknown session or window width')
    order=sorted(range(len(keys)),key=keys.__getitem__);result=[];block=[]
    def finish():
        if block:
            result.append(Span(windows[block[0]].sid,mapping[windows[block[0]].sid],
                tuple(windows[i] for i in block),x[np.asarray(block)].copy()))
            block.clear()
    for i in order:
        if block and (windows[i].sid!=windows[block[-1]].sid or windows[i].start_ms-windows[block[-1]].start_ms!=7500): finish()
        block.append(i)
    finish();return result


def _selected(spans,subjects):
    subjects=set(subjects)
    if not subjects or subjects-{s.subject for s in spans}: raise ValueError('unknown or empty fit subjects')
    return [s for s in spans if s.subject in subjects]


def fit_normalizer(spans,subjects):
    selected=_selected(spans,subjects);rows=[]
    for subject in sorted(subjects):
        own=[s for s in selected if s.subject==subject]
        x=np.concatenate([s.features for s in own])
        keys=[(w.start_ms,s.sid) for s in own for w in s.windows]
        x=x[np.asarray(sorted(range(len(keys)),key=keys.__getitem__))]
        rows.append(x[np.linspace(0,len(x)-1,min(4096,len(x)),dtype=np.int64)])
    x=np.concatenate(rows).astype(np.float64);present=np.isfinite(x).any(0)
    center=np.zeros(47);center[present]=np.nanmedian(x[:,present],axis=0)
    filled=np.where(np.isnan(x),center,x)
    scale=np.std(filled,axis=0).clip(1e-4);scale[~present]=1
    return center.astype(np.float32),scale.astype(np.float32)


def make_bags(spans,truths,subjects):
    subjects=set(subjects);_selected(spans,subjects)
    targets=sorted((s,int(a),int(b)) for s,a,b in truths if s in subjects)
    if not targets or any(b<=a for _,a,b in targets): raise ValueError('valid fit meal intervals required')
    bags=[];missing=[];centers=[np.asarray([(w.start_ms+w.end_ms)//2 for w in s.windows],np.int64) for s in spans]
    for subject,left,right in targets:
        refs=[(i,int(j)) for i,s in enumerate(spans) if s.subject==subject
              for j in np.flatnonzero((centers[i]>=left)&(centers[i]<right))]
        if len(refs)>=2: bags.append(Bag(subject,1,np.asarray(refs,np.int64),(left,right)))
        else: missing.append((subject,left,right,len(refs)))
    durations=[b-a for _,a,b in targets];position=0
    for i,span in enumerate(spans):
        if span.subject not in subjects: continue
        start,end=span.windows[0].start_ms,span.windows[-1].end_ms
        own=[(a,b) for s,a,b in targets if s==span.subject]
        for center in range(start+300000,end,300000):
            duration=durations[position%len(durations)];position+=1
            left=center-duration//2;right=left+duration
            if left<start or right>end: continue
            if not all(right<=a-300000 or left>=b+300000 for a,b in own): continue
            rows=np.flatnonzero((centers[i]>=left)&(centers[i]<right))
            if len(rows)>=2:
                bags.append(Bag(span.subject,0,np.column_stack((np.full(len(rows),i),rows)).astype(np.int64),(left,right)))
    identities=[(b.subject,b.label,b.interval,b.refs.tolist()) for b in bags]
    stats=dict(positive_truths=len(targets),positive_bags=sum(b.label==1 for b in bags),
        negative_bags=sum(b.label==0 for b in bags),untrainable_truths=missing,
        bag_identity_sha256=hashlib.sha256(json.dumps(identities,separators=(',',':')).encode()).hexdigest(),
        subjects={s:{str(y):sum(b.subject==s and b.label==y for b in bags) for y in (0,1)} for s in sorted(subjects)})
    return bags,stats


def pack_spans(spans,center,scale):
    center,scale=np.asarray(center),np.asarray(scale)
    if center.shape!=(47,) or scale.shape!=(47,) or not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale<=0):
        raise ValueError('invalid train-only normalizer')
    arrays=[];masks=[];records=[];offsets=[]
    for i,span in enumerate(spans):
        offsets.append(len(arrays));x=np.clip(np.nan_to_num((span.features-center)/scale,nan=0.),-8,8).astype(np.float32)
        for start in range(0,len(x),BODY):
            left=max(0,start-HALO);right=min(len(x),start+BODY+HALO)
            row=np.zeros((BODY+2*HALO,47),np.float32);mask=np.zeros(BODY+2*HALO,bool)
            target=HALO+left-start;row[target:target+right-left]=x[left:right];mask[target:target+right-left]=True
            arrays.append(row);masks.append(mask);records.append(dict(span=i,start=start,length=min(BODY,len(x)-start)))
    if not arrays: raise ValueError('no physical chunks')
    return dict(x=torch.from_numpy(np.stack(arrays)),mask=torch.from_numpy(np.stack(masks)),records=records,offsets=offsets)


class TemporalMIL(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Sequential(nn.Linear(47,64),nn.LayerNorm(64),nn.GELU())
        self.convolutions=nn.ModuleList(nn.Conv1d(64,64,3,padding=d,dilation=d) for d in (1,2,4,8,16))
        self.norms=nn.ModuleList(nn.LayerNorm(64) for _ in range(5))
        self.dropout=nn.Dropout(.1);self.head=nn.Linear(64,1)

    def forward(self,x,mask):
        h=self.stem(x)*mask[:,:,None]
        for conv,norm in zip(self.convolutions,self.norms):
            h=(h+self.dropout(F.gelu(norm(conv(h.transpose(1,2)).transpose(1,2)))))*mask[:,:,None]
        return self.head(h).squeeze(-1)*mask


def bag_logit(logits):
    if logits.ndim!=1 or not logits.numel(): raise ValueError('nonempty bag logits required')
    return torch.topk(logits,max(1,math.ceil(.2*len(logits)))).values.mean()


def fit_model(spans,truths,subjects,*,seed,device,steps=1000,callback=None):
    if steps<1 or device not in ('cpu','cuda'): raise ValueError('invalid fit budget/device')
    if callback is not None: callback()
    torch.set_num_threads(8);torch.manual_seed(seed)
    if device=='cuda':
        torch.cuda.manual_seed_all(seed);torch.cuda.reset_peak_memory_stats()
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    selected=_selected(spans,subjects);center,scale=fit_normalizer(selected,subjects)
    bags,stats=make_bags(selected,truths,subjects)
    by_class={y:{} for y in (0,1)}
    for i,bag in enumerate(bags): by_class[bag.label].setdefault(bag.subject,[]).append(i)
    if not all(by_class.values()): raise ValueError('both weak bag classes required')
    packed=pack_spans(selected,center,scale)
    x,mask=packed['x'].to(device),packed['mask'].to(device)
    refs=[]
    for bag in bags:
        span,row=bag.refs.T
        refs.append(np.column_stack((np.asarray(packed['offsets'])[span]+row//BODY,HALO+row%BODY)))
    model=TemporalMIL().to(device).train();optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    rng=np.random.default_rng(seed);owners={y:sorted(by_class[y]) for y in (0,1)};started=time.perf_counter();losses=[]
    for step in range(steps):
        chosen=[];labels=[]
        for label in (0,1):
            for _ in range(8):
                owner=owners[label][int(rng.integers(len(owners[label])))];pool=by_class[label][owner]
                chosen.append(int(pool[int(rng.integers(len(pool)))]));labels.append(label)
        chunks=np.unique(np.concatenate([refs[i][:,0] for i in chosen]));local={int(c):i for i,c in enumerate(chunks)}
        optimizer.zero_grad(set_to_none=True)
        logits=torch.cat([model(x[torch.as_tensor(ids,device=device)],mask[torch.as_tensor(ids,device=device)])
            for ids in np.array_split(chunks,math.ceil(len(chunks)/32))])
        values=[]
        for i in chosen:
            rows=torch.as_tensor([local[int(c)] for c in refs[i][:,0]],device=device)
            columns=torch.as_tensor(refs[i][:,1],device=device)
            values.append(bag_logit(logits[rows,columns]))
        loss=F.binary_cross_entropy_with_logits(torch.stack(values),torch.tensor(labels,dtype=torch.float32,device=device))
        if not torch.isfinite(loss): raise ValueError('nonfinite weak bag loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if callback is not None: callback()
        if (step+1)%250==0: print(f'dense-MIL seed{seed} step{step+1}/{steps} train_loss={np.mean(losses[-50:]):.6f}',flush=True)
    stats.update(steps=steps,seed=seed,train_subjects=sorted(subjects),parameters=sum(p.numel() for p in model.parameters()),
        chunks=len(x),last50_train_loss=float(np.mean(losses[-50:])),elapsed_seconds=time.perf_counter()-started,
        cuda_peak_bytes=int(torch.cuda.max_memory_allocated()) if device=='cuda' else None,
        normalizer_sha256=hashlib.sha256(center.tobytes()+scale.tobytes()).hexdigest())
    model.eval()
    return dict(model=model,center=center,scale=scale,fit_subjects=set(subjects),device=device),stats


def predict_model(fit,spans,subjects):
    if set(subjects)&fit['fit_subjects']: raise ValueError('scored subjects participated in encoder fit')
    selected=_selected(spans,subjects);packed=pack_spans(selected,fit['center'],fit['scale']);pieces=[]
    with torch.inference_mode():
        for left in range(0,len(packed['x']),128):
            p=fit['model'](packed['x'][left:left+128].to(fit['device']),packed['mask'][left:left+128].to(fit['device'])).sigmoid().cpu().numpy()
            pieces.extend(p)
    result=[[] for _ in selected]
    for p,row in zip(pieces,packed['records']): result[row['span']].extend(p[HALO:HALO+row['length']].tolist())
    return [(s,np.asarray(p,np.float32)) for s,p in zip(selected,result)]


def dense_proposals(span,probabilities):
    p=np.asarray(probabilities)
    if p.shape!=(len(span.windows),) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('invalid dense probabilities')
    hits=np.flatnonzero(p>=.5);result=[]
    if not len(hits): return result
    breaks=np.flatnonzero(np.diff(hits)*7500>120000)+1
    for run in np.split(hits,breaks):
        if len(run)<2: continue
        left=max(span.windows[0].start_ms,span.windows[int(run[0])].start_ms-60000)
        right=min(span.windows[-1].end_ms,span.windows[int(run[-1])].end_ms+60000)
        result.append((EventRef(span.sid,int(left),int(right)),float(np.mean(p[run]))))
    return result


def _overlap_above(a,b,numerator,denominator):
    overlap=max(0,min(a.end_ms,b.end_ms)-max(a.start_ms,b.start_ms))
    union=max(a.end_ms,b.end_ms)-min(a.start_ms,b.start_ms)
    return denominator*overlap>numerator*union


def rescue(base,candidates,mapping,threshold,cap):
    if threshold not in (.5,.65,.8,.9,.95,.99) or cap not in (0,1,2,3): raise ValueError('unregistered rescue policy')
    if any(e.sid not in mapping for e in base) or any(e.sid not in mapping or not np.isfinite(p) or not 0<=p<=1 for e,p in candidates):
        raise ValueError('invalid rescue candidate')
    result=list(base)
    if not cap: return result
    kept=[];counts={}
    for event,score in sorted(candidates,key=lambda row:(-row[1],mapping[row[0].sid],row[0].sid,row[0].start_ms,row[0].end_ms)):
        subject=mapping[event.sid]
        if score<threshold or counts.get(subject,0)>=cap: continue
        if any(mapping[b.sid]==subject and _overlap_above(event,b,1,10) for b in base): continue
        if any(mapping[b.sid]==subject and _overlap_above(event,b,3,10) for b in kept): continue
        kept.append(event);counts[subject]=counts.get(subject,0)+1
    return result+kept
