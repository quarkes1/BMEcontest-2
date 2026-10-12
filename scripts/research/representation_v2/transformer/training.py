"""Train-only moments, paired subject-balanced sampling and fixed-step readouts."""
import hashlib
import time

import numpy as np
import torch

from .heads import build_head


def fit_moments(sequences,allowed_subjects):
    by_subject={}
    for sequence in sequences:
        subject=sequence['subject']
        if subject not in allowed_subjects: raise ValueError('unexpected moments subject')
        tokens=by_subject.setdefault(subject,{})
        for chunk in sequence['chunks']:
            for index,vector,valid in zip(chunk['indices'],chunk['embeddings'],chunk['valid']):
                if valid:
                    key=(sequence['sid'],int(index))
                    if key in tokens and not np.array_equal(tokens[key],vector):
                        raise ValueError('duplicate token embedding differs')
                    tokens[key]=np.asarray(vector,dtype=np.float64)
    populations=[np.stack(list(v.values())) for v in by_subject.values() if v]
    if not populations: raise ValueError('no valid train tokens')
    center=np.mean([v.mean(0) for v in populations],axis=0)
    second=np.mean([np.square(v).mean(0) for v in populations],axis=0)
    scale=np.sqrt(np.maximum(second-center**2,0)).clip(1e-6)
    if not np.isfinite(center).all() or not np.isfinite(scale).all(): raise ValueError('invalid moments')
    return center.astype(np.float32),scale.astype(np.float32)


def pack_sequences(sequences,center,scale,device):
    chunks=[]; owners=[]; regions=[]; coverage=np.zeros((len(sequences),3),np.float32)
    for owner,sequence in enumerate(sequences):
        totals=np.zeros(3); valid_counts=np.zeros(3)
        for c in sequence['chunks']:
            region=c['region']; totals[region]+=len(c['valid']); valid_counts[region]+=c['valid'].sum()
            chunks.append((c,sequence['start_ms'])); owners.append(owner); regions.append(region)
        coverage[owner]=valid_counts/np.maximum(totals,1)
    length=max((len(c['valid']) for c,_ in chunks),default=1)
    x=np.zeros((len(chunks),length,17),np.float32)
    mask=np.zeros((len(chunks),length),bool); times=np.zeros((len(chunks),length),np.float32)
    for i,(c,start) in enumerate(chunks):
        n=len(c['valid']); keep=c['valid']; mask[i,:n]=keep
        x[i,:n,:16]=np.where(keep[:,None],(c['embeddings']-center)/scale,0.)
        x[i,:n,16]=np.where(keep,c['coverage'],0.)
        times[i,:n]=(c['times']-start)/1000.
    return {'x':torch.as_tensor(x,device=device),'mask':torch.as_tensor(mask,device=device),
            'times':torch.as_tensor(times,device=device),'owner':torch.tensor(owners,dtype=torch.long,device=device),
            'region':torch.tensor(regions,dtype=torch.long,device=device),
            'coverage':torch.as_tensor(coverage,device=device),'count':len(sequences)}


def fit_head(arm,train_sequences,labels,groups,*,seed,device,steps=1000,resource_callback=None):
    torch.set_num_threads(4)
    labels=np.asarray(labels); groups=np.asarray(groups)
    if len(labels)!=len(train_sequences) or groups.shape!=labels.shape or set(labels.tolist())!={0,1}:
        raise ValueError('aligned binary training labels required')
    if any(s['subject']!=g for s,g in zip(train_sequences,groups)): raise ValueError('sequence owner mismatch')
    if steps<1: raise ValueError('positive step budget required')
    center,scale=fit_moments(train_sequences,set(groups))
    torch.manual_seed(seed)
    if device=='cuda': torch.cuda.manual_seed_all(seed)
    model=build_head(arm).to(device).train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    members=[np.flatnonzero(groups==s) for s in np.unique(groups)]
    rng=np.random.default_rng(seed); sample_hash=hashlib.sha256(); history=[]
    started=time.perf_counter()
    for step in range(1,steps+1):
        if resource_callback: resource_callback()
        if time.perf_counter()-started>1200: raise RuntimeError('head fit time budget exceeded')
        indices=np.asarray([rng.choice(members[s]) for s in rng.integers(len(members),size=16)],dtype=np.int64)
        sample_hash.update(indices.tobytes())
        batch=pack_sequences([train_sequences[i] for i in indices],center,scale,device)
        optimizer.zero_grad(set_to_none=True)
        logits=model(batch)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.as_tensor(labels[indices],dtype=torch.float32,device=device))
        if not torch.isfinite(loss): raise RuntimeError('nonfinite head loss')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if step%100==0 or step==steps:
            history.append({'step':step,'loss':float(loss.detach()),'preclip_norm':float(norm)})
            print(f'A6 arm={arm} step={step}/{steps} loss={float(loss.detach()):.5f}',flush=True)
    if device=='cuda': torch.cuda.synchronize()
    return {'model':model.eval(),'center':center,'scale':scale,'steps':steps,'clip_norm':1.,
            'sampling_sha256':sample_hash.hexdigest(),'history':history,
            'fit_subjects':sorted(map(str,set(groups))),'elapsed_seconds':time.perf_counter()-started,
            'parameters':sum(p.numel() for p in model.parameters()),'device':device}


def predict_head(fitted,sequences):
    result=[]; model=fitted['model']; device=fitted['device']
    model.eval()
    with torch.inference_mode():
        for left in range(0,len(sequences),16):
            batch=pack_sequences(sequences[left:left+16],fitted['center'],fitted['scale'],device)
            result.extend(torch.sigmoid(model(batch)).cpu().tolist())
    p=np.asarray(result,dtype=np.float64)
    if not np.isfinite(p).all(): raise ValueError('nonfinite head predictions')
    return p


def fuse_scores(base,motion,valid):
    base,motion,valid=np.asarray(base),np.asarray(motion),np.asarray(valid,dtype=bool)
    if base.ndim!=1 or motion.shape!=base.shape or valid.shape!=base.shape:
        raise ValueError('score alignment differs')
    if not np.isfinite(base).all() or not np.isfinite(motion).all(): raise ValueError('nonfinite probability')
    if np.any((base<0)|(base>1)|(motion<0)|(motion>1)): raise ValueError('invalid probability')
    return np.where(valid,.75*base+.25*motion,base)
