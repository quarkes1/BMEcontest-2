"""A3 source-only gesture learning with explicit dimensionless input."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly
from scipy.special import logit
import torch
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from scripts.research.representation_v2.explore import write_json
from scripts.research.representation_v2.fd_source import FDIAdapter
from scripts.research.representation_v2.health import logit_diagnostics
from scripts.research.representation_v2.models import MotionBranch
from src.pipeline.challenger.protection import protected_snapshot


def normalize_triads(values, mask):
    x=np.asarray(values,dtype=np.float64); keep=np.asarray(mask,dtype=bool)
    if x.ndim!=3 or x.shape[1:]!=(6,125) or keep.shape!=(len(x),125):
        raise ValueError('unit-free input requires N x6 x125 plus sample masks')
    valid=(keep.sum(1)>=100) & np.all(np.isfinite(x)|~keep[:,None,:],axis=(1,2))
    safe=np.where(keep[:,None,:]&np.isfinite(x),x,0.)
    centered=safe-safe.sum(2,keepdims=True)/keep.sum(1).clip(1)[:,None,None]
    centered=np.where(keep[:,None,:],centered,0.)
    z=np.zeros_like(centered)
    for offset in (0,3):
        rms=np.sqrt(np.sum(centered[:,offset:offset+3]**2,axis=(1,2))/keep.sum(1).clip(1))
        good=np.isfinite(rms)&(rms>0)
        valid &= good
        z[:,offset:offset+3]=centered[:,offset:offset+3]/np.where(good,rms,1.)[:,None,None]
    z[~valid]=0.
    return z.astype(np.float32),valid


def label_starts(labels):
    y=np.asarray(labels)
    if y.ndim!=1 or not np.isin(y,(0,1,2)).all():
        raise ValueError('FD point labels must be 0/1/2')
    starts=np.arange(0,len(y)-319,320)
    pos=np.r_[0,np.cumsum(y==1)]
    nonother=np.r_[0,np.cumsum(y!=0)]
    positive=(pos[starts+193]-pos[starts+128])>0
    negative=(nonother[np.minimum(starts+480,len(y))]-nonother[np.maximum(starts-160,0)])==0
    selected=positive|negative
    return starts[selected],positive[selected].astype(np.int8)


def source_resample(values):
    x=np.asarray(values,dtype=np.float64)
    if x.shape!=(320,6) or not np.isfinite(x).all():
        raise ValueError('source token must have320 finite six-channel samples')
    mean=x.mean(0)
    filtered=sosfiltfilt(butter(4,10,fs=64,output='sos'),x-mean,axis=0)
    return (resample_poly(filtered,25,64,axis=0,padtype='line')+mean).T.astype(np.float32)


def rotation_matrices(count,device):
    q=torch.randn(count,4,device=device)
    q=q/torch.linalg.vector_norm(q,dim=1,keepdim=True).clamp_min(1e-12)
    w,x,y,z=q.unbind(1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),dim=1).reshape(-1,3,3)


def rotate_triads(batch,rotations):
    return torch.cat((rotations@batch[:,:3],rotations@batch[:,3:]),dim=1)


def sample_source_rows(labels,participant,seed=20260926):
    rng=np.random.default_rng(np.random.SeedSequence([seed,participant]))
    selected=[]
    for label in (0,1):
        idx=np.flatnonzero(labels==label)
        selected.extend(rng.choice(idx,min(256,len(idx)),replace=False).tolist())
    return np.array(sorted(selected),dtype=int)


def training_priors(counts,train,groups,labels):
    if not len(labels) or set(groups)-set(train):
        raise ValueError('prior input includes non-training participants or is empty')
    before=[row['available'] for row in counts if row['participant'] in train]
    total=sum(sum(row.values()) for row in before)
    if not total: raise ValueError('no eligible source population')
    return {'sampler':float(np.mean([np.mean(np.unique(labels[groups==g])) for g in np.unique(groups)])),
            'bank':float(np.mean(labels)),'eligible_raw':sum(row.get(1,0) for row in before)/total}


def _file_sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def available_memory_bytes():
    if os.name=='nt':
        value=subprocess.check_output(['powershell','-NoProfile','-Command',
                                       '(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory'],
                                      text=True,creationflags=subprocess.CREATE_NO_WINDOW)
        return int(value.strip())*1024
    return os.sysconf('SC_AVPHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')


def load_source_bank(adapter,seed=20260926):
    features,labels=adapter._load_hand('R')
    tokens,targets,groups,starts_kept=[],[],[],[]
    counts=[]
    for pid in range(adapter.expected_participants):
        x,y=features[pid],labels[pid]
        adapter._validate_participant_arrays(x,y,pid,'R')
        starts,target=label_starts(y)
        selected=sample_source_rows(target,pid,seed)
        if not len(selected): raise ValueError(f'FD participant{pid} has no eligible source tokens')
        raw=np.stack([source_resample(x[start:start+320]) for start in starts[selected]])
        normalized,valid=normalize_triads(raw,np.ones((len(raw),125),bool))
        counts.append({'participant':pid,'available':dict(Counter(map(int,target))),
                       'selected':dict(Counter(map(int,target[selected]))),
                       'valid':dict(Counter(map(int,target[selected][valid])))})
        tokens.extend(normalized[valid]); targets.extend(target[selected][valid].tolist())
        groups.extend([pid]*int(valid.sum())); starts_kept.extend(starts[selected][valid].tolist())
    del features,labels
    return {'values':np.asarray(tokens),'labels':np.asarray(targets),'groups':np.asarray(groups),
            'starts':np.asarray(starts_kept),'counts':counts}


def run(root):
    root=Path(root); output=root/'outputs/challenger_70/representation_v2/amendment_a3_source'
    if (output/'plan.json').exists(): raise RuntimeError('source attempt exists; no silent retry')
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
        raise RuntimeError('commit source registration and code first')
    prior=json.loads((root/'outputs/challenger_70/representation_v2/amendment_a2/state.json').read_text())
    if prior['status']!='R1_NO_SUPPORTED_SSL_BRANCH': raise RuntimeError('A2 completion gate absent')
    started=time.perf_counter(); snapshot=protected_snapshot(root)
    adapter=FDIAdapter(root/'FDdatasets/doi-10.48804-CN8VBB/v1.0')
    train,val=adapter.participant_split(seed=20260926)
    paths=[adapter.source/name for name in ('X_R.pkl','Y_R.pkl')]
    if 2*sum(p.stat().st_size for p in paths)>.8*available_memory_bytes():
        raise RuntimeError('insufficient available RAM for bounded source decode')
    if shutil.disk_usage(root).free<20*1024**3: raise RuntimeError('disk reserve below20GiB')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    free_cuda=torch.cuda.mem_get_info()[0] if device=='cuda' else None
    if free_cuda is not None and free_cuda<1280*1024**2: raise RuntimeError('insufficient GPU reserve')
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    plan={'id':'RV2-A3-S0-001','code_sha':sha,'seed':20260926,'steps':5000,'device':device,
          'source_train':list(train),'source_validation':list(val),'target_signal_read':False,'outer_calls':0,
          'spec_sha256':_file_sha(root/'docs/challenge_070_amendment_rv2_a3.md'),
          'source_files':{p.relative_to(root).as_posix():{'bytes':p.stat().st_size,'sha256':_file_sha(p)} for p in paths},
          'protected_digest':hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest()}
    write_json(output/'plan.json',plan)
    plan_sha=_file_sha(output/'plan.json')
    write_json(output/'state.json',{'status':'S0_IN_PROGRESS','outer_calls':0})
    try:
        print('S0 loading only FD-I right wrist; no target IMU access',flush=True)
        bank=load_source_bank(adapter)
        write_json(output/'population.json',{'counts':bank['counts'],'valid_tokens':len(bank['labels']),
                                           'persistent_token_cache_bytes':0})
        keep=np.isin(bank['groups'],train); held=np.isin(bank['groups'],val)
        if np.any(keep&held) or not np.all(keep|held): raise RuntimeError('source partition mismatch')
        x,y,g=bank['values'][keep],bank['labels'][keep],bank['groups'][keep]
        xv,yv=bank['values'][held],bank['labels'][held]
        del bank['values']
        choices={int(subject):{int(label):np.flatnonzero((g==subject)&(y==label)) for label in np.unique(y[g==subject])}
                 for subject in np.unique(g)}
        names=sorted(choices)
        priors=training_priors(bank['counts'],train,g,y)
        sampler_prior=priors['sampler']
        if set(y.tolist())!={0,1} or set(yv.tolist())!={0,1}: raise RuntimeError('source groups lack both classes')
        torch.set_num_threads(4); torch.manual_seed(20260926)
        encoder=MotionBranch().token_encoder.to(device)
        initial={k:v.detach().cpu().clone() for k,v in encoder.state_dict().items()}
        head=nn.Linear(16,1).to(device)
        params=list(encoder.parameters())+list(head.parameters())
        optimizer=torch.optim.AdamW(params,lr=1e-3,weight_decay=1e-4)
        rng=np.random.default_rng(20260926); history=[]; recent=[]; class_counts=Counter(); subject_counts=Counter()
        training_start=time.perf_counter()
        if device=='cuda': torch.cuda.reset_peak_memory_stats()
        for step in range(1,5001):
            if time.perf_counter()-started>7200 or time.perf_counter()-training_start>3600:
                raise TimeoutError('source fixed-budget incomplete')
            selected=[]
            for _ in range(128):
                subject=int(rng.choice(names)); label=int(rng.choice(list(choices[subject])))
                selected.append(int(rng.choice(choices[subject][label])))
                class_counts[label]+=1; subject_counts[subject]+=1
            batch=torch.as_tensor(x[selected],device=device)
            r=rotation_matrices(len(batch),device)
            batch=rotate_triads(batch,r)
            masks=torch.ones(len(batch),125,device=device,dtype=torch.bool)
            optimizer.zero_grad(set_to_none=True)
            logits=head(encoder(batch,masks)).squeeze(1)
            loss=F.binary_cross_entropy_with_logits(logits,torch.as_tensor(y[selected],dtype=torch.float32,device=device))
            if not torch.isfinite(loss): raise RuntimeError('non-finite source loss')
            loss.backward(); norm=torch.nn.utils.clip_grad_norm_(params,1.)
            if not torch.isfinite(norm): raise RuntimeError('non-finite source gradient')
            optimizer.step(); recent.append([float(loss.detach()),float(norm)])
            if step%100==0:
                means=np.mean(recent,axis=0); recent=[]
                history.append({'step':step,'loss':float(means[0]),'gradient_norm':float(means[1]),
                                'elapsed_seconds':time.perf_counter()-training_start})
                write_json(output/'training_history.json',history)
                if step%1000==0: print(f'S0 step={step} train_loss={means[0]:.4f}',flush=True)
        train_seconds=time.perf_counter()-training_start
        encoder.eval(); head.eval(); outputs=[]
        with torch.inference_mode():
            for left in range(0,len(xv),512):
                batch=torch.as_tensor(xv[left:left+512],device=device)
                outputs.extend(head(encoder(batch,torch.ones(len(batch),125,dtype=torch.bool,device=device))).squeeze(1).cpu().tolist())
        z=np.asarray(outputs); metric=logit_diagnostics(yv,z)
        baseline=logit_diagnostics(yv,np.full(len(yv),logit(sampler_prior)))
        natural=logit_diagnostics(yv,np.full(len(yv),logit(priors['eligible_raw'])))
        bank_baseline=logit_diagnostics(yv,np.full(len(yv),logit(priors['bank'])))
        weights={'schema':'rv2-a3-unitfree-source-v1','encoder':{k:v.cpu() for k,v in encoder.state_dict().items()},
                 'head':{k:v.cpu() for k,v in head.state_dict().items()},'random_encoder':initial,
                 'provenance':{'code_sha':sha,'plan_sha256':plan_sha,'source_train':list(train),'source_validation':list(val),
                               'target_subjects':[],'steps':5000,'normalization':'per-token-per-triad-demean-rms'}}
        temporary=output/'source_final.pt.tmp'
        try:
            torch.save(weights,temporary); torch.load(temporary,weights_only=True,map_location='cpu')
            temporary.replace(output/'source_final.pt')
        finally: temporary.unlink(missing_ok=True)
        checks={'finite':bool(np.isfinite(z).all()),'isolation':not bool(set(train)&set(val)),'optimizer':history[-1]['step']==5000,
                'resource':train_seconds<=3600 and time.perf_counter()-started<=7200 and
                           sum(p.stat().st_size for p in (root/'outputs/challenger_70/representation_v2').rglob('*.pt'))<2*1024**3 and
                           (device=='cpu' or torch.cuda.max_memory_allocated()<=.8*free_cuda)}
        passed=all(checks.values()) and metric['average_precision']>=metric['prevalence']+.1 and metric['raw_logit_bce']<=baseline['raw_logit_bce']
        unchanged=protected_snapshot(root)==snapshot
        if not unchanged: raise RuntimeError('protected release changed')
        report={'status':'S0_SUPPORTS_TARGET_PROBE' if passed else 'S0_NO_SUPPORTED_SOURCE_ENCODER',
                'passed':bool(passed),'metrics':metric,'sampler_prior':sampler_prior,'sampler_baseline':baseline,
                'eligible_raw_prior_baseline_on_sampled_validation':natural,'capped_bank_prior_baseline':bank_baseline,
                'training_priors':priors,'checks':checks,'training_seconds':train_seconds,
                'wall_seconds':time.perf_counter()-started,'optimizer_steps':5000,'outer_calls':0,'target_signal_read':False,
                'protected_unchanged':True,'persistent_token_cache_bytes':0,'train_token_count':len(y),
                'class_draws':dict(class_counts),'subject_draws':dict(subject_counts),
                'peak_cuda_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
                'checkpoint_sha256':_file_sha(output/'source_final.pt')}
        rows=[{'participant':int(pid),'start_sample':int(start),'label':int(label),'logit':float(score)}
              for pid,start,label,score in zip(bank['groups'][held],bank['starts'][held],yv,z)]
        write_json(output/'validation.json',rows); write_json(output/'summary.json',report)
        write_json(output/'state.json',{'status':report['status'],'outer_calls':0})
        print(json.dumps(report,sort_keys=True),flush=True)
    except Exception as exc:
        write_json(output/'state.json',{'status':'S0_INCOMPLETE','error':str(exc),'wall_seconds':time.perf_counter()-started,'outer_calls':0})
        raise


if __name__=='__main__':
    with threadpool_limits(limits=4): run(Path(__file__).resolve().parents[3])
