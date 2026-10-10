"""S12: frozen source-head probability morphology, strict nested event test."""
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time
import warnings

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning

from scripts.research.representation_v2 import adapt_probe as ap

ep=ap.ep
SCHEMA=('prob_mean','prob_std','prob_max','prob_q25','prob_q50','prob_q75','prob_q90','prob_q95',
        'above_020','above_050','above_080','runs_020','longest_fraction_020',
        'runs_050','longest_fraction_050','runs_080','longest_fraction_080',
        'lag1_correlation','lag2_correlation','lag1_mean_absolute_difference',
        'valid_token_fraction','observed_sample_fraction','log_valid_count','log_duration_seconds')
S11_COMMIT='3053f66'
S11_PATH='outputs/challenger_70/representation_v2/amendment_a3_optimization_event'


def morphology(probabilities,bag):
    p=np.asarray(probabilities,dtype=np.float64);valid=np.asarray(bag.token_mask)
    times=np.asarray(bag.token_center_ms);samples=np.asarray(bag.sample_mask);n=len(valid)
    if (valid.dtype!=bool or p.shape!=(n,) or times.shape!=(n,) or times.dtype.kind not in 'iu' or
        samples.dtype!=bool or samples.shape!=(n,125) or bag.end_ms<=bag.start_ms or
        np.any(times[1:]<=times[:-1])):
        raise ValueError('ordered aligned token geometry and positive duration required')
    if not np.isfinite(p[valid]).all() or np.any((p[valid]<0)|(p[valid]>1)):
        raise ValueError('finite valid source probabilities in [0,1] required')
    metadata=[float(valid.mean()) if n else 0.,float(samples.mean()) if n else 0.,
              np.log1p(valid.sum()),np.log1p((bag.end_ms-bag.start_ms)/1000)]
    if not valid.any():return np.r_[np.full(20,np.nan),metadata]
    v=p[valid];parts=[v.mean(),v.std(),v.max(),*np.quantile(v,[.25,.5,.75,.9,.95],method='linear')]
    parts.extend(float((v>=threshold).mean()) for threshold in (.2,.5,.8))
    adjacent=(np.diff(times)<=5000)&valid[:-1]&valid[1:]
    for threshold in (.2,.5,.8):
        count=longest=current=0
        for i,above in enumerate(valid&(p>=threshold)):
            if above:
                if i==0 or not adjacent[i-1] or current==0:count+=1;current=0
                current+=1;longest=max(longest,current)
            else:current=0
        parts.extend((count,longest/len(v)))
    for lag in (1,2):
        pairs=np.ones(max(n-lag,0),bool)
        for shift in range(lag):pairs &= adjacent[shift:shift+len(pairs)]
        left=p[:n-lag][pairs] if n>=lag else np.empty(0)
        right=p[lag:][pairs]
        parts.append(float(np.corrcoef(left,right)[0,1]) if len(left)>=2 and left.std()>0 and right.std()>0 else np.nan)
    parts.append(float(np.abs(np.diff(p)[adjacent]).mean()) if adjacent.any() else np.nan)
    result=np.r_[parts,metadata]
    if result.shape!=(24,) or np.isinf(result).any():raise RuntimeError('invalid fixed morphology schema')
    return result


def source_probabilities(vectors,bags,head,*,device):
    if len(vectors)!=len(bags) or any(p.requires_grad for p in head.parameters()):
        raise ValueError('aligned bags and frozen source head required')
    head.to(device).eval();before={k:v.detach().clone() for k,v in head.state_dict().items()};rows=[]
    with torch.inference_mode():
        for z,b in zip(vectors,bags):
            z=np.asarray(z)
            if z.shape!=(len(b.token_mask),16) or not np.isfinite(z[b.token_mask]).all():
                raise ValueError('finite aligned valid token embeddings required')
            p=np.full(len(z),np.nan)
            if b.token_mask.any():
                logits=head(torch.as_tensor(z[b.token_mask],dtype=torch.float32,device=device)).squeeze(-1)
                p[b.token_mask]=torch.sigmoid(logits).cpu().numpy()
            if not np.isfinite(p[b.token_mask]).all():raise RuntimeError('nonfinite source score')
            rows.append(p)
    if any(not torch.equal(v,before[k]) for k,v in head.state_dict().items()):raise RuntimeError('source head changed')
    return rows


def gesture_features(base,motion):
    base=np.asarray(base);motion=np.asarray(motion)
    if (base.ndim!=2 or base.shape[1]!=116 or motion.shape!=(len(base),24) or
        np.isinf(base).any() or np.isinf(motion).any() or not np.isfinite(motion[:,20:]).all()):
        raise ValueError('aligned 116D + 24D morphology required')
    return np.concatenate((base,motion),axis=1)


def gesture_probabilities(x,y,v,*,seed):
    x=np.asarray(x);v=np.asarray(v);y=np.asarray(y)
    if (x.ndim!=2 or v.ndim!=2 or x.shape[1]!=140 or v.shape[1]!=140 or y.shape!=(len(x),) or
        set(y.tolist())!={0,1} or np.isinf(x).any() or np.isinf(v).any()):
        raise ValueError('binary aligned 140D verifier required')
    probabilities=[]
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        for model in (ap.jp.runner._verifier_estimator(seed+20,.1),ap.jp.runner._verifier_lgbm_estimator(seed+21)):
            model.fit(x,y);probabilities.append(ap.jp.runner._positive_probability(model,v))
    result=.5*probabilities[0]+.5*probabilities[1]
    if not np.isfinite(result).all() or ((result<0)|(result>1)).any():raise RuntimeError('nonfinite event probabilities')
    return result


def frozen_reference_replay(root,prefix,evidence):
    path='outputs/challenger_70/representation_v2/amendment_a3_adapt/'+prefix+'_scores.json'
    raw=subprocess.check_output(['git','show','8206cc6:'+path],cwd=root)
    old=json.loads(raw);normalized=json.loads(json.dumps(evidence,allow_nan=False))
    for key in ('train','score','excluded','candidates','groups','labels'):
        if normalized[key]!=old[key]:raise RuntimeError('frozen FD reference ledger differs: '+key)
    np.testing.assert_allclose(normalized['scores']['r'],old['scores']['r'],rtol=1e-6,atol=1e-7)
    return hashlib.sha256(raw).hexdigest()


def prepare(root,payload,output,device):
    head=torch.nn.Linear(16,1);head.load_state_dict(payload['head'],strict=True);head.requires_grad_(False).to(device).eval()
    ep.write_json(output/'feature_schema.json',{'id':'rv2-a3-gesture-morphology-v1','names':list(SCHEMA),
        'feature_widths':{'b':116,'r':251,'f':140},'thresholds':[.2,.5,.8],'comparison':'>=',
        'run_max_gap_ms':5000,'quantile_method':'linear','source_checkpoint_sha256':ep.REGISTERED_SOURCE_SHA,
        'source_provenance':payload['provenance'],'reference_arm':'frozen FD, not random','persistent_token_cache_bytes':0})

    def split(root,data,config,train,score,excluded,encoders,source_train,source_val,output,prefix,device):
        if set(train)&(set(score)|set(excluded)) or set(score)&set(excluded):raise ValueError('split subject overlap')
        graph={};t,v=ep.build_candidate_ledgers(data,config,train,score,fit_audit=graph)
        graph['source']={'fit_subjects':sorted(source_train),'parents':[]}
        graph['base_verifier']={'fit_subjects':list(train),'parents':['candidates']}
        scores={'b':ep.verifier_probabilities(t.features,t.labels,v.features,seed=config.seed)}
        tb=ep.unitfree_bags(ep.load_candidate_bags(root,t.candidates,config=ep.PreprocessConfig(gravity_align=False)))
        vb=ep.unitfree_bags(ep.load_candidate_bags(root,v.candidates,config=ep.PreprocessConfig(gravity_align=False)))
        if device=='cuda':torch.cuda.synchronize()
        started=time.perf_counter()
        tz=ap.encode_tokens(encoders['f'],tb,device);vz=ap.encode_tokens(encoders['f'],vb,device)
        tx=np.array([ap.pool_joint_ordered(z,b) for z,b in zip(tz,tb)]).reshape(-1,135)
        vx=np.array([ap.pool_joint_ordered(z,b) for z,b in zip(vz,vb)]).reshape(-1,135)
        scores['r']=ap.ordered_joint_probabilities(ap.ordered_joint_features(t.features,tx),t.labels,
                                                  ap.ordered_joint_features(v.features,vx),seed=config.seed)
        tm=np.array([morphology(p,b) for p,b in zip(source_probabilities(tz,tb,head,device=device),tb)]).reshape(-1,24)
        vm=np.array([morphology(p,b) for p,b in zip(source_probabilities(vz,vb,head,device=device),vb)]).reshape(-1,24)
        scores['f']=gesture_probabilities(gesture_features(t.features,tm),t.labels,gesture_features(v.features,vm),seed=config.seed)
        for arm in ('r','f'):graph[arm+'_verifier']={'fit_subjects':list(train),'parents':['candidates','source']}
        graph['score']={'fit_subjects':[],'parents':['base_verifier','r_verifier','f_verifier']}
        ep.check_graph(graph,'score',set(score)|set(excluded)|source_val,set(train)|source_train)
        if device=='cuda':torch.cuda.synchronize()
        seconds=time.perf_counter()-started
        evidence={'train':list(train),'score':list(score),'excluded':sorted(excluded),'graph':graph,
            'candidates':[asdict(c) for c in v.candidates],'groups':v.groups.tolist(),'labels':v.labels.tolist(),
            'motion_valid':[bool(b.token_mask.any()) for b in vb],'scores':{a:p.tolist() for a,p in scores.items()},
            'feature_widths':{'b':116,'r':251,'f':140},'encoding_seconds':seconds,
            'source_checkpoint_sha256':ep.REGISTERED_SOURCE_SHA,
            'arm_semantics':{'b':'incumbent','r':'frozen FD ordered251D','f':'frozen source-head morphology140D'}}
        evidence['baseline_source_sha256']=ap.jp.baseline_replay(root,prefix+'_scores.json',evidence)
        evidence['frozen_reference_sha256']=frozen_reference_replay(root,prefix,evidence)
        ep.write_json(output/(prefix+'_scores.json'),evidence)
        del tb,vb,tz,vz,tx,vx,tm,vm;gc.collect()
        return v,scores,graph,seconds
    return split


def assert_s11_closed(root):
    contents={}
    for name in ('summary.json','state.json'):
        raw=subprocess.check_output(['git','show',S11_COMMIT+':'+S11_PATH+'/'+name],cwd=root)
        if (root/S11_PATH/name).read_bytes()!=raw:raise RuntimeError('sealed S11 evidence differs')
        contents[name]=json.loads(raw)
    summary=contents['summary.json']
    if (summary['status']!='S11_NO_SUPPORTED_OPTIMIZATION_EVENT_GAIN' or summary['passed'] is not False or
        summary['completed_splits']!=12 or summary['protected_unchanged'] is not True or summary['outer_calls']!=0 or
        contents['state.json']['status']!=summary['status']):raise RuntimeError('complete failed S11 required')
    return hashlib.sha256((root/S11_PATH/'summary.json').read_bytes()).hexdigest()


def run(root):
    s11_sha=assert_s11_closed(root)
    def checked_prepare(root,payload,output,device):
        ep.write_json(output/'s11_closure_identity.json',{'commit':S11_COMMIT,'summary_sha256':s11_sha})
        return prepare(root,payload,output,device)
    ap.jp.run(root,experiment={'id':'RV2-A3-S12-001','label':'S12','output':'amendment_a3_gesture_morphology',
        'spec':'challenge_070_amendment_rv2_a3_gesture_morphology.md','prepare':checked_prepare,
        'max_wall_seconds':14400,'max_work_seconds':7200,'pass_status':'S12_SUPPORTS_WIDER_INNER_EVENT_VALIDATION',
        'fail_status':'S12_NO_SUPPORTED_GESTURE_MORPHOLOGY'})


if __name__=='__main__':
    with ep.threadpool_limits(limits=4):run(Path(__file__).resolve().parents[3])
