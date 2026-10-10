"""S9: fixed-training-set autopsy; no validation scoring or weight updates."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score

from scripts.research.representation_v2 import adapt_probe as ap
from scripts.research.representation_v2.low_capacity import subject_weights

ep=ap.ep


def saved_logits(x,payload):
    x=np.asarray(x);mean=np.asarray(payload['mean']);scale=np.asarray(payload['scale'])
    weight=np.asarray(payload['coefficient']).reshape(-1);bias=float(payload['intercept'][0])
    if x.ndim!=2 or x.shape[1:]!=mean.shape or scale.shape!=mean.shape or weight.shape!=mean.shape:
        raise ValueError('readout width mismatch')
    if not np.isfinite(x).all() or not np.isfinite(scale).all() or (scale<=0).any():
        raise ValueError('finite inputs and positive scale required')
    return ((x-mean)/scale)@weight+bias


def gradient_relation(target,source):
    target=target.detach().double().flatten();source=source.detach().double().flatten()
    if target.shape!=source.shape or not torch.isfinite(target).all() or not torch.isfinite(source).all():
        raise ValueError('aligned finite gradient vectors required')
    a=float(target.norm());b=float(source.norm())
    return {'target_norm':a,'source_norm':b,'cosine':float(torch.dot(target,source)/(a*b)) if a*b else None,
            'weighted_source_to_target':.25*b/a if a else None}


def target_gradient(encoder,head,bags,labels,readout,device):
    if len(bags)!=len(labels) or not bags:raise ValueError('nonempty aligned target batch required')
    rows=[]
    for bag in bags:
        idx=np.flatnonzero(bag.token_mask)
        if not len(idx):raise ValueError('target gradient requires valid tokens')
        x=torch.as_tensor(bag.values[idx],device=device);m=torch.as_tensor(bag.sample_mask[idx],device=device)
        chunks=[encoder(x[i:i+512],m[i:i+512]) for i in range(0,len(idx),512)]
        z=x.new_zeros((len(bag.token_mask),16));z[torch.as_tensor(idx,device=device)]=torch.cat(chunks)
        rows.append(ap.pool_tensor(z,bag))
    x=torch.stack(rows)
    mean=torch.as_tensor(readout['mean'],dtype=x.dtype,device=device)
    scale=torch.as_tensor(readout['scale'],dtype=x.dtype,device=device)
    logits=head((x-mean)/scale).squeeze(-1)
    loss=nn.functional.binary_cross_entropy_with_logits(logits,torch.as_tensor(labels,dtype=x.dtype,device=device))
    grads=torch.autograd.grad(loss,tuple(encoder.parameters()))
    return torch.cat([g.detach().flatten() for g in grads])


def fixed_metrics(logits,labels,groups):
    weights=subject_weights(groups)
    loss=np.logaddexp(0,logits)-labels*logits
    return {'subject_weighted_bce':float(np.average(loss,weights=weights)),
            'subject_weighted_ap':float(average_precision_score(labels,logits,sample_weight=weights)),
            'subject_bce':{str(s):float(np.mean(loss[groups==s])) for s in np.unique(groups)}}


def _head(payload,device):
    head=nn.Linear(133,1).to(device)
    with torch.no_grad():
        head.weight.copy_(torch.as_tensor(payload['coefficient'],dtype=torch.float32,device=device))
        head.bias.copy_(torch.as_tensor(payload['intercept'],dtype=torch.float32,device=device))
    return head.requires_grad_(False).eval()


def run(root):
    output=root/'outputs/challenger_70/representation_v2/amendment_a3_optimization'
    if output.exists():raise RuntimeError('S9 attempt exists; no silent overwrite')
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
        raise RuntimeError('commit audit code/spec first')
    started=time.perf_counter();snapshot=ep.protected_snapshot(root)
    source=root/'outputs/challenger_70/representation_v2/amendment_a3_source'
    previous=root/'outputs/challenger_70/representation_v2/amendment_a3_adapt'
    inputs={name:hashlib.sha256((previous/name).read_bytes()).hexdigest()
            for name in ('fold0_policy0_scores.json','fold0_policy0_adaptation.json')}
    for name,digest in inputs.items():
        raw=subprocess.check_output(['git','show','8206cc6:outputs/challenger_70/representation_v2/amendment_a3_adapt/'+name],cwd=root)
        if hashlib.sha256(raw).hexdigest()!=digest:raise RuntimeError('sealed S8 input identity mismatch')
    metadata=json.loads((previous/'fold0_policy0_scores.json').read_text())
    saved=json.loads((previous/'fold0_policy0_adaptation.json').read_text())
    train=metadata['train'];excluded=set(metadata['score'])|set(metadata['excluded'])
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4)
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    spec=root/'docs/challenge_070_amendment_rv2_a3_optimization.md'
    ep.write_json(output/'plan.json',{'id':'RV2-A3-S9-001','code_sha':code,'inputs':inputs,
        'spec_sha256':hashlib.sha256(spec.read_bytes()).hexdigest(),'train_subjects':train,
        'excluded_subjects':sorted(excluded),'device':device,'producer_fits':8,'encoder_steps':0,
        'verifier_fits':0,'readout_replay_fits':1,'health_files_read':0,'outer_calls':0})
    try:
        payload=ep.load_source_artifact(source,ep.REGISTERED_SOURCE_SHA)
        config=ep.registered_e03_config(0,workers=4)
        data=ep.runner.FilesystemDataSource(root).load_outer_train_only(config)
        data=ep.slice_outer_train(data,train);graph={}
        ledger,validation=ep.build_candidate_ledgers(data,config,train,(),fit_audit=graph)
        assert validation is None
        ep.check_graph(graph,'candidates',excluded,set(train))
        bags=ep.unitfree_bags(ep.load_candidate_bags(root,ledger.candidates,config=ep.PreprocessConfig(gravity_align=False)))
        valid=np.array([b.token_mask.any() for b in bags]);bags=tuple(b for b,v in zip(bags,valid) if v)
        y=ledger.labels[valid];groups=ledger.groups[valid]
        if set(map(str,groups))!=set(saved['fit_subjects']):raise RuntimeError('S8 training ownership mismatch')
        encoders={}
        for name,state in [('initial',payload['encoder']),('final',saved['encoder'])]:
            encoder=ep.MotionBranch().token_encoder
            state={k:torch.as_tensor(v,dtype=encoder.state_dict()[k].dtype) for k,v in state.items()}
            encoder.load_state_dict(state,strict=True);encoders[name]=encoder.requires_grad_(False).to(device).eval()
        features={name:np.array([ap.pool_ordered(z,b) for z,b in zip(ap.encode_tokens(e,bags,device),bags)])
                  for name,e in encoders.items()}
        readout=saved['initial_readout'];recomputed=ap.readout_payload(ap.fit_readout(features['initial'],y,groups))
        for key in ('mean','scale','coefficient','intercept'):
            np.testing.assert_allclose(recomputed[key],readout[key],rtol=1e-7,atol=1e-8,err_msg='S8 train ledger/readout replay')
        final_readout={**readout,'coefficient':saved['final_target_head']['weight'],
                       'intercept':saved['final_target_head']['bias']}
        readouts={'initial':readout,'final':final_readout}
        metrics={f'{e}_encoder_{h}_head':fixed_metrics(saved_logits(x,r),y,groups)
                 for e,x in features.items() for h,r in readouts.items()}
        drift=np.abs((features['final']-features['initial'])/np.asarray(readout['scale']))[:,:128]
        params=[(encoders['initial'].state_dict()[k],v) for k,v in encoders['final'].state_dict().items()]
        relative=float(torch.sqrt(sum(((b-a)**2).sum() for a,b in params)/sum((a**2).sum() for a,b in params)))
        ep.write_json(output/'train_metrics.json',{'valid_candidates':int(valid.sum()),'invalid_candidates':int((~valid).sum()),
            'positive_candidates':int(y.sum()),'subject_count':len(np.unique(groups)),'metrics':metrics,
            'standardized_latent_absolute_drift_quantiles':np.quantile(drift,[.5,.9,.99,1]).tolist(),
            'encoder_relative_l2_drift':relative,'initial_readout_replayed':True,'producer_graph':graph})
        # Source validation arrays exist in the pickle; only registered source-train tokens enter gradients.
        source_plan=json.loads((source/'plan.json').read_text())
        for path,identity in source_plan['source_files'].items():
            if ap._file_sha(root/path)!=identity['sha256']:raise RuntimeError('source raw identity mismatch')
        bank=ap.load_source_bank(ap.FDIAdapter(root/'FDdatasets/doi-10.48804-CN8VBB/v1.0'))
        keep=np.isin(bank['groups'],payload['provenance']['source_train'])
        bank={k:bank[k][keep] for k in ('values','labels','groups')}
        source_head=nn.Linear(16,1).to(device);source_head.load_state_dict(payload['head']);source_head.requires_grad_(False)
        owners=np.unique(groups);members=[np.flatnonzero(groups==s) for s in owners]
        source_owners=np.unique(bank['groups']);source_members={s:{v:np.flatnonzero((bank['groups']==s)&(bank['labels']==v))
            for v in np.unique(bank['labels'][bank['groups']==s])} for s in source_owners}
        rng=np.random.default_rng(20261012);batches=[]
        for batch in range(4):
            idx=[rng.choice(members[s]) for s in rng.integers(len(owners),size=4)]
            si=[]
            for s in rng.choice(source_owners,size=64):
                label=rng.choice(list(source_members[s]));si.append(rng.choice(source_members[s][label]))
            row={'batch':batch,'target_subjects':groups[idx].tolist(),'target_labels':y[idx].tolist(),'states':{}}
            for name,original in encoders.items():
                encoder=deepcopy(original).requires_grad_(True).eval()
                target=target_gradient(encoder,_head(readouts[name],device),[bags[i] for i in idx],y[idx],readout,device)
                torch.manual_seed(20261012+batch)
                if device=='cuda':torch.cuda.manual_seed_all(20261012+batch)
                x=ap.rotate_triads(torch.as_tensor(bank['values'][si],device=device),ap.rotation_matrices(64,device))
                logits=source_head(encoder(x,torch.ones(64,125,dtype=torch.bool,device=device))).squeeze(-1)
                loss=nn.functional.binary_cross_entropy_with_logits(logits,torch.as_tensor(bank['labels'][si],dtype=torch.float32,device=device))
                gradient=torch.cat([g.detach().flatten() for g in torch.autograd.grad(loss,tuple(encoder.parameters()))])
                row['states'][name]=gradient_relation(target,gradient)
                if row['states'][name]['target_norm']==0:raise RuntimeError('target gradient does not reach encoder')
            batches.append(row)
        ep.write_json(output/'gradient_analysis.json',{'batches':batches,'source_train':payload['provenance']['source_train'],
            'optimizer_steps':0,'note':'four fixed stochastic batches; diagnostic not causal proof'})
        if any(hashlib.sha256((previous/n).read_bytes()).hexdigest()!=h for n,h in inputs.items()):
            raise RuntimeError('audit input changed')
        if ep.protected_snapshot(root)!=snapshot:raise RuntimeError('release protected surface changed')
        size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if time.perf_counter()-started>1800 or size>2*1024**2 or shutil.disk_usage(root).free<20*1024**3:
            raise RuntimeError('audit resource budget exceeded')
        ep.write_json(output/'state.json',{'status':'S9_TRAIN_ONLY_AUDIT_COMPLETE','outer_calls':0,'health_files_read':0,
            'encoder_steps':0,'producer_fits':8,'verifier_fits':0,'readout_replay_fits':1,'wall_seconds':time.perf_counter()-started,
            'protected_unchanged':True,'persistent_token_cache_bytes':0})
        print(json.dumps(metrics),flush=True)
    except Exception as exc:
        ep.write_json(output/'state.json',{'status':'S9_INCOMPLETE','error':str(exc),'outer_calls':0})
        raise


if __name__=='__main__':
    with ep.threadpool_limits(limits=4):run(Path(__file__).resolve().parents[3])
