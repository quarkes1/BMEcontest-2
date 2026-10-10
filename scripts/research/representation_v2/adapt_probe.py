"""S8: partition-contained supervised FD encoder adaptation, event evaluation."""
from copy import deepcopy
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
import torch
from torch import nn
from sklearn.exceptions import ConvergenceWarning

from scripts.research.representation_v2 import joint_probe as jp
from scripts.research.representation_v2.low_capacity import pool_ordered, fit_readout, readout_payload
from scripts.research.representation_v2.order_probe import encode_tokens
from scripts.research.representation_v2.unitfree import load_source_bank, FDIAdapter, _file_sha, rotation_matrices, rotate_triads

ep=jp.ep


def pool_tensor(z,bag):
    if z.shape!=(len(bag.token_mask),16):raise ValueError('aligned 16D tokens required')
    placement=np.clip((bag.token_center_ms-bag.start_ms)*4//(bag.end_ms-bag.start_ms),0,3)
    blocks=[]
    for part in range(4):
        rows=z[torch.as_tensor((placement==part)&bag.token_mask,device=z.device)]
        blocks.extend((rows.mean(0),rows.std(0,correction=0)) if len(rows) else (z.new_zeros(16),z.new_zeros(16)))
    # Metadata are deterministic and carry no gradient or fitted state.
    meta=pool_ordered(np.zeros((len(bag.token_mask),16)),bag)[-5:]
    return torch.cat((*blocks,torch.as_tensor(meta,dtype=z.dtype,device=z.device)))


def pool_joint_ordered(z,bag):
    row=pool_ordered(z,bag) if len(bag.token_mask) else np.r_[np.zeros(132),np.log1p((bag.end_ms-bag.start_ms)/1000)]
    if not bag.token_mask.any():row[:128]=np.nan
    return np.r_[row,float(bag.token_mask.mean()) if len(bag.token_mask) else 0.,float(bag.token_mask.any())]


def ordered_joint_features(base,motion):
    if base.ndim!=2 or base.shape[1]!=116 or motion.shape!=(len(base),135):raise ValueError('116D plus135D required')
    if np.isinf(base).any() or np.isinf(motion).any() or not np.isfinite(motion[:,128:]).all():raise ValueError('invalid joint values')
    return np.concatenate((base,motion),axis=1)


def ordered_joint_probabilities(x,y,v,*,seed):
    if x.ndim!=2 or v.ndim!=2 or x.shape[1]!=251 or v.shape[1]!=251 or set(y.tolist())!={0,1}:
        raise ValueError('binary 251D verifier required')
    if np.isinf(x).any() or np.isinf(v).any():raise ValueError('infinite verifier input')
    ps=[]
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        for model in (jp.runner._verifier_estimator(seed+20,.1),jp.runner._verifier_lgbm_estimator(seed+21)):
            model.fit(x,y);ps.append(jp.runner._positive_probability(model,v))
    result=.5*ps[0]+.5*ps[1]
    if not np.isfinite(result).all():raise ValueError('nonfinite verifier output')
    return result


def control_gradient_norm(parameters,clip_norm):
    if clip_norm is not None and (not np.isfinite(clip_norm) or clip_norm<=0):
        raise ValueError('positive finite clip norm or None required')
    norm=nn.utils.clip_grad_norm_(parameters,float('inf') if clip_norm is None else clip_norm)
    if not torch.isfinite(norm):raise RuntimeError('nonfinite adaptation gradient')
    return norm


def adapt_encoder(initial,source_head,bags,y,groups,bank,*,train_subjects,seed,device,steps=1000,batch_size=4,clip_norm=1.,resource_check=None):
    if resource_check is not None:resource_check()
    groups=np.asarray(groups);y=np.asarray(y)
    if len(bags)!=len(y) or len(groups)!=len(y) or set(map(str,groups))-set(train_subjects):
        raise ValueError('training subject ownership mismatch')
    if steps<1:raise ValueError('positive training budget required')
    if not isinstance(batch_size,int) or isinstance(batch_size,bool) or batch_size<1:raise ValueError('positive integer batch size required')
    if clip_norm is not None and (not np.isfinite(clip_norm) or clip_norm<=0):raise ValueError('positive finite clip norm or None required')
    valid=np.array([b.token_mask.any() for b in bags]);keep=np.flatnonzero(valid)
    if not len(keep) or set(y[valid].tolist())!={0,1}:raise ValueError('valid target rows need both classes')
    initial=deepcopy(initial).to(device).requires_grad_(False).eval()
    vectors=encode_tokens(initial,bags,device)
    x=np.array([pool_ordered(z,b) if len(b.token_mask) else np.zeros(133) for z,b in zip(vectors,bags)])
    fitted=fit_readout(x[valid],y[valid],groups[valid]);del vectors,x
    encoder=deepcopy(initial).requires_grad_(True).train()
    source=deepcopy(source_head).to(device).requires_grad_(False).eval()
    torch.manual_seed(seed)
    if device=='cuda':torch.cuda.manual_seed_all(seed)
    head=nn.Linear(133,1).to(device)
    with torch.no_grad():
        head.weight.copy_(torch.as_tensor(fitted['model'].coef_,dtype=torch.float32,device=device))
        head.bias.copy_(torch.as_tensor(fitted['model'].intercept_,dtype=torch.float32,device=device))
    mean=torch.as_tensor(fitted['scaler'].mean_,dtype=torch.float32,device=device)
    scale=torch.as_tensor(fitted['scaler'].scale_,dtype=torch.float32,device=device)
    owners=np.unique(groups[valid]);members=[keep[groups[keep]==s] for s in owners]
    bx,by,bg=bank['values'],bank['labels'],bank['groups']
    if len(bx)!=len(by) or len(bg)!=len(by) or set(by.tolist())!={0,1}:raise ValueError('invalid source training bank')
    source_owners=np.unique(bg)
    source_members={s:{label:np.flatnonzero((bg==s)&(by==label)) for label in np.unique(by[bg==s])} for s in source_owners}
    rng=np.random.default_rng(seed)
    opt=torch.optim.AdamW([{'params':encoder.parameters(),'lr':1e-5},{'params':head.parameters(),'lr':1e-4}],weight_decay=.001)
    history=[];target_losses=[];source_losses=[];started=time.perf_counter()
    for step in range(1,steps+1):
        if time.perf_counter()-started>1800:raise TimeoutError('partition adaptation budget exceeded')
        chosen=[rng.choice(members[s]) for s in rng.integers(len(owners),size=batch_size)]
        pooled=[]
        for i in chosen:
            b=bags[i];idx=np.flatnonzero(b.token_mask)
            inputs=torch.as_tensor(b.values[idx],device=device);masks=torch.as_tensor(b.sample_mask[idx],device=device)
            chunks=[encoder(inputs[j:j+512],masks[j:j+512]) for j in range(0,len(idx),512)]
            z=inputs.new_zeros((len(b.token_mask),16));z[torch.as_tensor(idx,device=device)]=torch.cat(chunks)
            pooled.append(pool_tensor(z,b))
        target_logits=head((torch.stack(pooled)-mean)/scale).squeeze(-1)
        target_loss=nn.functional.binary_cross_entropy_with_logits(target_logits,torch.as_tensor(y[chosen],dtype=torch.float32,device=device))
        source_idx=[]
        for s in rng.choice(source_owners,size=64):
            label=rng.choice(list(source_members[s]));source_idx.append(rng.choice(source_members[s][label]))
        values=torch.as_tensor(bx[source_idx],device=device)
        values=rotate_triads(values,rotation_matrices(len(values),device))
        logits=source(encoder(values,torch.ones(len(values),125,dtype=torch.bool,device=device))).squeeze(-1)
        source_loss=nn.functional.binary_cross_entropy_with_logits(logits,torch.as_tensor(by[source_idx],dtype=torch.float32,device=device))
        loss=target_loss+.25*source_loss
        if not torch.isfinite(loss):raise RuntimeError('nonfinite adaptation loss')
        opt.zero_grad(set_to_none=True);loss.backward()
        grad=control_gradient_norm(list(encoder.parameters())+list(head.parameters()),clip_norm)
        opt.step();target_losses.append(float(target_loss.detach()));source_losses.append(float(source_loss.detach()))
        if resource_check is not None and (step%100==0 or step==steps):resource_check()
        if step%100==0 or step==steps:
            history.append({'step':step,'target_bce':float(np.mean(target_losses)),'source_bce':float(np.mean(source_losses)),
                            'last_gradient_norm':float(grad)});target_losses=[];source_losses=[]
    report={'steps':steps,'batch_size':batch_size,'clip_norm':clip_norm,'sampled_target_candidate_examples':steps*batch_size,
            'fit_subjects':sorted(map(str,owners)),'history':history,'initial_readout':readout_payload(fitted),
            'final_target_head':{k:v.detach().cpu().tolist() for k,v in head.state_dict().items()},
            'encoder':{k:v.detach().cpu().tolist() for k,v in encoder.state_dict().items()},
            'source_train_ids':sorted(map(int,source_owners)),'wall_seconds':time.perf_counter()-started}
    return encoder.requires_grad_(False).eval(),report


def source_training_bank(root,payload,output):
    source_dir=root/'outputs/challenger_70/representation_v2/amendment_a3_source'
    source_plan=json.loads((source_dir/'plan.json').read_text())
    for relative,identity in source_plan['source_files'].items():
        if _file_sha(root/relative)!=identity['sha256']:raise RuntimeError('source raw identity mismatch')
    bank=load_source_bank(FDIAdapter(root/'FDdatasets/doi-10.48804-CN8VBB/v1.0'))
    ids=payload['provenance']['source_train'];keep=np.isin(bank['groups'],ids)
    bank={k:bank[k][keep] for k in ('values','labels','groups')}
    if set(bank['groups'])!=set(ids):raise RuntimeError('source training coverage mismatch')
    head=nn.Linear(16,1);head.load_state_dict(payload['head'],strict=True)
    ep.write_json(output/'source_bank_identity.json',{'source_plan_sha256':hashlib.sha256((source_dir/'plan.json').read_bytes()).hexdigest(),
                  'source_train':ids,'source_validation':payload['provenance']['source_validation'],'train_token_count':len(bank['labels']),
                  'source_raw_files':source_plan['source_files'],'persistent_token_cache_bytes':0})
    return bank,head


def record_replay(root,name,evidence,*,replay=None):
    if replay is None:
        evidence['baseline_source_sha256']=jp.baseline_replay(root,name,evidence)
    else:
        evidence['baseline_validation_sha256']=replay(root,name,evidence)


def prepare(root,payload,output,device,*,adaptation_options=None,replay=None,resource_check=None):
    options=dict(adaptation_options or {})
    if set(options)-{'batch_size','clip_norm'}:raise ValueError('unsupported registered adaptation option')
    bank,head=source_training_bank(root,payload,output)

    def split(root,data,config,train,score,excluded,encoders,source_train,source_val,output,prefix,device):
        if set(train)&(set(score)|set(excluded)) or set(score)&set(excluded):raise ValueError('split subject overlap')
        graph={};t,v=ep.build_candidate_ledgers(data,config,train,score,fit_audit=graph)
        graph['source']={'fit_subjects':sorted(source_train),'parents':[]}
        graph['base_verifier']={'fit_subjects':list(train),'parents':['candidates']}
        scores={'b':ep.verifier_probabilities(t.features,t.labels,v.features,seed=config.seed)}
        tb=ep.unitfree_bags(ep.load_candidate_bags(root,t.candidates,config=ep.PreprocessConfig(gravity_align=False)))
        vb=ep.unitfree_bags(ep.load_candidate_bags(root,v.candidates,config=ep.PreprocessConfig(gravity_align=False)))
        if device=='cuda':torch.cuda.synchronize()
        then=time.perf_counter()
        adapted,report=adapt_encoder(encoders['f'],head,tb,t.labels,t.groups,bank,train_subjects=set(train),seed=20261011,device=device,resource_check=resource_check,**options)
        graph['target_adaptation']={'fit_subjects':report['fit_subjects'],'parents':['source','candidates']}
        for arm,encoder in [('r',encoders['f']),('f',adapted)]:
            tx=np.asarray([pool_joint_ordered(z,b) for z,b in zip(encode_tokens(encoder,tb,device),tb)])
            vx=np.asarray([pool_joint_ordered(z,b) for z,b in zip(encode_tokens(encoder,vb,device),vb)])
            scores[arm]=ordered_joint_probabilities(ordered_joint_features(t.features,tx),t.labels,
                                                   ordered_joint_features(v.features,vx),seed=config.seed)
            graph[arm+'_verifier']={'fit_subjects':list(train),'parents':['candidates','source']+(['target_adaptation'] if arm=='f' else [])}
        graph['score']={'fit_subjects':[],'parents':['base_verifier','r_verifier','f_verifier']}
        ep.check_graph(graph,'score',set(score)|set(excluded)|source_val,set(train)|source_train)
        if device=='cuda':torch.cuda.synchronize()
        seconds=time.perf_counter()-then
        evidence={'train':list(train),'score':list(score),'excluded':sorted(excluded),'graph':graph,
                  'candidates':[asdict(c) for c in v.candidates],'groups':v.groups.tolist(),'labels':v.labels.tolist(),
                  'motion_valid':[bool(b.token_mask.any()) for b in vb], 'scores':{a:p.tolist() for a,p in scores.items()},
                  'feature_widths':{'b':116,'r':251,'f':251},'adaptation_and_encoding_seconds':seconds,
                  'source_checkpoint_sha256':ep.REGISTERED_SOURCE_SHA,'arm_semantics':{'b':'incumbent','r':'frozen FD','f':'adapted FD'}}
        if options:evidence['optimizer_controls']=options
        record_replay(root,prefix+'_scores.json',evidence,replay=replay)
        ep.write_json(output/(prefix+'_scores.json'),evidence);ep.write_json(output/(prefix+'_adaptation.json'),report)
        del tb,vb,adapted,tx,vx;gc.collect()
        return v,scores,graph,seconds
    return split


def run(root):
    jp.run(root,experiment={'id':'RV2-A3-S8-001','label':'S8','output':'amendment_a3_adapt',
           'spec':'challenge_070_amendment_rv2_a3_adapt.md','prepare':prepare,'neural_optimizer_steps':12000,
           'adaptation_readout_fits':12,'max_wall_seconds':21600,'max_work_seconds':10800,
           'elapsed_field':'adaptation_and_encoding_seconds','pass_status':'S8_SUPPORTS_WIDER_INNER_EVENT_VALIDATION',
           'fail_status':'S8_NO_SUPPORTED_TARGET_ADAPTATION'})


if __name__=='__main__':
    with ep.threadpool_limits(limits=4):run(Path(__file__).resolve().parents[3])
