"""S7: frozen token encoders, paired tiny order-aware readouts; inner only."""
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from scipy.special import logit
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from scripts.research.representation_v2.crossfit import build_candidate_ledgers, inner_subject_partitions, slice_outer_train
from scripts.research.representation_v2.dataset import PreprocessConfig, load_candidate_bags
from scripts.research.representation_v2.explore import write_json, _metrics
from scripts.research.representation_v2.health import validate_health_partition, logit_diagnostics
from scripts.research.representation_v2.low_capacity import pool_ordered, fit_readout, predict_readout, readout_payload, subject_weights
from scripts.research.representation_v2.models import MotionBranch, sample_time_coverage, aggregate_time_bins
from scripts.research.representation_v2.provenance import assert_no_heldout_ancestors
from scripts.research.representation_v2.transfer_probe import load_source_artifact, REGISTERED_SOURCE_SHA, unitfree_bags
from src.pipeline.challenger.e03 import registered_e03_config
from src.pipeline.challenger.protection import protected_snapshot
from src.pipeline.runner import FilesystemDataSource


def shuffle_tokens(z, bag):
    result=np.array(z,copy=True)
    seed=int.from_bytes(hashlib.sha256(json.dumps(list(bag.candidate_key)).encode()).digest()[:8],'little')
    idx=np.flatnonzero(bag.token_mask)
    result[idx]=result[np.random.default_rng(seed).permutation(idx)]
    return result


def fit_moments(vectors, bags, groups):
    valid=np.array([b.token_mask.any() for b in bags])
    weights=subject_weights(np.asarray(groups)[valid]); weights=weights/weights.sum()
    first=np.zeros(16); second=np.zeros(16)
    for w,z,b in zip(weights,[z for z,v in zip(vectors,valid) if v], [b for b,v in zip(bags,valid) if v]):
        rows=np.asarray(z,dtype=np.float64)[b.token_mask]
        first+=w*rows.mean(0); second+=w*np.square(rows).mean(0)
    return first,np.maximum(np.sqrt(np.maximum(second-first**2,0)),1e-6)


def bin_tokens(z,bag,center,scale):
    if not len(bag.token_mask): return np.zeros((64,36),np.float32),np.zeros(64,bool)
    x=np.where(bag.token_mask[:,None],(z-center)/scale,0).astype(np.float32)
    intervals=torch.tensor([[bag.start_ms,bag.end_ms]])
    coverage=sample_time_coverage(torch.from_numpy(bag.sample_mask[None]),intervals)
    result,mask=aggregate_time_bins(torch.from_numpy(x[None]),torch.from_numpy(bag.coverage[None]).float(),
                                   torch.from_numpy(bag.token_mask[None]),torch.from_numpy(bag.token_center_ms[None]),
                                   intervals,time_coverage=coverage)
    return result[0].numpy(),mask[0].numpy()


def encode_tokens(encoder,bags,device):
    encoder.to(device).eval()
    if any(p.requires_grad for p in encoder.parameters()): raise ValueError('encoder must be frozen')
    before={k:v.detach().clone() for k,v in encoder.state_dict().items()}
    result=[]
    with torch.inference_mode():
        for b in bags:
            z=np.zeros((len(b.token_mask),16),np.float32)
            idx=np.flatnonzero(b.token_mask)
            for left in range(0,len(idx),512):
                keep=idx[left:left+512]
                z[keep]=encoder(torch.as_tensor(b.values[keep],device=device),
                                torch.as_tensor(b.sample_mask[keep],device=device)).cpu().numpy()
            result.append(z)
    if any(not torch.equal(v,before[k]) for k,v in encoder.state_dict().items()): raise RuntimeError('encoder changed')
    return result


class TemporalReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Conv1d(36,8,1)
        self.temporal=nn.Conv1d(8,8,3,padding=1)
        self.head=nn.Linear(16,1)

    def forward(self,x,mask):
        weight=mask[:,None].to(x.dtype)
        x=torch.relu(self.stem(x.transpose(1,2)*weight))*weight
        x=torch.relu(self.temporal(x))*weight
        mean=x.sum(-1)/weight.sum(-1).clamp_min(1)
        maximum=x.masked_fill(~mask[:,None],float('-inf')).amax(-1)
        maximum=torch.where(mask.any(-1,keepdim=True),maximum,torch.zeros_like(maximum))
        return self.head(torch.cat((mean,maximum),-1)).squeeze(-1)


def fit_temporal(x,mask,y,groups,*,seed,device,steps=2000):
    if not len(y) or set(np.asarray(y).tolist())!={0,1}: raise ValueError('valid training rows need both classes')
    torch.manual_seed(seed)
    if device=='cuda': torch.cuda.manual_seed_all(seed)
    model=TemporalReadout().to(device)
    prior=float(np.average(y,weights=subject_weights(groups)))
    with torch.no_grad(): model.head.bias.fill_(logit(prior))
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    xt=torch.as_tensor(x,device=device); mt=torch.as_tensor(mask,device=device)
    yt=torch.as_tensor(y,dtype=torch.float32,device=device)
    owners=np.unique(groups); members=[np.flatnonzero(groups==s) for s in owners]
    rng=np.random.default_rng(seed); history=[]; losses=[]
    for step in range(1,steps+1):
        idx=np.array([rng.choice(members[s]) for s in rng.integers(len(owners),size=64)])
        opt.zero_grad(set_to_none=True)
        loss=nn.functional.binary_cross_entropy_with_logits(model(xt[idx],mt[idx]),yt[idx])
        if not torch.isfinite(loss): raise RuntimeError('nonfinite head loss')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        losses.append(float(loss.detach()))
        if step%100==0 or step==steps:
            history.append({'step':step,'mean_loss':float(np.mean(losses))});losses=[]
    return model.eval(),history


def order_gate(rows):
    if len(rows)!=3: raise ValueError('three complete folds required')
    means={a:{k:float(np.mean([r['metrics'][a]['all'][k] for r in rows]))
              for k in ('average_precision','raw_logit_bce')} for a in ('r','f','fs','l','prior')}
    if not all(np.isfinite(v) for m in means.values() for v in m.values()): raise ValueError('undefined gate metric')
    ap=lambda a:means[a]['average_precision']
    bce=lambda a:means[a]['raw_logit_bce']
    wins={a:sum(r['metrics']['f']['all']['average_precision']>r['metrics'][a]['all']['average_precision'] for r in rows)
          for a in ('l','fs')}
    integrity=all(r['checks'].get(k) is True for r in rows for k in ('provenance','finite','frozen','optimizer','resource'))
    passed=(integrity and ap('f')>=ap('l')+.02-1e-12 and ap('f')>ap('r') and ap('f')>=ap('fs')+.01-1e-12
            and min(wins.values())>=2 and bce('f')<=bce('l')+.02+1e-12 and bce('f')<=bce('prior')+.02+1e-12)
    return {'status':'S7_SUPPORTS_NESTED_EVENT_PROBE' if passed else 'S7_NO_SUPPORTED_ORDER_READOUT',
            'passed':bool(passed),'macro_metrics':means,'positive_folds':wins,'outer_calls':0}


def run(root):
    root=Path(root);output=root/'outputs/challenger_70/representation_v2/amendment_a3_order'
    if output.exists():raise RuntimeError('S7 attempt exists; no overwrite')
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():raise RuntimeError('commit protocol first')
    payload=load_source_artifact(root/'outputs/challenger_70/representation_v2/amendment_a3_source',REGISTERED_SOURCE_SHA)
    source_train={f'FD-R-{p}' for p in payload['provenance']['source_train']}
    source_val={f'FD-R-{p}' for p in payload['provenance']['source_validation']}
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4)
    if device=='cuda':torch.cuda.reset_peak_memory_stats();free_cuda=torch.cuda.mem_get_info()[0]
    else:free_cuda=None
    if shutil.disk_usage(root).free<20*1024**3 or (free_cuda is not None and free_cuda<1280*1024**2):
        raise RuntimeError('insufficient reserve')
    started=time.perf_counter();snapshot=protected_snapshot(root)
    plan={'id':'RV2-A3-S7-001','code_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
          'spec_sha256':hashlib.sha256((root/'docs/challenge_070_amendment_rv2_a3_order.md').read_bytes()).hexdigest(),
          'source_checkpoint_sha256':REGISTERED_SOURCE_SHA,'device':device,'base_fits':30,'head_fits':9,
          'readout_fits':3,'neural_optimizer_steps':18000,'head_parameters':513,'outer_calls':0,
          'protected_digest':hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest()}
    write_json(output/'plan.json',plan)
    plan_sha=hashlib.sha256((output/'plan.json').read_bytes()).hexdigest()
    records=[];gpu_seconds=0.
    try:
        config=registered_e03_config(0,workers=4)
        data=FilesystemDataSource(root).load_outer_train_only(config)
        original_t,original_v=inner_subject_partitions(data,n_splits=4)[0]
        data=slice_outer_train(data,original_t)
        encoders={}
        for arm,key in [('r','random_encoder'),('f','encoder')]:
            e=MotionBranch().token_encoder;e.load_state_dict(payload[key],strict=True)
            for p in e.parameters():p.requires_grad_(False)
            encoders[arm]=e.to(device).eval()
        for fold,(train,health) in enumerate(inner_subject_partitions(data,n_splits=3)):
            validate_health_partition(original_t,original_v,train,health)
            if time.perf_counter()-started>10800:raise TimeoutError('wall budget exceeded')
            print(f'S7 fold{fold}: T-only producers',flush=True)
            graph={};t,h=build_candidate_ledgers(data,config,train,health,fit_audit=graph)
            tb=unitfree_bags(load_candidate_bags(root,t.candidates,config=PreprocessConfig(gravity_align=False)))
            hb=unitfree_bags(load_candidate_bags(root,h.candidates,config=PreprocessConfig(gravity_align=False)))
            tv=np.array([b.token_mask.any() for b in tb]);hv=np.array([b.token_mask.any() for b in hb])
            owners=sorted(set(map(str,t.groups[tv])))
            graph['fd_encoder']={'fit_subjects':sorted(source_train),'parents':[]}
            graph['random_encoder']={'fit_subjects':[],'parents':[]}
            for arm in ('r','f','fs','l'):
                parent='random_encoder' if arm=='r' else 'fd_encoder'
                graph[f'norm_{arm}']={'fit_subjects':owners,'parents':[parent,'candidates']}
                graph[f'head_{arm}']={'fit_subjects':owners,'parents':[f'norm_{arm}']}
                assert_no_heldout_ancestors(f'head_{arm}',set(health)|set(original_v)|source_val,graph)
            if any(set(n['fit_subjects'])-(set(train)|source_train) for n in graph.values()):raise RuntimeError('unexpected ancestor')
            write_json(output/f'fold{fold}_provenance.json',graph)
            write_json(output/f'fold{fold}_subjects.json',{'train':list(train),'health':list(health),'excluded_original_validation':list(original_v)})
            rows=[{'sid':c.event.sid,'start_ms':int(c.event.start_ms),'end_ms':int(c.event.end_ms),
                   'subject':str(g),'label':int(y),'valid':bool(v)} for c,g,y,v in zip(h.candidates,h.groups,h.labels,hv)]
            reference=root/f'outputs/challenger_70/representation_v2/amendment_a3_target/fold{fold}_report.json'
            old=json.loads(reference.read_text())
            if rows!=old['rows']:raise RuntimeError('S1 candidate identity/label/valid mismatch')
            prior=float(np.average(t.labels[tv],weights=subject_weights(t.groups[tv])))
            zprior=np.full(len(hb),logit(prior));scores={'prior':zprior.tolist()};metrics={'prior':_metrics(h.labels,zprior,hv)}
            models={};histories={};convergence={};norms={}
            for encoder_arm in ('r','f'):
                if device=='cuda':torch.cuda.synchronize()
                then=time.perf_counter()
                tz=encode_tokens(encoders[encoder_arm],tb,device);hz=encode_tokens(encoders[encoder_arm],hb,device)
                if device=='cuda':torch.cuda.synchronize()
                gpu_seconds+=time.perf_counter()-then
                center,scale=fit_moments(tz,tb,t.groups);norms[encoder_arm]={'center':center.tolist(),'scale':scale.tolist()}
                if encoder_arm=='f':
                    tx=np.array([pool_ordered(z,b) if len(b.token_mask) else np.zeros(133) for z,b in zip(tz,tb)])
                    hx=np.array([pool_ordered(z,b) if len(b.token_mask) else np.zeros(133) for z,b in zip(hz,hb)])
                    fitted=fit_readout(tx[tv],t.labels[tv],t.groups[tv]);lz=zprior.copy();lz[hv]=predict_readout(fitted,hx[hv])
                    scores['l']=lz.tolist();metrics['l']=_metrics(h.labels,lz,hv);models['l']=readout_payload(fitted)
                for arm in (('r',) if encoder_arm=='r' else ('f','fs')):
                    def binned(zs,bs):
                        pairs=[bin_tokens(shuffle_tokens(z,b) if arm=='fs' else z,b,center,scale) for z,b in zip(zs,bs)]
                        return np.stack([p[0] for p in pairs]),np.stack([p[1] for p in pairs])
                    x,m=binned(tz,tb);v,vm=binned(hz,hb)
                    if device=='cuda':torch.cuda.synchronize()
                    then=time.perf_counter()
                    model,hist=fit_temporal(x[tv],m[tv],t.labels[tv],t.groups[tv],seed=20261010+fold,device=device)
                    z=zprior.copy()
                    with torch.inference_mode():
                        z[hv]=model(torch.as_tensor(v[hv],device=device),torch.as_tensor(vm[hv],device=device)).cpu().numpy()
                    if device=='cuda':torch.cuda.synchronize()
                    gpu_seconds+=time.perf_counter()-then
                    tail=[r['mean_loss'] for r in hist[-4:]]
                    unresolved=np.mean(tail[2:])<.9*np.mean(tail[:2])
                    convergence[arm]={'unresolved':bool(unresolved),'last400_loss':tail}
                    histories[arm]=hist;models[arm]={k:w.detach().cpu().tolist() for k,w in model.state_dict().items()}
                    scores[arm]=z.tolist();metrics[arm]=_metrics(h.labels,z,hv)
                    print(f'S7 fold{fold} {arm}: AP={metrics[arm]["all"]["average_precision"]:.6f}',flush=True)
                    del model,x,m,v,vm
                del tz,hz;gc.collect()
            peak=torch.cuda.max_memory_allocated() if device=='cuda' else None
            checks={'provenance':True,'frozen':True,'finite':all(np.isfinite(z).all() for z in scores.values()),
                    'optimizer':all(not c['unresolved'] for c in convergence.values()),
                    'resource':gpu_seconds<=3600 and time.perf_counter()-started<=10800 and shutil.disk_usage(root).free>=20*1024**3
                               and (peak is None or peak<=.8*free_cuda)}
            subjects=[{'subject':str(s),'arm':a,'metrics':logit_diagnostics(h.labels[h.groups==s],np.asarray(z)[h.groups==s])}
                      for s in sorted(set(h.groups)) for a,z in scores.items()]
            record={'fold':fold,'rows':rows,'metrics':metrics,'logits':scores,'models':models,'histories':histories,
                    'normalizers':norms,'convergence':convergence,'checks':checks,'subject_metrics':subjects,
                    'reference_sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),'plan_sha256':plan_sha,
                    'train_count':len(tb),'train_valid_count':int(tv.sum()),'peak_cuda_bytes':peak,'outer_calls':0}
            write_json(output/f'fold{fold}_report.json',record);records.append(record)
            write_json(output/'state.json',{'status':'S7_IN_PROGRESS','completed_folds':len(records),'outer_calls':0})
            if not checks['finite'] or not checks['resource']:raise RuntimeError('integrity or resource check failed')
            del tb,hb;gc.collect()
        gate=order_gate(records)
        if protected_snapshot(root)!=snapshot:raise RuntimeError('protected release changed')
        size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if size>16*1024**2:raise RuntimeError('artifact budget exceeded')
        gate.update({'wall_seconds':time.perf_counter()-started,'encoding_and_head_seconds':gpu_seconds,
                     'artifact_bytes_before_summary':size,'persistent_token_cache_bytes':0,'copied_checkpoint_bytes':0,
                     'neural_optimizer_steps':18000,'protected_unchanged':True})
        write_json(output/'summary.json',gate);write_json(output/'state.json',{'status':gate['status'],'outer_calls':0})
        print(json.dumps(gate,sort_keys=True),flush=True)
    except Exception as exc:
        write_json(output/'state.json',{'status':'S7_INCOMPLETE','error':str(exc),'completed_folds':len(records),'outer_calls':0})
        raise


if __name__=='__main__':
    with threadpool_limits(limits=4):run(Path(__file__).resolve().parents[3])
