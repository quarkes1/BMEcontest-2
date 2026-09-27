"""Strict nested inner-only event complementarity of frozen FD scores."""
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import warnings

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, log_loss
import torch
from threadpoolctl import threadpool_limits

from scripts.research.representation_v2.crossfit import build_candidate_ledgers, inner_subject_partitions, slice_outer_train
from scripts.research.representation_v2.dataset import PreprocessConfig, load_candidate_bags
from scripts.research.representation_v2.explore import write_json
from scripts.research.representation_v2.low_capacity import encode_candidates, fit_readout, predict_readout, readout_payload
from scripts.research.representation_v2.models import MotionBranch
from scripts.research.representation_v2.provenance import assert_no_heldout_ancestors
from scripts.research.representation_v2.transfer_probe import load_source_artifact, REGISTERED_SOURCE_SHA, unitfree_bags
from src.pipeline import runner
from src.pipeline.candidate_control import CandidateAdmissionConfig, admit_candidates
from src.pipeline.challenger.e03 import registered_e03_config
from src.pipeline.challenger.protection import protected_snapshot
from src.pipeline.event_stack import compute_event_metrics, select_event_threshold


def fuse_scores(base, random, fd):
    arrays=[np.asarray(v,dtype=float) for v in (base,random,fd)]
    if any(v.ndim!=1 or v.shape!=arrays[0].shape or not np.isfinite(v).all() or
           ((v<0)|(v>1)).any() for v in arrays):
        raise ValueError('aligned finite probability vectors required')
    return {'b':arrays[0], 'r':.75*arrays[0]+.25*arrays[1], 'f':.75*arrays[0]+.25*arrays[2]}


def motion_probabilities(x,y,groups,valid,score_x,score_valid):
    if not np.asarray(valid).any(): raise ValueError('no valid train motion candidates')
    fitted=fit_readout(x[valid],y[valid],groups[valid])
    p=np.full(len(score_x),fitted['prior'])
    if np.asarray(score_valid).any(): p[score_valid]=expit(predict_readout(fitted,score_x[score_valid]))
    return p,readout_payload(fitted)


def verifier_probabilities(x,y,v,*,seed):
    if x.shape[1]!=116 or v.shape[1]!=116 or set(y.tolist())!={0,1}:
        raise ValueError('binary 116D verifier input required')
    scores=[]
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        for model in (runner._verifier_estimator(seed+20,.1),runner._verifier_lgbm_estimator(seed+21)):
            model.fit(x,y)
            scores.append(runner._positive_probability(model,v))
    return .5*scores[0]+.5*scores[1]


def check_graph(graph,node,heldout,allowed):
    assert_no_heldout_ancestors(node,set(heldout),graph)
    if any(set(n['fit_subjects'])-set(allowed) for n in graph.values()):
        raise ValueError('unexpected learned ancestor')


def choose_policy(candidates,scores,groups,truths,config):
    choices=[]
    for index,admission in enumerate(CandidateAdmissionConfig(n,t,c)
            for n in config.admission_nms_iou_grid for t in config.admission_threshold_grid
            for c in config.admission_subject_cap_grid):
        rows=np.asarray(admit_candidates(candidates,scores,groups,admission),dtype=int)
        events=[candidates[i].event for i in rows]
        recall=compute_event_metrics(events,truths).sensitivity
        selected=select_event_threshold(events,np.asarray(scores)[rows],truths)
        rank=(selected.metrics.f1,selected.metrics.ppv,-len(rows),-index,-selected.threshold)
        # Final threshold tie prefers higher threshold, as the registered selector does.
        rank=(*rank[:-1],selected.threshold)
        choices.append((recall,rank,{'admission':asdict(admission),'threshold':selected.threshold,
                                    'selection_metrics':asdict(selected.metrics)}))
    feasible=[r for r in choices if r[0]>=config.admission_minimum_recall]
    if not choices: raise ValueError('empty registered admission grid')
    return max(feasible or choices,key=lambda r:r[1] if feasible else (r[0],*r[1]))[2]


def decode_policy(candidates,scores,groups,policy):
    rows=admit_candidates(candidates,scores,groups,CandidateAdmissionConfig(**policy['admission']))
    return [candidates[i].event for i in rows if scores[i]>=policy['threshold']]


def counts_metrics(rows):
    tp=sum(r['n_tp'] for r in rows); pred=sum(r['n_pred'] for r in rows); true=sum(r['n_true'] for r in rows)
    return {'n_tp':tp,'n_pred':pred,'n_true':true,'fp':pred-tp,'fn':true-tp,
            'f1':2*tp/(pred+true) if pred+true else 0.,'ppv':tp/pred if pred else 0.,
            'recall':tp/true if true else 0.}


def event_gate(rows):
    if len(rows)!=3: raise ValueError('three complete health folds required')
    metrics={a:counts_metrics([r['events'][a] for r in rows]) for a in ('b','r','f')}
    b,r,f=(metrics[a] for a in ('b','r','f'))
    wins=sum(counts_metrics([row['events']['f']])['f1']>counts_metrics([row['events']['b']])['f1'] for row in rows)
    passed=(f['f1']-b['f1']>=.015-1e-12 and f['f1']>r['f1'] and wins>=2 and f['n_tp']>=b['n_tp'] and
            (f['fp']<b['fp'] or f['n_tp']-b['n_tp']>f['fp']-b['fp']) and
            all(row.get('checks',{}).get(k) is True for row in rows for k in ('integrity','resource')))
    return {'status':'S2_SUPPORTS_WIDER_INNER_EVENT_VALIDATION' if passed else 'S2_NO_SUPPORTED_FIXED_FUSION',
            'passed':bool(passed),'pooled_health_events':metrics,'positive_folds':int(wins),'outer_calls':0}


def split_scores(root,data,config,train,score,excluded,encoders,source_train,source_val,output,prefix,device):
    """Every learned target ancestor is recreated in this train partition."""
    if set(train)&(set(score)|set(excluded)): raise ValueError('split overlaps heldout subjects')
    graph={}
    t,v=build_candidate_ledgers(data,config,train,score,fit_audit=graph)
    graph['source']={'fit_subjects':sorted(source_train),'parents':[]}
    graph['base_verifier']={'fit_subjects':sorted(set(map(str,t.groups))),'parents':['candidates']}
    base=verifier_probabilities(t.features,t.labels,v.features,seed=config.seed)
    tb=unitfree_bags(load_candidate_bags(root,t.candidates,config=PreprocessConfig(gravity_align=False)))
    vb=unitfree_bags(load_candidate_bags(root,v.candidates,config=PreprocessConfig(gravity_align=False)))
    tv=np.array([b.token_mask.any() for b in tb]); vv=np.array([b.token_mask.any() for b in vb])
    probabilities={}; readouts={}; encode_seconds=0.
    for arm in ('r','f'):
        then=time.perf_counter()
        x=encode_candidates(encoders[arm],tb,np.zeros(6,np.float32),np.ones(6,np.float32),device=device)
        z=encode_candidates(encoders[arm],vb,np.zeros(6,np.float32),np.ones(6,np.float32),device=device)
        encode_seconds+=time.perf_counter()-then
        probabilities[arm],readouts[arm]=motion_probabilities(x,t.labels,t.groups,tv,z,vv)
        graph[arm+'_readout']={'fit_subjects':sorted(set(map(str,t.groups[tv]))),
                              'parents':['candidates']+(['source'] if arm=='f' else [])}
    graph['score']={'fit_subjects':[],'parents':['base_verifier','r_readout','f_readout']}
    check_graph(graph,'score',set(score)|set(excluded)|source_val,set(train)|source_train)
    scores=fuse_scores(base,probabilities['r'],probabilities['f'])
    evidence={'train':list(train),'score':list(score),'excluded':sorted(excluded),'graph':graph,
              'candidates':[asdict(c) for c in v.candidates],'groups':v.groups.tolist(),'labels':v.labels.tolist(),
              'motion_valid':vv.tolist(),'scores':{a:p.tolist() for a,p in scores.items()},
              'component_probabilities':{'base':base.tolist(),**{a:p.tolist() for a,p in probabilities.items()}},
              'readouts':readouts,'micro_threshold':t.micro_threshold,'encoding_seconds':encode_seconds,
              'source_checkpoint_sha256':REGISTERED_SOURCE_SHA}
    write_json(output/f'{prefix}_scores.json',evidence)
    del tb,vb,x,z
    gc.collect()
    return v,scores,graph,encode_seconds


def run(root):
    root=Path(root); output=root/'outputs/challenger_70/representation_v2/amendment_a3_events'
    if output.exists(): raise RuntimeError('S2 attempt already exists; do not overwrite')
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
        raise RuntimeError('commit code and protocol first')
    source=root/'outputs/challenger_70/representation_v2/amendment_a3_source'
    payload=load_source_artifact(source,REGISTERED_SOURCE_SHA)
    source_train={f'FD-R-{p}' for p in payload['provenance']['source_train']}
    source_val={f'FD-R-{p}' for p in payload['provenance']['source_validation']}
    started=time.perf_counter(); snapshot=protected_snapshot(root)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    free_cuda=torch.cuda.mem_get_info()[0] if device=='cuda' else None
    if shutil.disk_usage(root).free<20*1024**3 or (free_cuda is not None and free_cuda<1280*1024**2):
        raise RuntimeError('resource reserve insufficient')
    torch.set_num_threads(4)
    if device=='cuda': torch.cuda.reset_peak_memory_stats()
    plan={'id':'RV2-A3-S2-001','code_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
          'spec_sha256':hashlib.sha256((root/'docs/challenge_070_amendment_rv2_a3_events.md').read_bytes()).hexdigest(),
          'source_checkpoint_sha256':REGISTERED_SOURCE_SHA,'device':device,'outer_calls':0,'neural_optimizer_steps':0,
          'base_fits':120,'baseline_verifier_fits':24,'motion_readout_fits':24,
          'protected_digest':hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest()}
    write_json(output/'plan.json',plan); records=[]; encoding_seconds=0.; completed_splits=0

    def resource_ok():
        peak=torch.cuda.max_memory_allocated() if device=='cuda' else None
        return (time.perf_counter()-started<=10800 and encoding_seconds<=3600 and
                shutil.disk_usage(root).free>=20*1024**3 and (peak is None or peak<=.8*free_cuda))

    try:
        config=registered_e03_config(0,workers=4)
        data=runner.FilesystemDataSource(root).load_outer_train_only(config)
        original_t,original_v=inner_subject_partitions(data,n_splits=4)[0]
        data=slice_outer_train(data,original_t)
        encoders={}
        for arm,key in [('r','random_encoder'),('f','encoder')]:
            encoder=MotionBranch().token_encoder; encoder.load_state_dict(payload[key],strict=True)
            encoder.requires_grad_(False); encoders[arm]=encoder.to(device).eval()
        for fold,(train,health) in enumerate(inner_subject_partitions(data,n_splits=3)):
            tdata=slice_outer_train(data,train); pc=[]; pg=[]; ps={a:[] for a in ('b','r','f')}; policy_graph={}
            for inner,(fit,valid) in enumerate(inner_subject_partitions(tdata,n_splits=3)):
                if not resource_ok(): raise RuntimeError('resource budget exceeded')
                print(f'S2 health{fold} policy{inner}: strict split train{len(fit)}/val{len(valid)}',flush=True)
                ledger,scores,graph,elapsed=split_scores(root,tdata,config,fit,valid,set(health)|set(original_v),
                    encoders,source_train,source_val,output,f'fold{fold}_policy{inner}',device)
                encoding_seconds+=elapsed; completed_splits+=1
                pc.extend(ledger.candidates); pg.extend(ledger.groups)
                for a in ps: ps[a].extend(scores[a])
                for name,node in graph.items():
                    policy_graph[f'p{inner}_{name}']={'fit_subjects':node['fit_subjects'],
                                                    'parents':[f'p{inner}_{p}' for p in node['parents']]}
            if set(pg)!=set(train): raise RuntimeError('policy OOF subject coverage mismatch')
            policy_graph['policy']={'fit_subjects':list(train),'parents':[f'p{i}_score' for i in range(3)]}
            check_graph(policy_graph,'policy',set(health)|set(original_v)|source_val,set(train)|source_train)
            policies={a:choose_policy(pc,np.asarray(ps[a]),pg,tdata.train_truths,config) for a in ps}
            write_json(output/f'fold{fold}_policies.json',{'policies':policies,'graph':policy_graph,
                       'policy_truths':[asdict(t) for t in tdata.train_truths]})
            if not resource_ok(): raise RuntimeError('resource budget exceeded')
            print(f'S2 health{fold}: policies frozen, refit train{len(train)}/health{len(health)}',flush=True)
            ledger,scores,graph,elapsed=split_scores(root,data,config,train,health,set(original_v),encoders,
                source_train,source_val,output,f'fold{fold}_health',device)
            encoding_seconds+=elapsed; completed_splits+=1
            predictions={a:decode_policy(ledger.candidates,scores[a],ledger.groups,policies[a]) for a in ps}
            hdata=slice_outer_train(data,health)
            events={a:asdict(compute_event_metrics(predictions[a],hdata.train_truths)) for a in ps}
            subject_events=[]
            for subject in health:
                sids={sid for sid,s in data.subject_by_session.items() if s==subject}
                truths=[t for t in hdata.train_truths if t.sid in sids]
                subject_events.append({'subject':subject,'events':{a:asdict(compute_event_metrics(
                    [e for e in predictions[a] if e.sid in sids],truths)) for a in ps}})
            record={'fold':fold,'events':events,'subject_events':subject_events,
                    'predictions':{a:[asdict(e) for e in predictions[a]] for a in ps},
                    'truths':[asdict(t) for t in hdata.train_truths],
                    'candidate_metrics':{a:{'ap':float(average_precision_score(ledger.labels,scores[a])),
                                            'bce':float(log_loss(ledger.labels,scores[a],labels=[0,1]))} for a in ps},
                    'checks':{'integrity':True,'resource':bool(resource_ok())},'outer_calls':0}
            write_json(output/f'fold{fold}_report.json',record)
            if not all(record['checks'].values()): raise RuntimeError('health integrity checks failed')
            records.append(record)
            write_json(output/'state.json',{'status':'S2_IN_PROGRESS','completed_health_folds':len(records),'completed_splits':completed_splits})
            print('S2 health event counts '+json.dumps(events),flush=True)
        gate=event_gate(records)
        if protected_snapshot(root)!=snapshot: raise RuntimeError('protected release changed')
        artifact_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if artifact_bytes>2*1024**3 or list(output.rglob('*.pt')) or not resource_ok():
            raise RuntimeError('final resource/artifact budget exceeded')
        gate.update({'wall_seconds':time.perf_counter()-started,'encoding_seconds':encoding_seconds,'completed_splits':completed_splits,
                     'artifact_bytes_before_summary':artifact_bytes,'persistent_token_cache_bytes':0,'copied_checkpoint_bytes':0,
                     'protected_unchanged':True,'neural_optimizer_steps':0})
        write_json(output/'summary.json',gate); write_json(output/'state.json',{'status':gate['status'],'outer_calls':0})
        print(json.dumps(gate),flush=True)
    except Exception as exc:
        write_json(output/'state.json',{'status':'S2_INCOMPLETE','completed_health_folds':len(records),
                   'completed_splits':completed_splits,'error':str(exc),'outer_calls':0})
        raise


if __name__=='__main__':
    with threadpool_limits(limits=4): run(Path(__file__).resolve().parents[3])
