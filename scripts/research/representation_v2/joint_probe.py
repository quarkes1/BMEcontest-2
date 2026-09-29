"""S5 frozen FD/random features jointly interpreted by a bounded verifier."""
import warnings
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning

from scripts.research.representation_v2.models import sample_time_coverage
from src.pipeline import runner
from scripts.research.representation_v2 import event_probe as ep
from scripts.research.representation_v2.neutral_probe import load_frozen_policies, assert_baseline_replay


SOURCE_COMMIT='46d7c18ab11da2da025934a73f3c273e68c2aabb'


def baseline_replay(root,name,evidence):
    # Dataclass probability tuples become JSON arrays in the sealed ledger.
    evidence=json.loads(json.dumps(evidence,allow_nan=False))
    path='outputs/challenger_70/representation_v2/amendment_a3_events/'+name
    raw=subprocess.check_output(['git','show',f'{SOURCE_COMMIT}:{path}'],cwd=root)
    old=json.loads(raw)
    if name.endswith('_scores.json'):
        for key in ('train','score','excluded','candidates','groups','labels'):
            if evidence[key]!=old[key]:raise RuntimeError('baseline ledger differs: '+key)
        np.testing.assert_allclose(evidence['scores']['b'],old['scores']['b'],rtol=0,atol=1e-12)
    elif name.endswith('_policies.json'):
        if evidence['policies']['b']!=old['policies']['b']:raise RuntimeError('baseline policy differs')
    else:
        assert_baseline_replay(evidence,old)
    return hashlib.sha256(raw).hexdigest()


def pool_joint(embeddings, bag):
    z=np.asarray(embeddings)
    mask=np.asarray(bag.token_mask)
    if mask.dtype!=bool or z.shape!=(len(mask),16) or not np.isfinite(z).all():
        raise ValueError('finite aligned 16D token embeddings and boolean validity required')
    valid=z[mask]
    moments=np.concatenate([valid.mean(0),valid.std(0)]) if len(valid) else np.full(32,np.nan)
    if bag.end_ms<=bag.start_ms or bag.sample_mask.shape!=(len(mask),125):
        raise ValueError('positive interval and aligned 25Hz sample mask required')
    coverage=(sample_time_coverage(torch.from_numpy(bag.sample_mask[None]),
                                  torch.tensor([[bag.start_ms,bag.end_ms]]),bins=4)[0].numpy()
              if len(mask) else np.zeros(4))
    quality=np.r_[coverage,float(mask.mean()) if len(mask) else 0.,float(mask.any())]
    if not np.isfinite(quality).all():raise ValueError('finite quality features required')
    return np.r_[moments,quality]


def joint_features(base,motion):
    base,motion=np.asarray(base),np.asarray(motion)
    if (base.ndim!=2 or base.shape[1]!=116 or motion.shape!=(len(base),38) or
            np.isinf(base).any() or np.isinf(motion).any() or not np.isfinite(motion[:,32:]).all()):
        raise ValueError('aligned 116D base and 38D motion required; no infinite values')
    return np.concatenate([base,motion],axis=1)


def encode_joint(encoder,bags,*,device='cpu',chunk=512):
    if chunk<1:raise ValueError('positive chunk size required')
    if any(p.requires_grad for p in encoder.parameters()):raise ValueError('encoder must be frozen')
    encoder.to(device).eval()
    rows=[]
    with torch.inference_mode():
        for bag in bags:
            z=np.zeros((len(bag.token_mask),16),np.float32)
            indices=np.flatnonzero(bag.token_mask)
            for left in range(0,len(indices),chunk):
                idx=indices[left:left+chunk]; mask=bag.sample_mask[idx]
                x=np.where(mask[:,None,:],bag.values[idx],0.)
                z[idx]=encoder(torch.as_tensor(x,dtype=torch.float32,device=device),
                               torch.as_tensor(mask,device=device)).cpu().numpy()
            rows.append(pool_joint(z,bag))
    return np.asarray(rows).reshape(-1,38)


def joint_probabilities(x,y,v,*,seed):
    x,y,v=np.asarray(x),np.asarray(y),np.asarray(v)
    if (x.ndim!=2 or v.ndim!=2 or x.shape[1]!=154 or v.shape[1]!=154 or
            y.shape!=(len(x),) or set(y.tolist())!={0,1} or np.isinf(x).any() or np.isinf(v).any()):
        raise ValueError('binary aligned 154D verifier input required')
    probabilities=[]
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        for model in (runner._verifier_estimator(seed+20,.1),runner._verifier_lgbm_estimator(seed+21)):
            model.fit(x,y)
            probabilities.append(runner._positive_probability(model,v))
    result=.5*probabilities[0]+.5*probabilities[1]
    if not np.isfinite(result).all() or ((result<0)|(result>1)).any():
        raise ValueError('verifier produced invalid probabilities')
    return result


def split_scores(root,data,config,train,score,excluded,encoders,source_train,source_val,output,prefix,device):
    if set(train)&(set(score)|set(excluded)) or set(score)&set(excluded):
        raise ValueError('split overlaps heldout subjects')
    graph={}
    t,v=ep.build_candidate_ledgers(data,config,train,score,fit_audit=graph)
    graph['source']={'fit_subjects':sorted(source_train),'parents':[]}
    graph['base_verifier']={'fit_subjects':list(train),'parents':['candidates']}
    scores={'b':ep.verifier_probabilities(t.features,t.labels,v.features,seed=config.seed)}
    tb=ep.unitfree_bags(ep.load_candidate_bags(root,t.candidates,config=ep.PreprocessConfig(gravity_align=False)))
    vb=ep.unitfree_bags(ep.load_candidate_bags(root,v.candidates,config=ep.PreprocessConfig(gravity_align=False)))
    elapsed=0.
    for arm in ('r','f'):
        then=time.perf_counter()
        x=encode_joint(encoders[arm],tb,device=device); z=encode_joint(encoders[arm],vb,device=device)
        elapsed+=time.perf_counter()-then
        scores[arm]=joint_probabilities(joint_features(t.features,x),t.labels,joint_features(v.features,z),seed=config.seed)
        graph[arm+'_verifier']={'fit_subjects':list(train),'parents':['candidates']+(['source'] if arm=='f' else [])}
    graph['score']={'fit_subjects':[],'parents':['base_verifier','r_verifier','f_verifier']}
    ep.check_graph(graph,'score',set(score)|set(excluded)|source_val,set(train)|source_train)
    evidence={'train':list(train),'score':list(score),'excluded':sorted(excluded),'graph':graph,
              'candidates':[asdict(c) for c in v.candidates],'groups':v.groups.tolist(),'labels':v.labels.tolist(),
              'motion_valid':[bool(b.token_mask.any()) for b in vb],
              'scores':{a:p.tolist() for a,p in scores.items()},'feature_widths':{'b':116,'r':154,'f':154},
              'encoding_seconds':elapsed,'source_checkpoint_sha256':ep.REGISTERED_SOURCE_SHA}
    evidence['baseline_source_sha256']=baseline_replay(root,prefix+'_scores.json',evidence)
    ep.write_json(output/(prefix+'_scores.json'),evidence)
    del tb,vb,x,z
    gc.collect()
    return v,scores,graph,elapsed


def run(root):
    root=Path(root); output=root/'outputs/challenger_70/representation_v2/amendment_a3_joint'
    if output.exists():raise RuntimeError('S5 attempt exists; do not overwrite')
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
        raise RuntimeError('commit protocol/code first')
    payload=ep.load_source_artifact(root/'outputs/challenger_70/representation_v2/amendment_a3_source',ep.REGISTERED_SOURCE_SHA)
    source_train={f'FD-R-{p}' for p in payload['provenance']['source_train']}
    source_val={f'FD-R-{p}' for p in payload['provenance']['source_validation']}
    started=time.perf_counter(); snapshot=ep.protected_snapshot(root)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    free_cuda=torch.cuda.mem_get_info()[0] if device=='cuda' else None
    if shutil.disk_usage(root).free<20*1024**3 or (free_cuda is not None and free_cuda<1280*1024**2):
        raise RuntimeError('resource reserve insufficient')
    torch.set_num_threads(4)
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    plan={'id':'RV2-A3-S5-001','code_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
          'spec_sha256':hashlib.sha256((root/'docs/challenge_070_amendment_rv2_a3_joint.md').read_bytes()).hexdigest(),
          'source_checkpoint_sha256':ep.REGISTERED_SOURCE_SHA,'device':device,'outer_calls':0,
          'neural_optimizer_steps':0,'producer_fits':120,'verifier_fits':72,
          'protected_digest':hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest()}
    ep.write_json(output/'plan.json',plan)
    records=[]; elapsed=0.; completed=0

    def resource_ok():
        peak=torch.cuda.max_memory_allocated() if device=='cuda' else None
        size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        return (time.perf_counter()-started<=10800 and elapsed<=3600 and size<=64*1024**2 and
                shutil.disk_usage(root).free>=20*1024**3 and (peak is None or peak<=.8*free_cuda))

    try:
        config=ep.registered_e03_config(0,workers=4)
        data=runner.FilesystemDataSource(root).load_outer_train_only(config)
        original_t,original_v=ep.inner_subject_partitions(data,n_splits=4)[0]
        data=ep.slice_outer_train(data,original_t)
        encoders={}
        for arm,key in [('r','random_encoder'),('f','encoder')]:
            encoder=ep.MotionBranch().token_encoder; encoder.load_state_dict(payload[key],strict=True)
            encoders[arm]=encoder.requires_grad_(False).to(device).eval()
        partitions=list(ep.inner_subject_partitions(data,n_splits=3))
        policy_hashes={}
        # Complete ALL policy selection before any health scoring or reference reading.
        for fold,(train,health) in enumerate(partitions):
            tdata=ep.slice_outer_train(data,train); pc=[]; pg=[]; ps={a:[] for a in ('b','r','f')}; graph={}
            for inner,(fit,valid) in enumerate(ep.inner_subject_partitions(tdata,n_splits=3)):
                if not resource_ok():raise RuntimeError('resource budget exceeded')
                print(f'S5 policy {fold}/{inner}: train {len(fit)}, validation {len(valid)}',flush=True)
                ledger,scores,g,seconds=split_scores(root,tdata,config,fit,valid,set(health)|set(original_v),
                    encoders,source_train,source_val,output,f'fold{fold}_policy{inner}',device)
                elapsed+=seconds; completed+=1; pc.extend(ledger.candidates); pg.extend(ledger.groups)
                for a in ps:ps[a].extend(scores[a])
                for name,node in g.items():
                    graph[f'p{inner}_{name}']={'fit_subjects':node['fit_subjects'],'parents':[f'p{inner}_{p}' for p in node['parents']]}
            if set(pg)!=set(train):raise RuntimeError('policy OOF coverage mismatch')
            graph['policy']={'fit_subjects':list(train),'parents':[f'p{i}_score' for i in range(3)]}
            ep.check_graph(graph,'policy',set(health)|set(original_v)|source_val,set(train)|source_train)
            policies={a:ep.choose_policy(pc,np.asarray(ps[a]),pg,tdata.train_truths,config) for a in ps}
            value={'fold':fold,'policies':policies,'graph':graph}
            value['baseline_source_sha256']=baseline_replay(root,f'fold{fold}_policies.json',value)
            path=output/f'fold{fold}_policy_selection.json'; ep.write_json(path,value)
            policy_hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        marker=output/'policy_phase_frozen.json'
        ep.write_json(marker,{'folds':[0,1,2],'health_files_read':0,'policy_file_sha256':policy_hashes})
        marker_sha=hashlib.sha256(marker.read_bytes()).hexdigest()
        for fold,(train,health) in enumerate(partitions):
            policies=load_frozen_policies(output,marker_sha)[fold]['policies']
            if not resource_ok():raise RuntimeError('resource budget exceeded')
            print(f'S5 health {fold}: all policies verified; refitting',flush=True)
            ledger,scores,graph,seconds=split_scores(root,data,config,train,health,set(original_v),encoders,
                source_train,source_val,output,f'fold{fold}_health',device)
            elapsed+=seconds; completed+=1
            hdata=ep.slice_outer_train(data,health)
            predictions={a:ep.decode_policy(ledger.candidates,scores[a],ledger.groups,policies[a]) for a in scores}
            events={a:asdict(ep.compute_event_metrics(predictions[a],hdata.train_truths)) for a in scores}
            subject_events=[]
            for subject in health:
                sids={sid for sid,s in data.subject_by_session.items() if s==subject}
                truths=[t for t in hdata.train_truths if t.sid in sids]
                subject_events.append({'subject':subject,'events':{a:asdict(ep.compute_event_metrics(
                    [e for e in predictions[a] if e.sid in sids],truths)) for a in scores}})
            record={'fold':fold,'events':events,'subject_events':subject_events,
                    'predictions':{a:[asdict(e) for e in predictions[a]] for a in scores},
                    'truths':[asdict(t) for t in hdata.train_truths],
                    'candidate_metrics':{a:{'ap':float(ep.average_precision_score(ledger.labels,scores[a])),
                                            'bce':float(ep.log_loss(ledger.labels,scores[a],labels=[0,1]))} for a in scores},
                    'checks':{'integrity':True,'resource':bool(resource_ok())},'outer_calls':0}
            record['baseline_source_sha256']=baseline_replay(root,f'fold{fold}_report.json',record)
            ep.write_json(output/f'fold{fold}_report.json',record); records.append(record)
            if not all(record['checks'].values()):raise RuntimeError('health checks failed')
            print('S5 health counts '+json.dumps(events),flush=True)
        gate=ep.event_gate(records)
        gate['status']='S5_SUPPORTS_WIDER_INNER_EVENT_VALIDATION' if gate['passed'] else 'S5_NO_SUPPORTED_JOINT_VERIFIER'
        if ep.protected_snapshot(root)!=snapshot:raise RuntimeError('protected release changed')
        if not resource_ok():raise RuntimeError('final resource budget exceeded')
        gate.update({'wall_seconds':time.perf_counter()-started,'encoding_seconds':elapsed,'completed_splits':completed,
                     'protected_unchanged':True,'neural_optimizer_steps':0,'persistent_token_cache_bytes':0,
                     'copied_checkpoint_bytes':0,'policy_marker_sha256':marker_sha})
        ep.write_json(output/'summary.json',gate); ep.write_json(output/'state.json',{'status':gate['status'],'outer_calls':0})
        print(json.dumps(gate),flush=True)
    except Exception as exc:
        ep.write_json(output/'state.json',{'status':'S5_INCOMPLETE','completed_splits':completed,
                                         'completed_health_folds':len(records),'error':str(exc),'outer_calls':0})
        raise


if __name__=='__main__':
    with ep.threadpool_limits(limits=4):run(Path(__file__).resolve().parents[3])
