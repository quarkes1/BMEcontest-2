"""A3 S1: target-only readouts on paired frozen random/FD encoders."""
from dataclasses import replace
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
from threadpoolctl import threadpool_limits

from scripts.research.representation_v2.crossfit import build_candidate_ledgers, inner_subject_partitions, slice_outer_train
from scripts.research.representation_v2.dataset import PreprocessConfig, load_candidate_bags
from scripts.research.representation_v2.explore import write_json, _metrics
from scripts.research.representation_v2.health import validate_health_partition, logit_diagnostics
from scripts.research.representation_v2.low_capacity import encode_candidates, fit_readout, predict_readout, readout_payload, subject_weights
from scripts.research.representation_v2.models import MotionBranch
from scripts.research.representation_v2.provenance import assert_no_heldout_ancestors
from scripts.research.representation_v2.unitfree import normalize_triads
from src.pipeline.challenger.e03 import registered_e03_config
from src.pipeline.challenger.protection import protected_snapshot
from src.pipeline.runner import FilesystemDataSource


REGISTERED_SOURCE_SHA = 'c6276837b1366117075e7e8a7504e084926cbfc5d80b98b863e442255995efc4'


def load_source_artifact(source, expected_sha):
    source=Path(source)
    try:
        plan=json.loads((source/'plan.json').read_text())
        report=json.loads((source/'summary.json').read_text())
        digest=hashlib.sha256((source/'source_final.pt').read_bytes()).hexdigest()
        if digest!=expected_sha or digest!=report['checkpoint_sha256']:
            raise ValueError('source checkpoint identity mismatch')
        if (plan['id']!='RV2-A3-S0-001' or plan['steps']!=5000 or
                plan['target_signal_read'] is not False or plan['outer_calls']!=0):
            raise ValueError('source plan contract invalid')
        if (report['status']!='S0_SUPPORTS_TARGET_PROBE' or report['passed'] is not True or
                report['target_signal_read'] is not False or report['outer_calls']!=0 or
                any(report['checks'].get(k) is not True for k in ('finite','isolation','optimizer','resource'))):
            raise ValueError('source gate contract invalid')
        payload=torch.load(source/'source_final.pt',map_location='cpu',weights_only=True)
        identity=payload['provenance']
        if (payload['schema']!='rv2-a3-unitfree-source-v1' or identity['target_subjects']!=[] or
                identity['normalization']!='per-token-per-triad-demean-rms' or
                identity['steps']!=plan['steps'] or identity['code_sha']!=plan['code_sha'] or
                identity['plan_sha256']!=hashlib.sha256((source/'plan.json').read_bytes()).hexdigest()):
            raise ValueError('source provenance contract invalid')
        train=identity['source_train']; validation=identity['source_validation']
        if (train!=plan['source_train'] or validation!=plan['source_validation'] or
                len(set(train))!=27 or len(set(validation))!=7 or
                set(train)&set(validation) or set(train)|set(validation)!=set(range(34))):
            raise ValueError('source participant contract invalid')
        for key,module in [('encoder',MotionBranch().token_encoder),
                           ('random_encoder',MotionBranch().token_encoder),('head',torch.nn.Linear(16,1))]:
            state=payload[key]
            if not isinstance(state,dict) or any(not isinstance(v,torch.Tensor) or not torch.isfinite(v).all() for v in state.values()):
                raise ValueError('invalid source weights')
            module.load_state_dict(state,strict=True)
        return payload
    except (KeyError,TypeError,RuntimeError) as exc:
        raise ValueError('source artifact contract invalid') from exc


def unitfree_bags(bags):
    result=[]
    for bag in bags:
        values,valid=normalize_triads(bag.values,bag.sample_mask)
        result.append(replace(bag,values=values,token_mask=bag.token_mask & valid))
    return tuple(result)


def transfer_gate(rows):
    if len(rows)!=3: raise ValueError('three complete target folds required')
    means={}
    for arm in ('m','r','f','prior'):
        means[arm]={}
        for key in ('average_precision','raw_logit_bce'):
            values=[r['metrics'][arm]['all'][key] for r in rows]
            if any(v is None or not np.isfinite(v) for v in values): raise ValueError('undefined target gate metric')
            means[arm][key]=float(np.mean(values))
    gain=means['f']['average_precision']-means['r']['average_precision']
    wins=sum(r['metrics']['f']['all']['average_precision']>r['metrics']['r']['all']['average_precision'] for r in rows)
    checks=all(row.get('checks',{}).get(key) is True for row in rows for key in ('provenance','finite','optimizer','resource'))
    passed=(checks and gain>=.02-1e-12 and wins>=2 and means['f']['average_precision']>means['m']['average_precision'] and
            means['f']['raw_logit_bce']<=means['r']['raw_logit_bce']+.02+1e-12 and
            means['f']['raw_logit_bce']<=means['prior']['raw_logit_bce']+.02+1e-12)
    return {'status':'S1_SUPPORTS_WIDER_INNER_PROBE' if passed else 'S1_NO_SUPPORTED_TRANSFER','passed':bool(passed),
            'macro_metrics':means,'ap_gain_f_vs_r':gain,'positive_folds':int(wins),'outer_calls':0}


def run(root):
    root=Path(root); output=root/'outputs/challenger_70/representation_v2/amendment_a3_target'
    if (output/'plan.json').exists(): raise RuntimeError('target attempt exists; no overwrite')
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
        raise RuntimeError('commit target protocol and code first')
    source=root/'outputs/challenger_70/representation_v2/amendment_a3_source'
    payload=load_source_artifact(source,REGISTERED_SOURCE_SHA)
    weight_sha=REGISTERED_SOURCE_SHA
    identity=payload['provenance']
    if identity['target_subjects'] or identity['plan_sha256']!=hashlib.sha256((source/'plan.json').read_bytes()).hexdigest():
        raise RuntimeError('source ancestry or plan identity invalid')
    source_train={f'FD-R-{p}' for p in identity['source_train']}
    source_val={f'FD-R-{p}' for p in identity['source_validation']}
    if not source_train or source_train & source_val: raise RuntimeError('source participant overlap')
    started=time.perf_counter(); snapshot=protected_snapshot(root)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    free_cuda=torch.cuda.mem_get_info()[0] if device=='cuda' else None
    if free_cuda is not None and free_cuda<1280*1024**2: raise RuntimeError('GPU reserve insufficient')
    if shutil.disk_usage(root).free<20*1024**3: raise RuntimeError('disk reserve insufficient')
    torch.set_num_threads(4)
    if device=='cuda': torch.cuda.reset_peak_memory_stats()
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    plan={'id':'RV2-A3-S1-001','code_sha':sha,'source_checkpoint_sha256':weight_sha,
          'source_plan_sha256':identity['plan_sha256'],'source_train':sorted(source_train),'source_validation':sorted(source_val),
          'spec_sha256':hashlib.sha256((root/'docs/challenge_070_amendment_rv2_a3_target.md').read_bytes()).hexdigest(),
          'device':device,'base_fits':30,'readout_fits':9,'neural_optimizer_steps':0,'outer_calls':0,
          'protected_digest':hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest()}
    write_json(output/'plan.json',plan); plan_sha=hashlib.sha256((output/'plan.json').read_bytes()).hexdigest()
    write_json(output/'state.json',{'status':'S1_IN_PROGRESS','outer_calls':0})
    records=[]; encoding_seconds=0.
    try:
        config=registered_e03_config(0,workers=4)
        data=FilesystemDataSource(root).load_outer_train_only(config)
        original_t,original_v=inner_subject_partitions(data,n_splits=4)[0]
        data=slice_outer_train(data,original_t)
        encoders={}
        for arm,key in [('r','random_encoder'),('f','encoder')]:
            encoder=MotionBranch().token_encoder
            encoder.load_state_dict(payload[key],strict=True)
            for parameter in encoder.parameters(): parameter.requires_grad_(False)
            encoders[arm]=encoder.to(device).eval()
        for fold,(train,health) in enumerate(inner_subject_partitions(data,n_splits=3)):
            if time.perf_counter()-started>10800: raise TimeoutError('target wall budget incomplete')
            validate_health_partition(original_t,original_v,train,health)
            graph={}
            print(f'S1 fold{fold}: regenerating T-only candidates',flush=True)
            t,h=build_candidate_ledgers(data,config,train,health,fit_audit=graph)
            graph['fd_encoder']={'fit_subjects':sorted(source_train),'parents':[]}
            graph['random_encoder']={'fit_subjects':[],'parents':[]}
            graph['unitfree_input']={'fit_subjects':[],'parents':['candidates']}
            raw_config=PreprocessConfig(gravity_align=False)
            tb=unitfree_bags(load_candidate_bags(root,t.candidates,config=raw_config))
            hb=unitfree_bags(load_candidate_bags(root,h.candidates,config=raw_config))
            tv=np.array([b.token_mask.any() for b in tb]); hv=np.array([b.token_mask.any() for b in hb])
            if not tv.any() or not hv.any(): raise RuntimeError('no nondegenerate target input')
            owners=sorted(set(map(str,t.groups[tv])))
            for arm,parent in [('r','random_encoder'),('f','fd_encoder'),('m','unitfree_input')]:
                graph[f'readout_{arm}']={'fit_subjects':owners,'parents':['unitfree_input',parent]}
                assert_no_heldout_ancestors(f'readout_{arm}',set(health)|set(original_v)|source_val,graph)
            if any(set(n['fit_subjects'])-(set(train)|source_train) for n in graph.values()):
                raise RuntimeError('unexpected learned ancestor')
            write_json(output/f'fold{fold}_provenance.json',graph)
            write_json(output/f'fold{fold}_subjects.json',{'train':list(map(str,train)),'health':list(map(str,health)),
                                                        'excluded_original_validation':list(map(str,original_v))})
            prior=float(np.average(t.labels[tv],weights=subject_weights(t.groups[tv])))
            zprior=np.full(len(hb),logit(prior))
            metrics={'prior':_metrics(h.labels,zprior,hv)}; scores={'prior':zprior.tolist()}; models={}

            def score_arm(arm,x,v):
                fitted=fit_readout(x[tv],t.labels[tv],t.groups[tv])
                z=zprior.copy(); z[hv]=predict_readout(fitted,v[hv])
                metrics[arm]=_metrics(h.labels,z,hv); scores[arm]=z.tolist(); models[arm]=readout_payload(fitted)

            for arm in ('r','f'):
                then=time.perf_counter()
                x=encode_candidates(encoders[arm],tb,np.zeros(6,np.float32),np.ones(6,np.float32),device=device)
                v=encode_candidates(encoders[arm],hb,np.zeros(6,np.float32),np.ones(6,np.float32),device=device)
                encoding_seconds+=time.perf_counter()-then
                if arm=='r': score_arm('m',x[:,-5:],v[:,-5:])
                score_arm(arm,x,v)
            peak=torch.cuda.max_memory_allocated() if device=='cuda' else None
            checks={'provenance':True,'finite':all(np.isfinite(z).all() for z in scores.values()),'optimizer':True,
                    'resource':encoding_seconds<=3600 and time.perf_counter()-started<=10800 and
                               shutil.disk_usage(root).free>=20*1024**3 and (peak is None or peak<=.8*free_cuda)}
            rows=[{'sid':c.event.sid,'start_ms':int(c.event.start_ms),'end_ms':int(c.event.end_ms),
                   'subject':str(g),'label':int(y),'valid':bool(valid)}
                  for c,g,y,valid in zip(h.candidates,h.groups,h.labels,hv)]
            subject_metrics=[]
            for subject in sorted(set(h.groups)):
                keep=h.groups==subject
                for arm,z in scores.items():
                    subject_metrics.append({'subject':str(subject),'arm':arm,'metrics':logit_diagnostics(h.labels[keep],np.asarray(z)[keep])})
            record={'fold':fold,'metrics':metrics,'logits':scores,'rows':rows,'readouts':models,
                    'subject_metrics':subject_metrics,'checks':checks,'train_count':len(tb),'train_valid_count':int(tv.sum()),
                    'source_checkpoint_sha256':weight_sha,'plan_sha256':plan_sha,'outer_calls':0,'peak_cuda_bytes':peak}
            write_json(output/f'fold{fold}_report.json',record)
            if not all(checks.values()): raise RuntimeError(f'target integrity check failed: {checks}')
            records.append(record)
            print(f'S1 fold{fold} AP '+json.dumps({a:metrics[a]['all']['average_precision'] for a in metrics}),flush=True)
            del tb,hb,x,v
            gc.collect()
        gate=transfer_gate(records)
        if protected_snapshot(root)!=snapshot: raise RuntimeError('protected release changed')
        artifact_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if artifact_bytes>2*1024**3 or list(output.rglob('*.pt')):
            raise RuntimeError('target artifact budget or checkpoint-copy violation')
        gate.update({'wall_seconds':time.perf_counter()-started,'encoding_seconds':encoding_seconds,'neural_optimizer_steps':0,
                     'artifact_bytes_before_summary':artifact_bytes,'copied_checkpoint_bytes':0,
                     'protected_unchanged':True,'persistent_token_cache_bytes':0,
                     'source_checkpoint_sha256':weight_sha})
        write_json(output/'summary.json',gate); write_json(output/'state.json',{'status':gate['status'],'outer_calls':0})
        print(json.dumps(gate,sort_keys=True),flush=True)
    except Exception as exc:
        write_json(output/'state.json',{'status':'S1_INCOMPLETE','completed_folds':len(records),'error':str(exc),
                                        'wall_seconds':time.perf_counter()-started,'outer_calls':0})
        raise


if __name__=='__main__':
    with threadpool_limits(limits=4): run(Path(__file__).resolve().parents[3])
