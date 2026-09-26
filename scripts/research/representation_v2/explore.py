"""A2 preregistered three-fold candidate probe, never an outer evaluation."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

from collections import Counter
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
from scripts.research.representation_v2.dataset import load_candidate_bags
from scripts.research.representation_v2.health import logit_diagnostics, validate_health_partition
from scripts.research.representation_v2.low_capacity import encode_candidates, fit_readout, predict_readout, readout_payload, subject_weights
from scripts.research.representation_v2.models import MotionBranch
from scripts.research.representation_v2.provenance import assert_no_heldout_ancestors
from scripts.research.representation_v2.ssl import build_token_bank, fit_ssl
from scripts.research.representation_v2.train import fit_robust_channels
from src.pipeline.challenger.e03 import registered_e03_config
from src.pipeline.challenger.protection import protected_snapshot
from src.pipeline.runner import FilesystemDataSource


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    try:
        with temporary.open('w', encoding='utf-8', newline='\n') as handle:
            handle.write(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def select_gate(rows):
    if len(rows) != 3:
        raise ValueError('R1 requires all three preregistered folds')
    means = {}
    for arm in ('m','r','s','prior'):
        for row in rows:
            values = [row['metrics'][arm]['all'][metric] for metric in ('average_precision','raw_logit_bce')]
            if any(value is None or not np.isfinite(value) for value in values):
                raise ValueError('undefined or non-finite gate metric; attempt cannot pass')
        means[arm] = {metric:float(np.mean([row['metrics'][arm]['all'][metric] for row in rows]))
                      for metric in ('average_precision','raw_logit_bce')}
    gain = means['s']['average_precision']-means['r']['average_precision']
    wins = sum(row['metrics']['s']['all']['average_precision'] > row['metrics']['r']['all']['average_precision'] for row in rows)
    passed = (gain >= .02-1e-12 and wins>=2 and
              means['s']['raw_logit_bce']<=means['r']['raw_logit_bce']+.02+1e-12 and
              means['s']['raw_logit_bce']<=means['prior']['raw_logit_bce']+.02+1e-12 and
              means['s']['average_precision']>means['m']['average_precision'] and
              all(row['collapse']['passed'] for row in rows) and
              all(row.get('checks',{}).get(key) is True for row in rows
                  for key in ('provenance','finite','optimizer','resource')))
    return {'passed':bool(passed),'status':'R1_SUPPORTS_WIDER_INNER_PROBE' if passed else 'R1_NO_SUPPORTED_SSL_BRANCH',
            'macro_metrics':means,'ap_gain_s_vs_r':gain,'positive_folds':int(wins),'outer_calls':0}


def _metrics(y,z,valid):
    masks={'all':np.ones(len(y),bool),'valid':valid,'invalid':~valid}
    return {name:logit_diagnostics(y[mask],z[mask]) if mask.any() else None for name,mask in masks.items()}


def run(root):
    root=Path(root)
    output=root/'outputs/challenger_70/representation_v2/amendment_a2'
    if (output/'plan.json').exists():
        raise RuntimeError('registered attempt already exists; no silent retry or overwrite')
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
        raise RuntimeError('commit the registered implementation before training')
    if shutil.disk_usage(root).free < 20*1024**3:
        raise RuntimeError('disk reserve below20GiB')
    started=time.perf_counter()
    snapshot=protected_snapshot(root)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda' and torch.cuda.mem_get_info()[0] < 1280*1024**2:
        raise RuntimeError('less than1.25GiB free GPU memory; do not crowd existing workloads')
    torch.set_num_threads(min(8,max(1,(os.cpu_count() or 4)-2)))
    code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    spec=root/'docs/challenge_070_amendment_rv2_a2.md'
    plan={'id':'RV2-A2-R1-001','code_sha':code_sha,'spec_sha256':hashlib.sha256(spec.read_bytes()).hexdigest(),
          'seed':20260926,'steps_per_ssl':5000,'ssl_fits':3,'base_fits':30,'readout_fits':9,
          'device':device,'outer_calls':0,'max_wall_seconds':21600,'max_training_seconds':10800,
          'protected_digest':hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest()}
    write_json(output/'plan.json',plan)
    plan_sha=hashlib.sha256((output/'plan.json').read_bytes()).hexdigest()
    write_json(output/'state.json',{'status':'R1_IN_PROGRESS','outer_calls':0})
    completed=[]
    training_seconds=0.
    try:
        config=registered_e03_config(0,workers=4)
        data=FilesystemDataSource(root).load_outer_train_only(config)
        original_t,original_v=inner_subject_partitions(data,n_splits=4)[0]
        source=slice_outer_train(data,original_t)
        del data
        for fold,(train,health) in enumerate(inner_subject_partitions(source,n_splits=3)):
            if time.perf_counter()-started>21600:
                raise TimeoutError('R1 wall budget exhausted')
            validate_health_partition(original_t,original_v,train,health)
            split={k:list(map(str,v)) for k,v in [('original_train',original_t),('excluded_original_validation',original_v),
                                                 ('fit_subjects',train),('health_subjects',health)]}
            print(f'fold{fold}: generating fit-only producers, {len(train)} train / {len(health)} health',flush=True)
            write_json(output/f'fold{fold}_subjects.json',split)
            graph={}
            t,h=build_candidate_ledgers(source,config,train,health,fit_audit=graph)
            assert_no_heldout_ancestors('candidates',set(health)|set(original_v),graph)
            t_bags,h_bags=load_candidate_bags(root,t.candidates),load_candidate_bags(root,h.candidates)
            tv=np.array([b.token_mask.any() for b in t_bags]); hv=np.array([b.token_mask.any() for b in h_bags])
            if not tv.any() or not hv.any():
                raise RuntimeError('no valid train or health bags')
            fit_bags=[b for b,keep in zip(t_bags,tv) if keep]
            center,scale=fit_robust_channels(fit_bags)
            bank=build_token_bank(fit_bags,t.groups[tv],seed=20260926)
            if set(bank['subjects'])-set(train) or set(bank['subjects']) & set(health):
                raise RuntimeError('SSL bank ancestry violation')
            for node,owners,parents in [
                ('normalization',t.groups[tv],['candidates']),
                ('bank',bank['subjects'],['candidates']),
                ('ssl',bank['subjects'],['bank','normalization']),
                ('readout',t.groups[tv],['candidates','normalization','ssl'])]:
                graph[node]={'fit_subjects':sorted(set(map(str,owners))),'parents':parents}
            if any(set(node['fit_subjects'])-set(train) for node in graph.values()):
                raise RuntimeError('learned ancestor contains subjects outside current train')
            assert_no_heldout_ancestors('readout',set(health)|set(original_v),graph)
            write_json(output/f'fold{fold}_provenance.json',graph)
            torch.manual_seed(20260926)
            initial=MotionBranch().token_encoder
            xr=encode_candidates(initial,t_bags,center,scale,device=device)
            vr=encode_candidates(initial,h_bags,center,scale,device=device)
            prior=float(np.average(t.labels[tv],weights=subject_weights(t.groups[tv])))
            if not 0<prior<1:
                raise RuntimeError('finite fallback requires two classes')
            prior_z=np.full(len(h_bags),logit(prior))
            scores={'prior':prior_z.tolist()}
            metrics={'prior':_metrics(h.labels,prior_z,hv)}
            fitted_payload={}

            def fit_arm(arm,x,v):
                fitted=fit_readout(x[tv],t.labels[tv],t.groups[tv])
                z=prior_z.copy(); z[hv]=predict_readout(fitted,v[hv])
                scores[arm]=z.tolist(); metrics[arm]=_metrics(h.labels,z,hv)
                fitted_payload[arm]=readout_payload(fitted)

            fit_arm('m',xr[:,-5:],vr[:,-5:]); fit_arm('r',xr,vr)
            print(f'fold{fold}: SSL bank {len(bank["values"])} tokens; fitting fixed5000 steps',flush=True)
            free_cuda=torch.cuda.mem_get_info()[0] if device=='cuda' else None
            if device=='cuda': torch.cuda.reset_peak_memory_stats()
            ssl_history=[]

            def progress(row):
                if time.perf_counter()-started>21600:
                    raise TimeoutError('R1 wall budget exhausted')
                if row['step']%1000==0:
                    print(f'fold{fold} SSL step={row["step"]} loss={row["loss"]:.4f}',flush=True)
                ssl_history.append(row)
                write_json(output/f'fold{fold}_progress.json',ssl_history)

            trained=fit_ssl(initial,bank,center,scale,seed=20260926,steps=5000,device=device,
                            max_seconds=min(10800-training_seconds,21600-(time.perf_counter()-started)),progress=progress)
            training_seconds+=trained['wall_seconds']
            xs=encode_candidates(trained['encoder'],t_bags,center,scale,device=device)
            vs=encode_candidates(trained['encoder'],h_bags,center,scale,device=device)
            fit_arm('s',xs,vs)
            checkpoint={'schema':'rv2-a2-ssl-v1','encoder':{k:v.cpu() for k,v in trained['encoder'].state_dict().items()},
                        'projector':{k:v.cpu() for k,v in trained['projector'].state_dict().items()},
                        'random_encoder':{k:v.cpu() for k,v in initial.state_dict().items()},
                        'normalization':{'center':center.tolist(),'scale':scale.tolist()},
                        'steps':5000,'provenance':{**split,'plan_sha256':plan_sha,'code_sha':code_sha,'outer_calls':0}}
            checkpoint_path=output/f'fold{fold}_final.pt'
            temporary=checkpoint_path.with_suffix('.pt.tmp')
            try:
                torch.save(checkpoint,temporary)
                torch.load(temporary,weights_only=True,map_location='cpu')
                temporary.replace(checkpoint_path)
            finally:
                temporary.unlink(missing_ok=True)
            rows=[{'sid':c.event.sid,'start_ms':int(c.event.start_ms),'end_ms':int(c.event.end_ms),
                   'subject':str(g),'label':int(y),'valid':bool(v)} for c,g,y,v in zip(h.candidates,h.groups,h.labels,hv)]
            peak_cuda=torch.cuda.max_memory_allocated() if device=='cuda' else None
            checkpoint_bytes=sum(p.stat().st_size for p in (root/'outputs/challenger_70/representation_v2').rglob('*.pt'))
            finite=all(np.isfinite(np.asarray(values)).all() for values in scores.values()) and all(
                all(np.isfinite(value) for value in row.values()) for row in trained['history'])
            resource=(checkpoint_bytes<=2*1024**3 and checkpoint_path.stat().st_size<10*1024**2 and
                      training_seconds<=10800 and time.perf_counter()-started<=21600 and
                      shutil.disk_usage(root).free>=20*1024**3 and
                      (peak_cuda is None or peak_cuda<=.8*free_cuda))
            checks={'provenance':True,'finite':bool(finite),
                    'optimizer':trained['steps']==5000 and trained['history'][-1]['step']==5000,
                    'resource':bool(resource)}
            report={'fold':fold,'metrics':metrics,'logits':scores,'rows':rows,'readouts':fitted_payload,
                    'checks':checks,'checkpoint_bytes_all_rv2':checkpoint_bytes,
                    'training_history':trained['history'],'collapse':trained['collapse'],
                    'bank_count':len(bank['values']),'bank_subject_counts':dict(Counter(map(str,bank['subjects']))),
                    'bank_keys_sha256':hashlib.sha256(json.dumps(bank['keys']).encode()).hexdigest(),
                    'train_count':len(t_bags),'train_valid_count':int(tv.sum()),
                    'normalization':{'center':center.tolist(),'scale':scale.tolist()},'micro_threshold':float(t.micro_threshold),
                    'training_seconds':trained['wall_seconds'],'outer_calls':0,
                    'peak_cuda_bytes':peak_cuda,
                    'plan_sha256':plan_sha,'code_sha':code_sha,
                    'checkpoint_sha256':hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()}
            write_json(output/f'fold{fold}_report.json',report)
            if not all(checks.values()):
                raise RuntimeError(f'fold integrity or budget check failed: {checks}')
            completed.append(report)
            print(f'fold{fold} complete: '+json.dumps({a:metrics[a]['all']['average_precision'] for a in metrics}),flush=True)
            del bank,t_bags,h_bags,fit_bags,trained,initial,xr,vr,xs,vs,checkpoint
            gc.collect()
            if device=='cuda': torch.cuda.empty_cache()
        gate=select_gate(completed)
        if protected_snapshot(root)!=snapshot:
            raise RuntimeError('protected release snapshot changed')
        gate.update({'wall_seconds':time.perf_counter()-started,'training_seconds':training_seconds,
                     'optimizer_steps':15000,'protected_unchanged':True,'persistent_token_cache_bytes':0,
                     'output_bytes':sum(p.stat().st_size for p in output.rglob('*') if p.is_file())})
        write_json(output/'summary.json',gate)
        write_json(output/'state.json',{'status':gate['status'],'outer_calls':0})
        print(json.dumps(gate,sort_keys=True),flush=True)
    except Exception as exc:
        write_json(output/'state.json',{'status':'R1_INCOMPLETE','error':str(exc),'completed_folds':len(completed),
                                        'wall_seconds':time.perf_counter()-started,'outer_calls':0})
        raise


if __name__=='__main__':
    with threadpool_limits(limits=4):
        run(Path(__file__).resolve().parents[3])
