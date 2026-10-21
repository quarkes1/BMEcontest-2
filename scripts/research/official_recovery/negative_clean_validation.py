"""Training-side policy sealing and paired five-fold window-censoring validation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .consistent_validation import aggregate
from .eligibility_provenance import eligibility_inputs
from .event_probe import check_graph,choose_complete_policy,complete_metrics,decode,partition_truths
from .full_pool_probe import reachable_truths
from .negative_clean_probe import (_read,_write,_inputs,_paired,protected_sources,
                                   validate_policy_population,assess_clean_gate)
from .scope_audit import RUN_KEY,_sha,validated_session_subjects,read_fold_subjects,fold_subjects
from src.pipeline import runner
from src.pipeline.artifacts import verify_promotion_attestation
from src.pipeline.challenger.e03 import registered_e03_config

ARMS=('A','W')
SPEC='docs/official_recovery_negative_clean_validation_v1.md'


def policy_partitions(subjects):
    subjects=list(subjects)
    if len(subjects)!=len(set(subjects)) or len(subjects)<6:
        raise ValueError('unique policy subjects required')
    order=np.random.default_rng(20260901).permutation(sorted(subjects))
    return tuple((sorted(set(subjects)-set(v)),sorted(v.tolist())) for v in np.array_split(order,3))


def validation_gate(records):
    if len(records)!=5 or {r['fold'] for r in records}!=set(range(5)):
        raise ValueError('five complete unique folds required')
    if any(r['events']['A']['truths']!=r['events']['W']['truths'] for r in records):
        raise ValueError('paired denominator differs')
    totals={a:aggregate([r['events'][a] for r in records]) for a in ARMS}
    if totals['A']['truths']!=254: raise ValueError('complete truth count differs')
    delta=totals['W']['f1']-totals['A']['f1']
    wins=sum(aggregate([r['events']['W']])['f1']>aggregate([r['events']['A']])['f1'] for r in records)
    return dict(passed=bool(delta>=.02-1e-12 and wins>=3 and totals['W']['tp']>=totals['A']['tp']),
        aggregate=totals,delta_f1=delta,positive_folds=int(wins),
        delta_tp=totals['W']['tp']-totals['A']['tp'],delta_fp=totals['W']['fp']-totals['A']['fp'])


def load_context(root):
    from src.data.manifests import load_sensor_index,MEALS_CSV,INDEX_CSV
    source=root/'outputs/official_recovery/negative_clean_v1'
    gate=assess_clean_gate([_read(source/f'fold{k}_report.json') for k in range(3)])
    if gate['winner']!='W' or any(_read(source/'summary.json')[k]!=v for k,v in gate.items()):
        raise ValueError('completed W development gate required')
    run=root/'models/event_stack'/RUN_KEY
    if verify_promotion_attestation(run,expected_run_key=RUN_KEY): raise ValueError('release attestation differs')
    records,_=read_fold_subjects(run,_read(run/'promotion_summary.json'));subject_fold=fold_subjects(records)
    scope=_read(root/'outputs/official_recovery/scope_audit_confirmed/summary.json')
    if (len(subject_fold)!=39 or subject_fold!={r['subject']:r['fold'] for r in scope['per_subject']}):
        raise ValueError('cohort differs')
    for p in (Path(MEALS_CSV),Path(INDEX_CSV)):
        if scope['source_sha256'][str(p.relative_to(root))]!=_sha(p): raise ValueError('metadata changed')
    truths=[(r['subject'],int(r['start_ms']),int(r['end_ms'])) for r in scope['truth_rows']]
    if len(truths)!=254: raise ValueError('truth population differs')
    mapping=validated_session_subjects(load_sensor_index().to_dict('records'))
    eligibility=eligibility_inputs(root,{sid:s for sid,s in mapping.items() if s in subject_fold})
    configs=tuple(registered_e03_config(k,workers=8) for k in range(5))
    full=runner.build_full_target_dataset(configs,runner.FilesystemDataSource(root),expected_truth_count=153)
    mapping={sid:mapping[sid] for sid in full.subject_by_session}
    if set(mapping.values())!=set(subject_fold): raise ValueError('full candidate population differs')
    data=runner.OuterTrainOnlyData(full.window_train,full.candidate_train,full.train_truths,mapping,
        full.session_bounds_by_sid,full.micro_window_train,full.micro_candidate_train)
    if eligibility!=eligibility_inputs(root,eligibility): raise ValueError('eligibility changed during loading')
    return configs,data,truths,subject_fold,eligibility


def inputs(root):
    files=_inputs(root)
    paths={root/SPEC}
    paths.update((root/'outputs/official_recovery/negative_clean_v1').glob('*.json'))
    paths.update(root/'cache/splits'/f'fold{k}.json' for k in range(5))
    paths.update(root/'cache'/kind/f'fold{k}_{split}.npz' for kind in ('slide','micro15') for k in range(5)
        for split in ('train','meal_train','no_meal_train'))
    paths.update((root/'models/event_stack'/RUN_KEY).glob('*.json'))
    files.update({str(p.relative_to(root)):_sha(p) for p in paths})
    return files


def execute(root,output,phase='policy'):
    root,output=Path(root).resolve(),Path(output).resolve();completed=0;started=time.perf_counter()
    if phase=='policy': output.mkdir(parents=True,exist_ok=False)
    elif phase!='outer': raise ValueError('unknown validation phase')
    try:
        if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
            raise RuntimeError('commit code and evidence before validation')
        if phase=='policy':
            fingerprints=inputs(root);protected=protected_sources(root)
            configs,data,truths,subject_fold,eligibility=load_context(root)
            partitions={str(k):policy_partitions([s for s,f in subject_fold.items() if f!=k]) for k in range(5)}
            _write(output/'eligibility_inputs.json',eligibility)
            _write(output/'plan.json',dict(id='NEGATIVE-CLEAN-FIVEFOLD-V1',source_sha256=fingerprints,
                protected=protected,subject_fold=subject_fold,policy_partitions=partitions,started_epoch=time.time(),
                code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                expected_window_fits=400,expected_meta_fits=80,complete_truth_count=254,arms=ARMS))
        else:
            if _read(output/'state.json')['state']!='POLICY_FROZEN': raise ValueError('frozen policies required')
            if any(_sha(output/p)!=h for p,h in _read(output/'policy_seal.json')['sha256'].items()):
                raise ValueError('policy seal changed')
            plan=_read(output/'plan.json');fingerprints=plan['source_sha256'];protected=plan['protected']
            eligibility=_read(output/'eligibility_inputs.json')
            configs,data,truths,subject_fold,loaded_eligibility=load_context(root)
            if eligibility!=loaded_eligibility: raise ValueError('eligibility changed since policy')
            partitions=plan['policy_partitions']
            if plan['subject_fold']!=subject_fold: raise ValueError('frozen cohort changed')
        plan=_read(output/'plan.json')
        def unchanged(*,check_eligibility=True):
            if (fingerprints!=inputs(root) or protected!=protected_sources(root)
                or (check_eligibility and eligibility!=eligibility_inputs(root,eligibility))):
                raise ValueError('source, eligibility or protected files changed')
        def guard():
            if time.time()-plan['started_epoch']>7200 or shutil.disk_usage(root).free<10*1024**3:
                raise RuntimeError('validation wall/disk limit exceeded')
            if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>128*1024**2:
                raise RuntimeError('evidence budget exceeded')
        # load_context has just verified these arrays both before and after loading.
        unchanged(check_eligibility=False);guard();records=[]
        for fold,config in enumerate(configs):
            train={s for s,k in subject_fold.items() if k!=fold};score=set(subject_fold)-train
            if phase=='policy':
                cs={a:[] for a in ARMS};ps={a:[] for a in ARMS};gs={a:[] for a in ARMS};graphs={a:{} for a in ARMS};seen=[]
                for i,(fit,valid) in enumerate(partitions[str(fold)]):
                    guard()
                    if set(fit)&set(valid) or set(fit)|set(valid)!=train: raise ValueError('partition differs')
                    print(f'negative-clean fivefold policy {fold}/{i}: train{len(fit)} score{len(valid)}',flush=True)
                    ledgers,scores,parent=_paired(data,config,fit,valid,score,truths,output,f'fold{fold}_policy{i}',arms=ARMS)
                    seen.extend(valid);completed+=1
                    for a in ARMS:
                        cs[a].extend(ledgers[a].candidates);ps[a].extend(scores[a]);gs[a].extend(ledgers[a].groups)
                        for n,node in parent[a].items():
                            graphs[a][f'p{i}_{n}']=dict(fit_subjects=node['fit_subjects'],parents=[f'p{i}_{p}' for p in node['parents']])
                    del ledgers,scores,parent;gc.collect()
                policies={}
                for a in ARMS:
                    validate_policy_population(seen,gs[a],train)
                    graphs[a]['policy']=dict(fit_subjects=sorted(train),parents=[f'p{i}_score' for i in range(3)])
                    check_graph(graphs[a],'policy',score,train)
                    policies[a]=choose_complete_policy(cs[a],np.asarray(ps[a]),gs[a],partition_truths(truths,train),data.subject_by_session,config)
                _write(output/f'fold{fold}_policies.json',dict(policies=policies,graphs=graphs))
                print(f'negative-clean fivefold {fold} policies frozen',flush=True)
                continue
            frozen=_read(output/f'fold{fold}_policies.json')
            for graph in frozen['graphs'].values(): check_graph(graph,'policy',score,train)
            print(f'negative-clean fivefold outer {fold}: train{len(train)} score{len(score)}',flush=True)
            ledgers,scores,graphs=_paired(data,config,train,score,(),truths,output,f'fold{fold}_outer',arms=ARMS)
            targets=partition_truths(truths,score)
            predictions={a:decode(ledgers[a].candidates,scores[a],ledgers[a].groups,frozen['policies'][a]) for a in ARMS}
            events={a:complete_metrics(predictions[a],data.subject_by_session,targets) for a in ARMS}
            per_subject=[dict(subject=s,events={a:complete_metrics([p for p in predictions[a] if data.subject_by_session[p.sid]==s],
                data.subject_by_session,partition_truths(targets,{s})) for a in ARMS}) for s in sorted(score)]
            record=dict(fold=fold,train=sorted(train),score=sorted(score),events=events,subject_events=per_subject,truths=targets,
                raw_reachable={a:reachable_truths(ledgers[a].candidates,data.subject_by_session,targets) for a in ARMS},
                predictions={a:[asdict(p) for p in predictions[a]] for a in ARMS})
            records.append(record);completed+=1;_write(output/f'fold{fold}_report.json',record)
            print('negative-clean fivefold counts '+str(events),flush=True)
            del ledgers,scores,graphs;gc.collect()
        guard();unchanged()
        if phase=='policy':
            _write(output/'policy_seal.json',dict(sha256={p.name:_sha(p) for p in output.glob('*.json')
                if p.name not in ('state.json','policy_seal.json')},completed_policy_splits=completed,
                elapsed_seconds=time.perf_counter()-started))
            result=dict(state='POLICY_FROZEN',completed_splits=completed)
        else:
            gate=validation_gate(records)
            differences=[s['events']['W']['f1']-s['events']['A']['f1'] for r in records for s in r['subject_events']]
            result=dict(state='COMPLETE',evidence_kind='FIVEFOLD_NEGATIVE_CLEAN_DEVELOPMENT_ONLY',**gate,
                subject_diagnostics=dict(improved=sum(d>0 for d in differences),worsened=sum(d<0 for d in differences),tied=sum(d==0 for d in differences)),
                protected_unchanged=True,window_fits=400,meta_fits=80,total_wall_seconds=time.time()-plan['started_epoch'])
            _write(output/'summary.json',result)
        _write(output/'state.json',{'state':result['state']});return result
    except BaseException as exc:
        if output.is_dir(): _write(output/'state.json',dict(state='INCOMPLETE',phase=phase,error=repr(exc),completed_splits=completed))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('policy','outer'),default='policy');args=parser.parse_args()
    with threadpool_limits(limits=8):
        print(execute(Path(__file__).resolve().parents[3],args.output,args.phase),flush=True)
