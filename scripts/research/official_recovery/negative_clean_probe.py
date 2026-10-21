"""Sealed-policy A/W/E nested development gate for negative-only censoring."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict,replace
import gc
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .consistent_validation import aggregate,canonical_train_role,fit_verifiers,_save_scores
from .eligibility_provenance import eligibility_inputs
from .event_probe import check_graph,choose_complete_policy,complete_metrics,decode,partition_truths
from .full_pool_probe import expand_candidates,reachable_truths
from .label_probe import load_pools,validate_partitions
from .negative_clean import prepare_train,censor_candidate_labels
from .scope_audit import _sha,protection,validated_session_subjects
from scripts.research.representation_v2.crossfit import build_candidate_ledgers
from src.pipeline import runner
from src.pipeline.challenger.e03 import registered_e03_config

ARMS=('A','W','E')


def _read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def _write(path,value): Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def assess_clean_gate(records):
    if len(records)!=3 or {r['fold'] for r in records}!=set(range(3)):
        raise ValueError('three complete unique health folds required')
    if any(len({r['events'][a]['truths'] for a in ARMS})!=1 for r in records):
        raise ValueError('paired GT denominator differs')
    totals={a:aggregate([r['events'][a] for r in records]) for a in ARMS}
    if totals['A']['truths']!=160: raise ValueError('complete GT denominator must be 160')
    arms={}
    for arm in ('W','E'):
        delta=totals[arm]['f1']-totals['A']['f1']
        wins=sum(aggregate([r['events'][arm]])['f1']>aggregate([r['events']['A']])['f1'] for r in records)
        arms[arm]=dict(passed=bool(delta>=.02-1e-12 and wins>=2 and totals[arm]['tp']>=totals['A']['tp']),
            delta_f1=delta,positive_folds=int(wins),delta_tp=totals[arm]['tp']-totals['A']['tp'],
            delta_fp=totals[arm]['fp']-totals['A']['fp'])
    passed=[a for a in ('W','E') if arms[a]['passed']]
    winner=max(passed,key=lambda a:(totals[a]['f1'],totals[a]['ppv'],-ARMS.index(a))) if passed else None
    return dict(aggregate=totals,arms=arms,winner=winner,passed=winner is not None)


def validate_policy_population(seen,groups,train):
    if Counter(seen)!=Counter({s:1 for s in train}) or set(groups)-set(train):
        raise ValueError('policy subject coverage differs')


def protected_sources(root):
    protected=protection(root)
    paths=[root/'dist/official_trials/consistent_refit_v1/EatingDetection.exe']
    paths.extend((root/'models/event_stack/official_consistent_refit_v1/deployment').iterdir())
    if any(not p.is_file() for p in paths): raise ValueError('first independent trial protection missing')
    protected.update({p.relative_to(root).as_posix():_sha(p) for p in paths})
    return protected


def load_context(root):
    from src.data.manifests import load_sensor_index,load_meals
    manifest=_read(root/'outputs/challenger_70/representation_v2/amendment_a6_transformer/t1/manifest.json')
    validate_partitions(manifest)
    allowed=set(manifest['identities']['t24'])
    forbidden=set(manifest['identities']['v7'])|set(manifest['identities']['outer'])
    config=registered_e03_config(0,workers=8)
    index=load_sensor_index().to_dict('records');mapping=validated_session_subjects(index)
    bounds={str(r['session_id']):(int(r['timeStamp.startTime']),int(r['timeStamp.endTime'])) for r in index}
    old=canonical_train_role(runner.FilesystemDataSource(root).load_outer_train_only(config),mapping,allowed)
    data=expand_candidates(old,load_pools(root,manifest),mapping,bounds)
    truths=partition_truths([(r.externalid,int(r.before_ms),int(r.after_ms)) for r in load_meals().itertuples()],allowed)
    population=_read(root/'outputs/official_recovery/event_probe_v1_confirmed/population.json')
    if len(truths)!=160 or set(map(tuple,population['complete_truths']))!=set(truths):
        raise ValueError('frozen complete T24 population changed')
    if [asdict(t) for t in old.train_truths]!=population['legacy_eligible_truths']:
        raise ValueError('legacy eligible targets changed')
    return config,data,truths,manifest,forbidden


def _paired(data,config,train,score,excluded,truths,output,prefix):
    if set(train)&(set(score)|set(excluded)) or set(score)&set(excluded): raise ValueError('split subject overlap')
    producers={};statistics={}
    for producer in ('A','W'):
        prepared,sampling=prepare_train(data,set(train),truths,producer)
        # Only training windows change; score role retains every original candidate window.
        scoring=replace(data,window_train=prepared.window_train,micro_window_train=prepared.micro_window_train)
        graph={}
        t,v=build_candidate_ledgers(scoring,replace(config,seed=config.seed+2),train,score,fit_audit=graph)
        producers[producer]=(t,v,graph);statistics[producer]=sampling
    ledgers={};scores={};graphs={};targets=partition_truths(truths,set(train))
    for arm in ARMS:
        t,v,parent=producers['W' if arm=='W' else 'A']
        y=np.asarray(t.labels);masked=dict(censored_negative=0,positive=int(np.count_nonzero(y==1)))
        if arm=='E': y,masked=censor_candidate_labels(t.candidates,y,targets,data.subject_by_session)
        keep=y>=0
        p,components=fit_verifiers(t.features[keep],y[keep],v.features,config,.1,.25)
        graph={name:dict(fit_subjects=node['fit_subjects'],parents=node['parents']) for name,node in parent.items()}
        graph['verifier']={'fit_subjects':sorted(set(train)),'parents':['candidates']}
        graph['score']={'fit_subjects':[],'parents':['verifier']}
        check_graph(graph,'score',set(score)|set(excluded),set(train))
        _save_scores(output/f'{prefix}_{arm}_scores.json',v.candidates,p,components,v.groups,graph,train,score)
        ledgers[arm]=v;scores[arm]=p;graphs[arm]=graph
        statistics[arm+'_meta']=dict(mask=masked,fit_rows=int(np.count_nonzero(keep)),
            fit_positive=int(np.count_nonzero(y[keep]==1)),oof_candidates=len(t.candidates),
            micro_threshold=t.micro_threshold,score_candidates=len(v.candidates))
    _write(output/f'{prefix}_training.json',statistics)
    gc.collect();return ledgers,scores,graphs


def _inputs(root):
    from src.data.manifests import MEALS_CSV,INDEX_CSV
    paths={Path(MEALS_CSV),Path(INDEX_CSV),root/'docs/official_recovery_negative_clean_v1.md',
        root/'outputs/challenger_70/representation_v2/amendment_a6_transformer/t1/manifest.json',
        root/'outputs/official_recovery/event_probe_v1_confirmed/population.json',
        root/'outputs/official_recovery/scope_audit_confirmed/summary.json',root/'cache/splits/fold0.json'}
    paths.update(p for tree in ('src','scripts') for p in (root/tree).rglob('*.py'))
    paths.update(root/'cache'/kind/f'fold{k}_val.npz' for kind in ('slide','micro15') for k in range(5))
    paths.update(root/'cache'/kind/f'fold0_{split}.npz' for kind in ('slide','micro15')
        for split in ('train','meal_train','no_meal_train'))
    return {str(p.relative_to(root)):_sha(p) for p in sorted(paths,key=str)}


def execute(root,output,phase='policy'):
    root,output=Path(root).resolve(),Path(output).resolve()
    if phase=='policy': output.mkdir(parents=True,exist_ok=False)
    elif phase!='health': raise ValueError('unknown negative-clean phase')
    started=time.perf_counter();completed=0
    try:
        if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True,stderr=subprocess.STDOUT).strip():
            raise RuntimeError('commit code and prior evidence before training')
        if phase=='policy':
            hashes=_inputs(root);protected=protected_sources(root);role_sids=set()
            for split in ('meal_train','no_meal_train'):
                with np.load(root/'cache/slide'/f'fold0_{split}.npz',allow_pickle=False) as z:
                    role_sids.update(json.loads(str(w))[0] for w in z['wid'])
            eligibility=eligibility_inputs(root,role_sids)
            _write(output/'eligibility_inputs.json',eligibility)
            _write(output/'plan.json',dict(id='NEGATIVE-CLEAN-V1',source_sha256=hashes,protected=protected,
                started_epoch=time.time(),code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                expected_window_fits=240,expected_meta_fits=72,blend=.25,verifier_c=.1,outer_calls=0))
        else:
            if _read(output/'state.json').get('state')!='POLICY_FROZEN': raise ValueError('frozen policy state required')
            seal=_read(output/'policy_seal.json')
            if any(_sha(output/p)!=digest for p,digest in seal['sha256'].items()): raise ValueError('policy seal changed')
            plan=_read(output/'plan.json');hashes=plan['source_sha256'];protected=plan['protected']
            eligibility=_read(output/'eligibility_inputs.json');role_sids=set(eligibility)
        plan=_read(output/'plan.json')
        def guard():
            if time.time()-plan['started_epoch']>7200 or shutil.disk_usage(root).free<10*1024**3:
                raise RuntimeError('negative-clean wall/disk limit exceeded')
            if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>128*1024**2:
                raise RuntimeError('negative-clean evidence limit exceeded')
        def unchanged():
            if (hashes!=_inputs(root) or protected!=protected_sources(root) or eligibility_inputs(root,role_sids)!=eligibility):
                raise ValueError('source, eligibility or protected release changed')
        guard();unchanged()
        config,data,truths,manifest,forbidden=load_context(root)
        if phase=='policy':
            _write(output/'population.json',dict(complete_truths=truths,subjects=manifest['identities']['t24'],
                wrappers=manifest['wrappers'],legacy_eligible_count=len(data.train_truths)))
            for wrapper in manifest['wrappers']:
                fold=wrapper['id'];train=wrapper['train'];health=wrapper['health']
                candidates={a:[] for a in ARMS};probabilities={a:[] for a in ARMS};groups={a:[] for a in ARMS}
                policy_graph={a:{} for a in ARMS};seen=[]
                for inner,split in enumerate(wrapper['policy_splits']):
                    guard();fit,valid=split['train'],split['score']
                    if set(fit)&set(valid) or set(fit)|set(valid)!=set(train): raise ValueError('policy split identity differs')
                    print(f'negative-clean policy health{fold}/inner{inner}: train{len(fit)} score{len(valid)}',flush=True)
                    ledgers,scores,graphs=_paired(data,config,fit,valid,set(health)|forbidden,truths,output,f'fold{fold}_policy{inner}')
                    seen.extend(valid);completed+=1
                    for arm in ARMS:
                        candidates[arm].extend(ledgers[arm].candidates);probabilities[arm].extend(scores[arm]);groups[arm].extend(ledgers[arm].groups)
                        for name,node in graphs[arm].items():
                            policy_graph[arm][f'p{inner}_{name}']=dict(fit_subjects=node['fit_subjects'],parents=[f'p{inner}_{p}' for p in node['parents']])
                    del ledgers,scores,graphs;gc.collect()
                policies={}
                for arm in ARMS:
                    validate_policy_population(seen,groups[arm],train)
                    graph=policy_graph[arm];graph['policy']={'fit_subjects':list(train),'parents':[f'p{i}_score' for i in range(3)]}
                    check_graph(graph,'policy',set(health)|forbidden,set(train))
                    policies[arm]=choose_complete_policy(candidates[arm],np.asarray(probabilities[arm]),groups[arm],
                        partition_truths(truths,set(train)),data.subject_by_session,config)
                _write(output/f'fold{fold}_policies.json',dict(policies=policies,graphs=policy_graph))
                print(f'negative-clean health{fold} policies frozen',flush=True)
            guard();unchanged()
            _write(output/'policy_seal.json',dict(state='POLICY_FROZEN',sha256={p.name:_sha(p) for p in output.glob('*.json') if p.name not in ('state.json','policy_seal.json')},
                phase_elapsed_seconds=time.perf_counter()-started,completed_policy_splits=completed))
            _write(output/'state.json',{'state':'POLICY_FROZEN'})
            return dict(state='POLICY_FROZEN',completed_splits=completed,elapsed_seconds=time.perf_counter()-started)
        records=[]
        for wrapper in manifest['wrappers']:
            guard();fold=wrapper['id'];train=wrapper['train'];health=wrapper['health']
            frozen=_read(output/f'fold{fold}_policies.json')
            for graph in frozen['graphs'].values(): check_graph(graph,'policy',set(health)|forbidden,set(train))
            print(f'negative-clean health{fold}: train{len(train)} score{len(health)}',flush=True)
            ledgers,scores,graphs=_paired(data,config,train,health,forbidden,truths,output,f'fold{fold}_health')
            targets=partition_truths(truths,set(health))
            predictions={a:decode(ledgers[a].candidates,scores[a],ledgers[a].groups,frozen['policies'][a]) for a in ARMS}
            events={a:complete_metrics(predictions[a],data.subject_by_session,targets) for a in ARMS}
            subjects=[dict(subject=s,events={a:complete_metrics([p for p in predictions[a] if data.subject_by_session[p.sid]==s],
                data.subject_by_session,partition_truths(targets,{s})) for a in ARMS}) for s in health]
            record=dict(fold=fold,events=events,subject_events=subjects,truths=targets,
                raw_reachable={a:reachable_truths(ledgers[a].candidates,data.subject_by_session,targets) for a in ARMS},
                predictions={a:[asdict(p) for p in pred] for a,pred in predictions.items()})
            records.append(record);completed+=1;_write(output/f'fold{fold}_report.json',record)
            print('negative-clean paired complete-meal counts '+json.dumps(events),flush=True)
            del ledgers,scores,graphs;gc.collect()
        guard();unchanged();gate=assess_clean_gate(records)
        result=dict(state='COMPLETE',evidence_kind='T24_NESTED_NEGATIVE_CLEAN_DEVELOPMENT_ONLY',**gate,
            protected_unchanged=True,outer_calls=0,window_fits=240,meta_fits=72,
            health_phase_seconds=time.perf_counter()-started,total_wall_seconds=time.time()-plan['started_epoch'])
        _write(output/'summary.json',result);_write(output/'state.json',{'state':'COMPLETE'});return result
    except BaseException as exc:
        if output.is_dir(): _write(output/'state.json',dict(state='INCOMPLETE',phase=phase,error=repr(exc),completed_splits=completed))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('policy','health'),default='policy');args=parser.parse_args()
    with threadpool_limits(limits=8):
        print(json.dumps(execute(Path(__file__).resolve().parents[3],args.output,args.phase)),flush=True)
