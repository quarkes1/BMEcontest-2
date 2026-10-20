"""Expand nested candidate coverage, keeping supervised window training fixed."""
from __future__ import annotations

import argparse
from dataclasses import asdict,replace
import gc
import json
from pathlib import Path
import shutil
import subprocess
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from .day_budget_probe import restore_candidates,validate_ledger
from .event_probe import assess_gate,check_graph,choose_complete_policy,complete_metrics,decode,full_candidate_labels,partition_truths
from .label_probe import load_pools,validate_partitions
from .scope_audit import _sha,match_events,protection,validated_session_subjects
from .eligibility_provenance import eligibility_inputs
from .event_probe import subject_events
from scripts.research.representation_v2.crossfit import build_candidate_ledgers,slice_outer_train
from src.pipeline import runner
from src.pipeline.challenger.e03 import registered_e03_config
from src.pipeline.event_stack import EventRef
from src.pipeline.runner import WindowBatch


def validate_subset(partial,full):
    keys=lambda windows:[(w.sid,w.start_ms,w.end_ms) for w in windows]
    identities=keys(full.windows)
    if len(set(identities))!=len(identities): raise ValueError('duplicate full window identity')
    lookup={w:i for i,w in enumerate(identities)}
    old=keys(partial.windows);sids={w.sid for w in partial.windows}
    if set(old)!={w for w in identities if w[0] in sids}:
        raise ValueError('partial session window identities differ from full cache')
    selected=np.asarray([lookup[w] for w in old],dtype=int)
    if not np.allclose(partial.features,full.features[selected],rtol=0,atol=0,equal_nan=True):
        raise ValueError('overlapping candidate features differ')


def expand_candidates(data,pools,mapping,bounds):
    allowed=set(data.subject_by_session.values())
    batches={}
    for kind in ('slide','micro15'):
        pool=pools[kind]
        if set(pool['subjects'])!=allowed: raise ValueError('full candidate pool must be exactly T24')
        features=pool['features'][:,:62] if kind=='slide' else pool['features']
        windows=tuple(EventRef(str(sid),int(span[0]),int(span[1])) for sid,span in zip(pool['sessions'],pool['intervals']))
        batches[kind]=WindowBatch(features,pool['old'],windows)
        validate_subset(data.candidate_train if kind=='slide' else data.micro_candidate_train,batches[kind])
    sids=set(data.subject_by_session)|{w.sid for b in batches.values() for w in b.windows}
    if any(mapping[sid] not in allowed for sid in sids): raise ValueError('forbidden expanded identity')
    return replace(data,candidate_train=batches['slide'],micro_candidate_train=batches['micro15'],
        subject_by_session={sid:mapping[sid] for sid in sorted(sids)},
        session_bounds_by_sid={sid:bounds[sid] for sid in sorted(sids)})


def verify_partial_candidates(full,partial,partial_sessions):
    if [c for c in full if c.event.sid in partial_sessions]!=list(partial):
        raise ValueError('partial candidate geometry/support did not replay')


def reachable_truths(candidates,mapping,truths):
    raw=subject_events([c.event for c in candidates],mapping)
    return sum(bool(match_events(raw,[truth])) for truth in truths)


def _read(path): return json.loads(path.read_text(encoding='utf-8'))
def _write(path,value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def _split(data,old_data,config,train,score,excluded,truths,source,output,prefix):
    baseline=_read(source/(prefix+'_scores.json'))
    validate_ledger(baseline,train,score,excluded)
    graph={}
    fixed_config=replace(config,micro_threshold_grid=(baseline['micro_threshold'],))
    t,v=build_candidate_ledgers(data,fixed_config,train,score,fit_audit=graph)
    partial_sids={w.sid for w in old_data.candidate_train.windows}
    verify_partial_candidates(v.candidates,restore_candidates(baseline['candidates']),partial_sids)
    y=full_candidate_labels(t.candidates,data.subject_by_session,partition_truths(truths,set(train)))
    if set(y.tolist())!={0,1}: raise ValueError('binary full-pool verifier targets required')
    probabilities=[]
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        for model in (runner._verifier_estimator(config.seed+20,.1),runner._verifier_lgbm_estimator(config.seed+21)):
            model.fit(t.features,y);probabilities.append(runner._positive_probability(model,v.features))
    scores=.5*probabilities[0]+.5*probabilities[1]
    if not np.isfinite(scores).all(): raise ValueError('nonfinite full-pool scores')
    for name,node in baseline['graph'].items():
        graph['source_'+name]=dict(fit_subjects=node['fit_subjects'],parents=['source_'+p for p in node['parents']])
    graph['candidates']['parents'].append('source_threshold')
    graph['verifier']={'fit_subjects':list(train),'parents':['candidates']}
    graph['score']={'fit_subjects':[],'parents':['verifier']}
    check_graph(graph,'score',set(score)|set(excluded),set(train))
    _write(output/(prefix+'_scores.json'),dict(train=list(train),score=list(score),excluded=sorted(excluded),graph=graph,
        candidates=[asdict(c) for c in v.candidates],groups=v.groups.tolist(),scores=scores.tolist(),
        train_candidates=len(t.candidates),train_positive=int(sum(y)),micro_threshold=t.micro_threshold,
        partial_score_candidates=len(baseline['candidates']),partial_candidate_replay=True))
    gc.collect()
    return v,scores,graph,baseline


def execute(root,source,output):
    from src.data.manifests import load_sensor_index,INDEX_CSV
    root,source,output=map(lambda p:Path(p).resolve(),(root,source,output))
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
        raise RuntimeError('commit experiment code and protocol before fitting')
    output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();protected=protection(root);records=[];completed=0
    try:
        original_summary=_read(source/'summary.json')
        if original_summary.get('state')!='COMPLETE' or not (source/'eligibility_inputs.json').exists():
            raise ValueError('complete confirmed source required')
        manifest_path=root/'outputs/challenger_70/representation_v2/amendment_a6_transformer/t1/manifest.json'
        manifest=_read(manifest_path);validate_partitions(manifest)
        allowed=set(manifest['identities']['t24']);forbidden=set(manifest['identities']['v7'])|set(manifest['identities']['outer'])
        config=registered_e03_config(0,workers=4)
        files=[manifest_path,Path(INDEX_CSV),root/'docs/official_recovery_full_pool_v1.md',Path(__file__).resolve()]
        files.extend(p for p in source.rglob('*') if p.is_file())
        files.extend(root/'cache'/kind/f'fold{fold}_val.npz' for kind in ('slide','micro15') for fold in range(5))
        files.extend(root/name for name in ('scripts/research/official_recovery/event_probe.py',
            'scripts/research/official_recovery/label_probe.py','scripts/research/official_recovery/day_budget_probe.py',
            'scripts/research/representation_v2/crossfit.py','src/pipeline/runner.py','src/pipeline/event_stack.py',
            'src/pipeline/candidate_control.py','src/pipeline/context_features.py'))
        # Original train caches and session-array identity are anchored by the confirmed source plan.
        source_hashes=_read(source/'plan.json')['source_sha256']
        for path,digest in source_hashes.items():
            # Updated controller/specs are versioned, not used to refit the old window pool here.
            if path.replace('\\','/').startswith('cache/') and _sha(root/path)!=digest:
                raise ValueError('confirmed training cache changed')
            if path.replace('\\','/').startswith('cache/'):
                files.append(root/path)
        eligibility_reference=_read(source/'eligibility_inputs.json')
        if eligibility_inputs(root,eligibility_reference)!=eligibility_reference:
            raise ValueError('confirmed eligible session arrays changed')
        inputs={str(p.relative_to(root)):_sha(p) for p in files}
        _write(output/'plan.json',dict(id='FULL-POOL-V1',source_sha256=inputs,config=asdict(config),
            code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),protected=protected,
            expected_window_fits=120,expected_verifier_fits=24,outer_calls=0))
        index=load_sensor_index().to_dict('records');mapping=validated_session_subjects(index)
        bounds={str(r['session_id']):(int(r['timeStamp.startTime']),int(r['timeStamp.endTime'])) for r in index}
        old_data=slice_outer_train(runner.FilesystemDataSource(root).load_outer_train_only(config),allowed)
        population=_read(source/'population.json')
        if [asdict(t) for t in old_data.train_truths]!=population['legacy_eligible_truths']:
            raise ValueError('legacy eligible truth snapshot changed')
        pools=load_pools(root,manifest)
        data=expand_candidates(old_data,pools,mapping,bounds)
        _write(output/'population.json',dict(subject_by_session=data.subject_by_session,
            session_bounds=data.session_bounds_by_sid,complete_truths=population['complete_truths'],
            partial_macro_sessions=len({w.sid for w in old_data.candidate_train.windows}),
            full_macro_sessions=len({w.sid for w in data.candidate_train.windows}),
            partial_features_replayed=True))
        del pools
        truths=[tuple(row) for row in population['complete_truths']]
        def budget():
            if time.perf_counter()-start>7200 or shutil.disk_usage(root).free<10*1024**3:
                raise RuntimeError('full pool wall/disk budget exceeded')
            if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>64*1024**2:
                raise RuntimeError('full pool evidence budget exceeded')
        frozen={}
        for wrapper in manifest['wrappers']:
            fold=wrapper['id'];train=wrapper['train'];health=wrapper['health']
            pc=[];pg=[];ps=[];seen=[];policy_graph={}
            for inner,split in enumerate(wrapper['policy_splits']):
                budget();fit,valid=split['train'],split['score']
                print(f'full-pool health{fold} policy{inner}: train{len(fit)}/score{len(valid)}',flush=True)
                v,scores,graph,_=_split(data,old_data,config,fit,valid,set(health)|forbidden,truths,
                    source,output,f'fold{fold}_policy{inner}')
                completed+=1;pc.extend(v.candidates);pg.extend(v.groups);ps.extend(scores);seen.extend(valid)
                for name,node in graph.items():
                    policy_graph[f'p{inner}_{name}']=dict(fit_subjects=node['fit_subjects'],parents=[f'p{inner}_{p}' for p in node['parents']])
            if sorted(seen)!=sorted(train): raise ValueError('OOF subject coverage mismatch')
            policy_graph['policy']={'fit_subjects':list(train),'parents':[f'p{i}_score' for i in range(3)]}
            check_graph(policy_graph,'policy',set(health)|forbidden,set(train))
            frozen[fold]=choose_complete_policy(pc,np.asarray(ps),pg,partition_truths(truths,set(train)),data.subject_by_session,config)
            _write(output/f'fold{fold}_policies.json',dict(policy=frozen[fold],graph=policy_graph))
            print(f'full-pool health{fold}: policy frozen',flush=True)
        for wrapper in manifest['wrappers']:
            budget();fold=wrapper['id'];train=wrapper['train'];health=wrapper['health']
            print(f'full-pool health{fold}: refit train{len(train)}/health{len(health)}',flush=True)
            v,scores,graph,baseline=_split(data,old_data,config,train,health,forbidden,truths,source,output,f'fold{fold}_health')
            completed+=1
            old_candidates=restore_candidates(baseline['candidates'])
            old_policy=_read(source/f'fold{fold}_policies.json')['policies']['full']
            predictions={'old':decode(old_candidates,np.asarray(baseline['scores']['full']),baseline['groups'],old_policy),
                         'full':decode(v.candidates,scores,v.groups,frozen[fold])}
            targets=partition_truths(truths,set(health))
            events={a:complete_metrics(pred,data.subject_by_session,targets) for a,pred in predictions.items()}
            subjects=[dict(subject=s,events={a:complete_metrics([p for p in pred if data.subject_by_session[p.sid]==s],
                data.subject_by_session,partition_truths(targets,{s})) for a,pred in predictions.items()}) for s in health]
            _write(output/f'fold{fold}_report.json',dict(events=events,subject_events=subjects,truths=targets,
                predictions={a:[asdict(p) for p in pred] for a,pred in predictions.items()},
                raw_reachable_truths={a:reachable_truths(c,data.subject_by_session,targets)
                                     for a,c in [('partial',old_candidates),('all',list(v.candidates))]}))
            records.append(events)
            print('full-pool complete-meal counts '+json.dumps(events),flush=True)
        budget();gate=assess_gate(records)
        if gate['aggregate']['old']!=original_summary['aggregate']['full']:
            raise ValueError('frozen baseline complete-event counts did not replay')
        if (protected!=protection(root) or inputs!={str(p.relative_to(root)):_sha(p) for p in files}
                or eligibility_inputs(root,eligibility_reference)!=eligibility_reference):
            raise ValueError('protected release or source changed')
        result=dict(state='COMPLETE',evidence_kind='T24_FULL_CANDIDATE_POOL_DEVELOPMENT_ONLY',**gate,
            completed_splits=completed,outer_calls=0,protected_unchanged=True,elapsed_seconds=time.perf_counter()-start)
        _write(output/'summary.json',result);_write(output/'state.json',{'state':'COMPLETE'})
        return result
    except BaseException as exc:
        _write(output/'state.json',dict(state='INCOMPLETE',error=repr(exc),completed_splits=completed,health_folds=len(records)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=4):
        print(json.dumps(execute(Path(__file__).resolve().parents[3],args.source,args.output),ensure_ascii=False),flush=True)
