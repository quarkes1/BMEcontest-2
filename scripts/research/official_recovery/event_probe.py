"""Paired event-only full-meal supervision with partition-contained producers."""
from __future__ import annotations

import argparse
from dataclasses import asdict
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

from .scope_audit import _sha, event_metrics, match_events, normalized_id, prepare_truths, protection
from .label_probe import validate_partitions
from .eligibility_provenance import eligibility_inputs
from scripts.research.representation_v2.crossfit import build_candidate_ledgers, slice_outer_train
from scripts.research.representation_v2.provenance import assert_no_heldout_ancestors
from src.pipeline import runner
from src.pipeline.candidate_control import CandidateAdmissionConfig, admit_candidates
from src.pipeline.challenger.e03 import registered_e03_config
from src.pipeline.event_stack import compute_event_metrics, select_event_threshold


def partition_truths(rows, subjects):
    kept, _ = prepare_truths([dict(subject=s, start_ms=a, end_ms=b) for s,a,b in rows],
                             {s:0 for s in subjects})
    return kept


def subject_events(events, subject_by_session):
    result=[]
    for event in events:
        if event.sid not in subject_by_session:
            raise ValueError('unknown candidate session')
        result.append((normalized_id(subject_by_session[event.sid]),int(event.start_ms),int(event.end_ms)))
    return result


def full_candidate_labels(candidates, subject_by_session, truths):
    events=subject_events([c.event for c in candidates],subject_by_session)
    # Independent candidate targets, not one-to-one assignment: competing valid candidates stay positive.
    return np.asarray([bool(match_events([event],truths)) for event in events],dtype=np.int8)


def complete_metrics(events, subject_by_session, truths):
    return event_metrics(subject_events(events,subject_by_session),truths)


def check_graph(graph, node, heldout, allowed):
    assert_no_heldout_ancestors(node,set(heldout),graph)
    if any(set(n['fit_subjects'])-set(allowed) for n in graph.values()):
        raise ValueError('unexpected learned ancestor')


def decode(candidates, scores, groups, policy):
    rows=admit_candidates(candidates,scores,groups,CandidateAdmissionConfig(**policy['admission']))
    return [candidates[i].event for i in rows if scores[i]>=policy['threshold']]


def _choose_policy(candidates,scores,groups,truths,config,subject_by_session=None):
    scores=np.asarray(scores,dtype=float)
    complete=subject_by_session is not None
    choices=[]
    for index,admission in enumerate(CandidateAdmissionConfig(n,t,c)
            for n in config.admission_nms_iou_grid for t in config.admission_threshold_grid
            for c in config.admission_subject_cap_grid):
        rows=np.asarray(admit_candidates(candidates,scores,groups,admission),dtype=int)
        events=[candidates[i].event for i in rows]
        if complete:
            recall=complete_metrics(events,subject_by_session,truths)['recall']
            options=[]
            for threshold in np.unique(np.r_[scores[rows],0.,1.,np.nextafter(1.,2.)]):
                metrics=complete_metrics([candidates[i].event for i in rows if scores[i]>=threshold],
                                         subject_by_session,truths)
                options.append((metrics['f1'],metrics['ppv'],float(threshold),metrics))
            _,_,threshold,metrics=max(options,key=lambda value:value[:3])
            f1,ppv=metrics['f1'],metrics['ppv']
        else:
            recall=compute_event_metrics(events,truths).sensitivity
            selected=select_event_threshold(events,scores[rows],truths)
            threshold,metrics=selected.threshold,asdict(selected.metrics)
            f1,ppv=selected.metrics.f1,selected.metrics.ppv
        rank=(f1,ppv,-len(rows),-index,threshold)
        choices.append((recall,rank,dict(admission=asdict(admission),threshold=threshold,
                                       selection_metrics=metrics)))
    if not choices:
        raise ValueError('empty registered admission grid')
    feasible=[row for row in choices if row[0]>=config.admission_minimum_recall]
    return max(feasible or choices,key=lambda row:row[1] if feasible else (row[0],*row[1]))[2]


def choose_complete_policy(candidates,scores,groups,truths,subject_by_session,config):
    return _choose_policy(candidates,scores,groups,truths,config,subject_by_session)


def assess_gate(records):
    if len(records)!=3:
        raise ValueError('three complete health folds required')
    aggregate={}
    for arm in ('old','full'):
        tp=sum(row[arm]['tp'] for row in records)
        pred=sum(row[arm]['predictions'] for row in records)
        true=sum(row[arm]['truths'] for row in records)
        aggregate[arm]=dict(tp=tp,fp=pred-tp,fn=true-tp,predictions=pred,truths=true,
            f1=2*tp/(pred+true) if pred+true else 0.,ppv=tp/pred if pred else 0.,recall=tp/true if true else 0.)
    wins=sum(2*r['full']['tp']/(r['full']['predictions']+r['full']['truths'])>
             2*r['old']['tp']/(r['old']['predictions']+r['old']['truths']) for r in records)
    delta=aggregate['full']['f1']-aggregate['old']['f1']
    return dict(passed=bool(delta>=.01-1e-12 and wins>=2),aggregate=aggregate,
                delta_f1=delta,positive_folds=int(wins))


def _write(path,payload):
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def _paired_split(data,config,train,score,excluded,truths,output,prefix):
    if set(train)&(set(score)|set(excluded)):
        raise ValueError('train/heldout overlap')
    graph={}
    t,v=build_candidate_ledgers(data,config,train,score,fit_audit=graph)
    training_truths=partition_truths(truths,set(train))
    labels={'old':t.labels,'full':full_candidate_labels(t.candidates,data.subject_by_session,training_truths)}
    scores={}
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        for arm,y in labels.items():
            if set(y.tolist())!={0,1}:
                raise ValueError('binary verifier targets required')
            probabilities=[]
            for model in (runner._verifier_estimator(config.seed+20,.1),runner._verifier_lgbm_estimator(config.seed+21)):
                model.fit(t.features,y)
                probabilities.append(runner._positive_probability(model,v.features))
            scores[arm]=.5*probabilities[0]+.5*probabilities[1]
            if not np.isfinite(scores[arm]).all():
                raise ValueError('nonfinite verifier score')
            graph[arm+'_verifier']={'fit_subjects':list(train),'parents':['candidates']}
    graph['score']={'fit_subjects':[],'parents':['old_verifier','full_verifier']}
    check_graph(graph,'score',set(score)|set(excluded),set(train))
    _write(output/(prefix+'_scores.json'),dict(train=list(train),score=list(score),excluded=sorted(excluded),
        graph=graph,candidates=[asdict(c) for c in v.candidates],groups=v.groups.tolist(),
        scores={a:p.tolist() for a,p in scores.items()},micro_threshold=t.micro_threshold,
        train_label_counts={a:{str(k):int(sum(y==k)) for k in (0,1)} for a,y in labels.items()},
        old0_to_full1=int(sum((labels['old']==0)&(labels['full']==1))),
        old1_to_full0=int(sum((labels['old']==1)&(labels['full']==0)))))
    gc.collect()
    return v,scores,graph


def execute(root,output):
    from src.data.manifests import MEALS_CSV,INDEX_CSV,load_meals,load_sensor_index
    from .scope_audit import validated_session_subjects
    root,output=Path(root).resolve(),Path(output).resolve()
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
        raise RuntimeError('commit code and protocol before training')
    output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter(); protected=protection(root)
    manifest_path=root/'outputs/challenger_70/representation_v2/amendment_a6_transformer/t1/manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    validate_partitions(manifest)
    inputs=[manifest_path,Path(MEALS_CSV),Path(INDEX_CSV),Path(__file__).resolve()]
    inputs.extend(root/name for name in ('docs/official_recovery_event_contract_v1.md',
        'scripts/research/official_recovery/scope_audit.py','scripts/research/official_recovery/label_probe.py',
        'scripts/research/official_recovery/eligibility_provenance.py',
        'scripts/research/representation_v2/crossfit.py','scripts/research/representation_v2/provenance.py',
        'src/pipeline/runner.py','src/pipeline/event_stack.py','src/pipeline/candidate_control.py',
        'src/pipeline/context_features.py','src/data/manifests.py','src/pipeline/features/macro.py'))
    inputs.extend(root/'cache'/kind/f'fold0_{split}.npz' for kind in ('slide','micro15')
                  for split in ('train','meal_train','no_meal_train'))
    inputs.append(root/'cache/splits/fold0.json')
    hashes={str(p.relative_to(root)):_sha(p) for p in inputs}
    _write(output/'plan.json',dict(id='EVENT-COVERAGE-V1',source_sha256=hashes,protected=protected,
        code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        outer_calls=0,wrappers=manifest['wrappers'],window_labels='legacy unchanged',
        verifier_blend=.5,expected_window_fits=120,expected_verifier_fits=48))
    def budget():
        if time.perf_counter()-start>7200 or shutil.disk_usage(root).free<10*1024**3:
            raise RuntimeError('event probe wall/disk budget exceeded')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>64*1024**2:
            raise RuntimeError('event probe evidence budget exceeded')
    records=[]; completed=0
    try:
        validated_session_subjects(load_sensor_index().to_dict('records'))
        eligible_sids=set()
        for split in ('meal_train','no_meal_train'):
            with np.load(root/'cache/slide'/f'fold0_{split}.npz',allow_pickle=False) as cache:
                eligible_sids.update(json.loads(str(w))[0] for w in cache['wid'])
        eligibility_before=eligibility_inputs(root,eligible_sids)
        _write(output/'eligibility_inputs.json',eligibility_before)
        config=registered_e03_config(0,workers=4)
        data=runner.FilesystemDataSource(root).load_outer_train_only(config)
        allowed=set(manifest['identities']['t24']); forbidden=set(manifest['identities']['v7'])|set(manifest['identities']['outer'])
        data=slice_outer_train(data,allowed)
        if set(data.subject_by_session.values())!=allowed or allowed&forbidden:
            raise ValueError('T24 population mismatch')
        all_truths=[(normalized_id(r.externalid),int(r.before_ms),int(r.after_ms)) for r in load_meals().itertuples()]
        truths=partition_truths(all_truths,allowed)
        _write(output/'population.json',dict(subject_by_session=data.subject_by_session,
            legacy_eligible_truths=[asdict(t) for t in data.train_truths],complete_truths=truths,
            session_bounds=data.session_bounds_by_sid))
        frozen={}
        for wrapper in manifest['wrappers']:
            fold=wrapper['id']; train=wrapper['train']; health=wrapper['health']
            tdata=slice_outer_train(data,train)
            pc=[];pg=[];ps={a:[] for a in ('old','full')};policy_graph={}
            seen=[]
            for inner,split in enumerate(wrapper['policy_splits']):
                budget();fit,valid=split['train'],split['score']
                if set(fit)&set(valid) or set(fit)|set(valid)!=set(train):
                    raise ValueError('invalid policy OOF split')
                print(f'event health{fold} policy{inner}: train{len(fit)}/score{len(valid)}',flush=True)
                ledger,scores,graph=_paired_split(tdata,config,fit,valid,set(health)|forbidden,truths,
                    output,f'fold{fold}_policy{inner}')
                seen.extend(valid);completed+=1
                pc.extend(ledger.candidates);pg.extend(ledger.groups)
                for arm in ps: ps[arm].extend(scores[arm])
                for name,node in graph.items():
                    policy_graph[f'p{inner}_{name}']=dict(fit_subjects=node['fit_subjects'],
                        parents=[f'p{inner}_{p}' for p in node['parents']])
            if sorted(seen)!=sorted(train) or set(pg)!=set(train):
                raise ValueError('policy OOF coverage mismatch')
            policy_graph['policy']={'fit_subjects':list(train),'parents':[f'p{i}_score' for i in range(3)]}
            check_graph(policy_graph,'policy',set(health)|forbidden,set(train))
            frozen[fold]={'old':_choose_policy(pc,np.asarray(ps['old']),pg,tdata.train_truths,config),
                'full':choose_complete_policy(pc,np.asarray(ps['full']),pg,partition_truths(truths,set(train)),
                    data.subject_by_session,config)}
            _write(output/f'fold{fold}_policies.json',dict(policies=frozen[fold],graph=policy_graph))
            print(f'event health{fold}: paired policies frozen',flush=True)
        # No health refit or outcome inspection until ALL policy files are frozen.
        for wrapper in manifest['wrappers']:
            budget();fold=wrapper['id'];train=wrapper['train'];health=wrapper['health']
            print(f'event health{fold}: refit train{len(train)}/health{len(health)}',flush=True)
            ledger,scores,graph=_paired_split(data,config,train,health,forbidden,truths,output,f'fold{fold}_health')
            completed+=1
            predictions={a:decode(ledger.candidates,scores[a],ledger.groups,frozen[fold][a]) for a in scores}
            health_truths=partition_truths(truths,set(health))
            events={a:complete_metrics(predictions[a],data.subject_by_session,health_truths) for a in scores}
            subjects=[]
            for subject in health:
                gt=partition_truths(health_truths,{subject})
                subjects.append(dict(subject=subject,events={a:complete_metrics(
                    [p for p in predictions[a] if data.subject_by_session[p.sid]==subject],data.subject_by_session,gt)
                    for a in scores}))
            record=dict(fold=fold,events=events,subject_events=subjects,truths=health_truths,
                predictions={a:[asdict(p) for p in pred] for a,pred in predictions.items()})
            records.append(events)
            _write(output/f'fold{fold}_report.json',record)
            _write(output/'state.json',dict(state='IN_PROGRESS',completed_splits=completed,health_folds=len(records)))
            print('paired complete-meal counts '+json.dumps(events),flush=True)
        budget()
        if (protected!=protection(root) or hashes!={str(p.relative_to(root)):_sha(p) for p in inputs}
                or eligibility_before!=eligibility_inputs(root,eligible_sids)):
            raise ValueError('protected release or source changed')
        result=dict(state='COMPLETE',evidence_kind='T24_NESTED_EVENT_COMPLETE_MEAL_DEVELOPMENT_ONLY',
            **assess_gate(records),elapsed_seconds=time.perf_counter()-start,completed_splits=completed,
            protected_unchanged=True,outer_calls=0)
        _write(output/'summary.json',result);_write(output/'state.json',dict(state='COMPLETE'))
        return result
    except BaseException as exc:
        _write(output/'state.json',dict(state='INCOMPLETE',error=repr(exc),completed_splits=completed,health_folds=len(records)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=4):
        print(json.dumps(execute(Path(__file__).resolve().parents[3],args.output),ensure_ascii=False),flush=True)
