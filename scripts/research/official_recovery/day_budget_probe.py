"""Frozen-score policy comparison: subject-global versus subject-day budgets."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .scope_audit import _sha, protection
from .event_probe import assess_gate, check_graph, choose_complete_policy, complete_metrics, decode, partition_truths
from .label_probe import validate_partitions
from src.pipeline.challenger.e03 import registered_e03_config
from src.pipeline.event_stack import CandidateEvent, EventRef, MultiScaleCandidate


def budget_groups(candidates,groups,*,daily):
    if len(candidates)!=len(groups):
        raise ValueError('unaligned budget groups')
    if not daily:
        return list(groups)
    return [f'{group}|{((int(c.event.start_ms)+int(c.event.end_ms))//2+28800000)//86400000}'
            for c,group in zip(candidates,groups)]


def restore_candidates(rows):
    def stream(payload):
        if payload is None: return None
        return CandidateEvent(**{**payload,'event':EventRef(**payload['event']),
                                  'probabilities':tuple(payload['probabilities'])})
    return [MultiScaleCandidate(EventRef(**r['event']),stream(r['macro']),stream(r['micro'])) for r in rows]


def validate_ledger(row,train,score,excluded):
    if (set(row['train'])!=set(train) or set(row['score'])!=set(score)
            or set(train)&(set(score)|set(excluded)) or set(row['groups'])-set(score)):
        raise ValueError('score ledger identity mismatch')
    check_graph(row['graph'],'score',set(score)|set(excluded),set(train))


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _write(path,payload):
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def execute(root,source,output):
    root,source,output=Path(root).resolve(),Path(source).resolve(),Path(output).resolve()
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
        raise RuntimeError('commit code, protocol and source evidence before policy comparison')
    output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();protected=protection(root);records=[]
    try:
        source_summary=_read(source/'summary.json')
        if source_summary.get('state')!='COMPLETE' or source_summary.get('protected_unchanged') is not True:
            raise ValueError('incomplete or unprotected source experiment')
        if not (source/'eligibility_inputs.json').exists():
            raise ValueError('confirmed eligibility provenance required')
        manifest_path=root/'outputs/challenger_70/representation_v2/amendment_a6_transformer/t1/manifest.json'
        manifest=_read(manifest_path);validate_partitions(manifest)
        forbidden=set(manifest['identities']['v7'])|set(manifest['identities']['outer'])
        population=_read(source/'population.json');mapping=population['subject_by_session']
        if set(mapping.values())!=set(manifest['identities']['t24']):
            raise ValueError('source population is not T24')
        truths=[tuple(row) for row in population['complete_truths']]
        config=registered_e03_config(0,workers=4)
        files=sorted(p for p in source.rglob('*') if p.is_file())
        files.extend([manifest_path,Path(__file__).resolve(),root/'docs/official_recovery_day_budget_v1.md',
            root/'scripts/research/official_recovery/event_probe.py',root/'src/pipeline/candidate_control.py',
            root/'scripts/research/official_recovery/scope_audit.py'])
        # Hash health bytes but do not parse their scores until all policies are frozen.
        inputs={str(p.relative_to(root)):_sha(p) for p in files}
        _write(output/'plan.json',dict(id='DAY-BUDGET-V1',source_sha256=inputs,config=asdict(config),
            code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            protected=protected,model_fits=0,outer_calls=0,score_arm='old',policy_target='complete subject truths'))
        def budget():
            if time.perf_counter()-start>600 or shutil.disk_usage(root).free<10*1024**3:
                raise RuntimeError('day budget wall/disk limit exceeded')
            if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>8*1024**2:
                raise RuntimeError('day budget evidence limit exceeded')
        def ledger(path,train,score,excluded):
            row=_read(path);validate_ledger(row,train,score,excluded)
            candidates=restore_candidates(row['candidates'])
            groups=row['groups'];scores=np.asarray(row['scores']['old'],dtype=float)
            if len(candidates)!=len(scores) or any(mapping[c.event.sid]!=g for c,g in zip(candidates,groups)):
                raise ValueError('source candidate/score identity mismatch')
            return candidates,groups,scores
        frozen={}
        for wrapper in manifest['wrappers']:
            budget();fold=wrapper['id'];train=wrapper['train'];health=wrapper['health']
            candidates=[];groups=[];scores=[];seen=[]
            for index,split in enumerate(wrapper['policy_splits']):
                if set(split['train'])|set(split['score'])!=set(train):
                    raise ValueError('policy split population mismatch')
                c,g,p=ledger(source/f'fold{fold}_policy{index}_scores.json',split['train'],split['score'],set(health)|forbidden)
                candidates.extend(c);groups.extend(g);scores.extend(p);seen.extend(split['score'])
            if sorted(seen)!=sorted(train): raise ValueError('policy OOF coverage mismatch')
            targets=partition_truths(truths,set(train));scores=np.asarray(scores)
            frozen[fold]={arm:choose_complete_policy(candidates,scores,budget_groups(candidates,groups,daily=daily),
                targets,mapping,config) for arm,daily in [('subject',False),('day',True)]}
            _write(output/f'fold{fold}_policies.json',dict(policies=frozen[fold],train=train,excluded=sorted(set(health)|forbidden)))
            print(f'day budget health{fold}: policies frozen',flush=True)
        for wrapper in manifest['wrappers']:
            budget();fold=wrapper['id'];train=wrapper['train'];health=wrapper['health']
            candidates,groups,scores=ledger(source/f'fold{fold}_health_scores.json',train,health,forbidden)
            predictions={arm:decode(candidates,scores,budget_groups(candidates,groups,daily=daily),frozen[fold][arm])
                         for arm,daily in [('subject',False),('day',True)]}
            targets=partition_truths(truths,set(health))
            events={arm:complete_metrics(predictions[arm],mapping,targets) for arm in predictions}
            subjects=[]
            for subject in health:
                subject_gt=partition_truths(targets,{subject})
                subjects.append(dict(subject=subject,events={arm:complete_metrics(
                    [p for p in predictions[arm] if mapping[p.sid]==subject],mapping,subject_gt) for arm in predictions}))
            _write(output/f'fold{fold}_report.json',dict(events=events,subject_events=subjects,
                predictions={a:[asdict(p) for p in pred] for a,pred in predictions.items()},truths=targets))
            records.append({'old':events['subject'],'full':events['day']})
            print('day budget complete-meal counts '+json.dumps(events),flush=True)
        budget()
        if protected!=protection(root) or inputs!={str(p.relative_to(root)):_sha(p) for p in files}:
            raise ValueError('source evidence or protected release changed')
        gate=assess_gate(records)
        gate['aggregate']={'subject':gate['aggregate']['old'],'day':gate['aggregate']['full']}
        result=dict(state='COMPLETE',evidence_kind='T24_FROZEN_SCORE_BUDGET_DEVELOPMENT_ONLY',**gate,
            model_fits=0,outer_calls=0,protected_unchanged=True,elapsed_seconds=time.perf_counter()-start)
        _write(output/'summary.json',result);_write(output/'state.json',{'state':'COMPLETE'})
        return result
    except BaseException as exc:
        _write(output/'state.json',dict(state='INCOMPLETE',error=repr(exc),completed_health_folds=len(records)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=4):
        print(json.dumps(execute(Path(__file__).resolve().parents[3],args.source,args.output),ensure_ascii=False))
