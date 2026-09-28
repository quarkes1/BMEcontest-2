"""Single-factor neutral fallback, reusing sealed S2 scores without model fits."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np
from sklearn.metrics import average_precision_score,log_loss
from threadpoolctl import threadpool_limits

from scripts.research.representation_v2.event_probe import choose_policy,decode_policy,event_gate,counts_metrics
from scripts.research.representation_v2.explore import write_json
from scripts.research.representation_v2.provenance import assert_no_heldout_ancestors
from scripts.research.representation_v2.stability_audit import validate_ledger,validate_policy_graph,restore_candidate
from src.pipeline.challenger.e03 import registered_e03_config
from src.pipeline.challenger.protection import protected_snapshot
from src.pipeline.event_stack import EventRef,compute_event_metrics


SOURCE_COMMIT='46d7c18ab11da2da025934a73f3c273e68c2aabb'


def neutral_scores(scores,motion_valid):
    mask=np.asarray(motion_valid)
    arrays={a:np.asarray(scores[a],dtype=float) for a in ('b','r','f')}
    if mask.dtype.kind!='b' or mask.ndim!=1:raise ValueError('boolean motion-valid vector required')
    if any(p.shape!=mask.shape or not np.isfinite(p).all() or ((p<0)|(p>1)).any() for p in arrays.values()):
        raise ValueError('aligned finite probability vectors required')
    result={a:p.copy() for a,p in arrays.items()}
    for a in ('r','f'):result[a][~mask]=arrays['b'][~mask]
    return result


def check_read_phase(name,frozen_folds):
    if not re.fullmatch(r'fold[0-2]_(policy[0-2]_scores|policies|health_scores|report)\.json',name):
        raise ValueError('unregistered evidence path')
    if ('health_scores' in name or name.endswith('_report.json')) and set(frozen_folds)!={0,1,2}:
        raise RuntimeError('all policies must be frozen before any health reads')
    return name


def validate_health_topology(health,policy_ledgers):
    train=set(s for d in policy_ledgers for s in d['score'])
    if set(health['train'])!=train or train&(set(health['score'])|set(health['excluded'])):
        raise ValueError('health and policy training topology mismatch')
    if any(set(d['excluded'])!=set(health['score'])|set(health['excluded']) for d in policy_ledgers):
        raise ValueError('policy exclusions do not match health and original validation')


def neutral_gate(rows,old_rows):
    if len(old_rows)!=3:raise ValueError('three frozen reference folds required')
    result=event_gate(rows)
    old=counts_metrics([r['events']['f'] for r in old_rows])
    result['old_fd_health_events']=old
    result['passed']=bool(result['passed'] and result['pooled_health_events']['f']['f1']>old['f1'])
    result['status']='S4_SUPPORTS_WIDER_INNER_EVENT_VALIDATION' if result['passed'] else 'S4_NO_SUPPORTED_NEUTRAL_FALLBACK'
    return result


def subject_event_counts(candidates,groups,truths,predictions,reference):
    """Retain sealed truth denominators, even for sessions with no candidates."""
    sid_subject={}
    for c,g in zip(candidates,groups):
        if c.event.sid in sid_subject and sid_subject[c.event.sid]!=g:raise ValueError('session has multiple subjects')
        sid_subject[c.event.sid]=g
    result=[]
    for ref in reference:
        subject=ref['subject']; true_count=ref['events']['b']['n_true']
        observed_truths=[t for t in truths if sid_subject.get(t.sid)==subject]
        if len(observed_truths)>true_count:raise ValueError('subject truth denominator mismatch')
        metrics={}
        for a,pred in predictions.items():
            m=asdict(compute_event_metrics([e for e in pred if sid_subject[e.sid]==subject],observed_truths))
            m['n_true']=true_count
            m['sensitivity']=m['n_tp']/true_count if true_count else 0.
            denominator=m['sensitivity']+m['ppv']
            m['f1']=2*m['sensitivity']*m['ppv']/denominator if denominator else 0.
            m['fp']=m['n_pred']-m['n_tp']; m['fn']=true_count-m['n_tp']
            metrics[a]=m
        result.append({'subject':subject,'events':metrics})
    return result


def assert_baseline_replay(current,old):
    for key in ('events','predictions','candidate_metrics'):
        if current[key]['b']!=old[key]['b']:raise RuntimeError(f'baseline {key} changed')
    now={r['subject']:r['events']['b'] for r in current['subject_events']}
    before={r['subject']:r['events']['b'] for r in old['subject_events']}
    if set(now)!=set(before) or len(now)!=len(current['subject_events']):raise RuntimeError('baseline subject set changed')
    for subject,metrics in before.items():
        if any(now[subject].get(key)!=value for key,value in metrics.items()):
            raise RuntimeError('baseline subject metrics changed')


def load_frozen_policies(output,expected_marker_sha):
    output=Path(output); raw=(output/'policy_phase_frozen.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_marker_sha:raise RuntimeError('policy phase marker changed')
    marker=json.loads(raw)
    expected={f'fold{k}_policy_selection.json' for k in range(3)}
    if marker['folds']!=[0,1,2] or marker['health_files_read']!=0 or set(marker['policy_file_sha256'])!=expected:
        raise RuntimeError('policy phase freeze is incomplete')
    result={}
    for k in range(3):
        name=f'fold{k}_policy_selection.json'; raw=(output/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=marker['policy_file_sha256'][name]:raise RuntimeError('frozen policy artifact changed')
        value=json.loads(raw)
        if value['fold']!=k or set(value['policies'])!={'b','r','f'}:raise RuntimeError('frozen policy contract changed')
        result[k]=value
    return result


def run(root):
    root=Path(root); source=root/'outputs/challenger_70/representation_v2/amendment_a3_events'
    output=root/'outputs/challenger_70/representation_v2/amendment_a3_neutral'
    if output.exists():raise RuntimeError('S4 attempt exists; do not overwrite')
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():raise RuntimeError('commit protocol/code first')
    started=time.perf_counter(); snapshot=protected_snapshot(root); frozen=set(); hashes={}; policy_store=[]; rows=[]; old_rows=[]

    def read(name):
        path=source/check_read_phase(name,frozen); raw=path.read_bytes()
        committed=subprocess.check_output(['git','show',f'{SOURCE_COMMIT}:{path.relative_to(root).as_posix()}'],cwd=root)
        if raw!=committed:raise ValueError('evidence differs from frozen S2 source commit')
        hashes[name]=hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def budget_ok():
        return time.perf_counter()-started<=1800 and sum(p.stat().st_size for p in output.rglob('*') if p.is_file())<=10*1024**2

    plan={'id':'RV2-A3-S4-001','source_commit':SOURCE_COMMIT,'training_calls':0,'outer_calls':0,'optimizer_steps':0,
          'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
          'spec_sha256':hashlib.sha256((root/'docs/challenge_070_amendment_rv2_a3_neutral.md').read_bytes()).hexdigest(),
          'protected_digest':hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest()}
    write_json(output/'plan.json',plan)
    try:
        config=registered_e03_config(0,workers=4)
        for k in range(3):
            if not budget_ok():raise RuntimeError('S4 budget exceeded')
            print(f'S4 policy{k}: choose on frozen policy OOF only',flush=True)
            ledgers=[read(f'fold{k}_policy{j}_scores.json') for j in range(3)]
            old_policy=read(f'fold{k}_policies.json'); validate_policy_graph(ledgers,old_policy)
            candidates=[restore_candidate(c) for d in ledgers for c in d['candidates']]
            groups=[g for d in ledgers for g in d['groups']]
            transformed=[neutral_scores(d['scores'],d['motion_valid']) for d in ledgers]
            scores={a:np.concatenate([d[a] for d in transformed]) for a in ('b','r','f')}
            truths=[EventRef(**t) for t in old_policy['policy_truths']]
            policies={a:choose_policy(candidates,scores[a],groups,truths,config) for a in scores}
            if policies['b']!=old_policy['policies']['b']:raise RuntimeError('unchanged baseline policy failed exact replay')
            graph={key:value for key,value in old_policy['graph'].items() if key!='policy'}
            for j in range(3):graph[f'p{j}_neutral']={'fit_subjects':[],'parents':[f'p{j}_score']}
            graph['neutral_policy']={'fit_subjects':old_policy['graph']['policy']['fit_subjects'],
                                     'parents':[f'p{j}_neutral' for j in range(3)]}
            assert_no_heldout_ancestors('neutral_policy',set(ledgers[0]['excluded']),graph)
            result={'fold':k,'policies':policies,'graph':graph,'source_commit':SOURCE_COMMIT,
                    'source_files':[f'fold{k}_policy{j}_scores.json' for j in range(3)],
                    'policy_scores':{a:p.tolist() for a,p in scores.items()},'input_sha256':dict(hashes)}
            path=output/f'fold{k}_policy_selection.json'; write_json(path,result)
            # Persist and verify before opening the health phase gate.
            saved=json.loads(path.read_text())
            if saved['policies']!=policies:raise RuntimeError('policy persistence mismatch')
            policy_store.append((ledgers,old_policy,policies,hashlib.sha256(path.read_bytes()).hexdigest()))
            frozen.add(k)
        write_json(output/'policy_phase_frozen.json',{'folds':sorted(frozen),
                   'policy_file_sha256':{f'fold{k}_policy_selection.json':p[3] for k,p in enumerate(policy_store)},
                   'health_files_read':0})
        marker_sha=hashlib.sha256((output/'policy_phase_frozen.json').read_bytes()).hexdigest()
        for k,(ledgers,old_policy,policies,policy_sha) in enumerate(policy_store):
            if not budget_ok():raise RuntimeError('S4 budget exceeded')
            policies=load_frozen_policies(output,marker_sha)[k]['policies']
            health=read(f'fold{k}_health_scores.json'); old=read(f'fold{k}_report.json')
            validate_ledger(health); validate_health_topology(health,ledgers)
            candidates=[restore_candidate(c) for c in health['candidates']]
            truths=[EventRef(**t) for t in old['truths']]
            for a in ('b','r','f'):
                replay=decode_policy(candidates,np.asarray(health['scores'][a]),health['groups'],old_policy['policies'][a])
                if [asdict(e) for e in replay]!=old['predictions'][a] or asdict(compute_event_metrics(replay,truths))!=old['events'][a]:
                    raise RuntimeError('old predictions/metrics failed replay')
            scores=neutral_scores(health['scores'],health['motion_valid'])
            predictions={a:decode_policy(candidates,scores[a],health['groups'],policies[a]) for a in scores}
            events={a:asdict(compute_event_metrics(predictions[a],truths)) for a in scores}
            if [asdict(e) for e in predictions['b']]!=old['predictions']['b']:raise RuntimeError('baseline predictions changed')
            subject_events=subject_event_counts(candidates,health['groups'],truths,predictions,old['subject_events'])
            if set(s['subject'] for s in subject_events)!=set(health['score']):raise ValueError('subject coverage mismatch')
            for a in scores:
                pooled=counts_metrics([s['events'][a] for s in subject_events])
                if any(pooled[key]!=events[a][key] for key in ('n_tp','n_pred','n_true')):raise ValueError('subject/global count mismatch')
            record={'fold':k,'events':events,'subject_events':subject_events,
                    'scores':{a:p.tolist() for a,p in scores.items()},'predictions':{a:[asdict(e) for e in p] for a,p in predictions.items()},
                    'candidate_metrics':{a:{'ap':float(average_precision_score(health['labels'],p)),
                                            'bce':float(log_loss(health['labels'],p,labels=[0,1]))} for a,p in scores.items()},
                    'source_health_file':f'fold{k}_health_scores.json','source_health_sha256':hashes[f'fold{k}_health_scores.json'],
                    'policy_file_sha256':policy_sha,'checks':{'integrity':True,'resource':bool(budget_ok())},'outer_calls':0}
            assert_baseline_replay(record,old)
            write_json(output/f'fold{k}_report.json',record); rows.append(record); old_rows.append(old)
            print('S4 health'+str(k)+' '+json.dumps(events),flush=True)
        result=neutral_gate(rows,old_rows)
        if any(hashlib.sha256((source/name).read_bytes()).hexdigest()!=value for name,value in hashes.items()):raise RuntimeError('source changed')
        if any(hashlib.sha256((output/f'fold{k}_policy_selection.json').read_bytes()).hexdigest()!=p[3] for k,p in enumerate(policy_store)):
            raise RuntimeError('policy changed after freeze')
        if protected_snapshot(root)!=snapshot:raise RuntimeError('protected release changed')
        if not budget_ok():raise RuntimeError('final S4 budget exceeded')
        write_json(output/'source_manifest.json',{'source_commit':SOURCE_COMMIT,'input_sha256':hashes})
        result.update({'training_calls':0,'optimizer_steps':0,'protected_unchanged':True,'source_unchanged':True,
                       'wall_seconds':time.perf_counter()-started,'persistent_token_cache_bytes':0,
                       'artifact_bytes_before_summary':sum(p.stat().st_size for p in output.iterdir() if p.is_file())})
        write_json(output/'summary.json',result); write_json(output/'state.json',{'status':result['status'],'outer_calls':0})
        print(json.dumps(result),flush=True)
    except Exception as exc:
        write_json(output/'state.json',{'status':'S4_INCOMPLETE','error':str(exc),'frozen_folds':sorted(frozen),
                                       'completed_health_folds':len(rows),'outer_calls':0})
        raise


if __name__=='__main__':
    with threadpool_limits(limits=4):run(Path(__file__).resolve().parents[3])
