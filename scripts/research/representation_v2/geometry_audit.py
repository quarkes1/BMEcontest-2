"""Exploratory, frozen policy OOF geometry and exact-IoU compatibility audit."""
from dataclasses import asdict
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from scripts.research.representation_v2.explore import write_json
from scripts.research.representation_v2.stability_audit import allowed_input,restore_candidate,validate_policy_graph
from src.pipeline.candidate_control import CandidateAdmissionConfig,admit_candidates
from src.pipeline.event_stack import EventRef,compute_event_metrics

SOURCE_COMMIT='46d7c18ab11da2da025934a73f3c273e68c2aabb'


def release_boundary_check(root):
    path=Path(root)/'outputs/challenger_70/e04_verifier_audit/tp_reference.csv'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!='e963fd43964c948f05f11c199923469200d4ab1eda433b473fdb5a405809cc41':
        raise RuntimeError('sealed TP export changed')
    with path.open(encoding='utf-8',newline='') as handle:rows=list(csv.DictReader(handle))
    counts=dict(Counter(r['fold'] for r in rows))
    if counts!={'0':16,'1':30,'2':15,'3':25,'4':28} or len({(r['fold'],r['final_index']) for r in rows})!=114:
        raise RuntimeError('release TP export incomplete')
    exact=0; values=[]
    for r in rows:
        prefix,start,end=r['matched_truth_key'].rsplit(':',2)
        if prefix.split(':',1)[1]!=r['session_id']:raise RuntimeError('release match crosses sessions')
        ps,pe,ts,te=int(r['start_ms']),int(r['end_ms']),int(start),int(end)
        overlap=max(0,min(pe,te)-max(ps,ts)); union=max(pe,te)-min(ps,ts)
        if union<=0 or 4*overlap<union:raise RuntimeError('invalid incumbent TP match')
        value=overlap/union
        if abs(value-float(r['matched_iou']))>1e-12:raise RuntimeError('exported IoU differs from geometry')
        exact+=4*overlap==union; values.append(value)
    return {'diagnostic_only':True,'source_sha256':digest,'matched_predictions':114,
            'fold_tp_counts':counts,'exact_quarter_matches':exact,'minimum_matched_iou':min(values),
            'frozen_predictions_tp_unchanged_under_strict_rule':exact==0,
            'scope':'Existing final predictions only; does not prove unchanged training labels or selected policies.'}


def match_events(predictions,truths,*,strict=False):
    pairs=[]
    for i,p in enumerate(predictions):
        for j,t in enumerate(truths):
            if p.sid!=t.sid:continue
            overlap=max(0,min(p.end_ms,t.end_ms)-max(p.start_ms,t.start_ms))
            union=max(p.end_ms,t.end_ms)-min(p.start_ms,t.start_ms)
            if union<=0:raise ValueError('positive event durations required')
            accepted=4*overlap>union if strict else 4*overlap>=union
            if accepted:pairs.append((i,j,overlap/union))
    pairs.sort(key=lambda r:-r[2])
    used_p=set(); used_t=set(); result=[]
    for i,j,iou in pairs:
        if i not in used_p and j not in used_t:
            result.append((i,j,iou)); used_p.add(i); used_t.add(j)
    return result


def boundary_pairs(predictions,truths):
    count=0
    for p in predictions:
        for t in truths:
            if p.sid!=t.sid:continue
            inter=max(0,min(p.end_ms,t.end_ms)-max(p.start_ms,t.start_ms))
            union=max(p.end_ms,t.end_ms)-min(p.start_ms,t.start_ms)
            count+=4*inter==union
    return int(count)


def clip_support(candidate):
    e=candidate.event
    if candidate.macro is None or candidate.micro is None:return e
    if candidate.macro.event!=e or candidate.micro.event.sid!=e.sid:
        raise ValueError('expected canonical macro boundary and same-session micro support')
    start=max(e.start_ms,candidate.micro.event.start_ms)
    end=min(e.end_ms,candidate.micro.event.end_ms)
    if end<=start:raise ValueError('dual-source support must overlap')
    return EventRef(e.sid,start,end)


def event_diagnostics(predictions,truths,*,strict=False,common=None):
    matches=match_events(predictions,truths,strict=strict)
    selected=matches if common is None else [m for m in matches if m[1] in common]
    used={m[0] for m in matches}
    fp_durations=[(p.end_ms-p.start_ms)/1000 for i,p in enumerate(predictions) if i not in used]
    return {'tp':len(matches),'fp':len(predictions)-len(matches),'fn':len(truths)-len(matches),
            'start_mae_ms':float(np.mean([abs(predictions[i].start_ms-truths[j].start_ms) for i,j,_ in selected])) if selected else None,
            'end_mae_ms':float(np.mean([abs(predictions[i].end_ms-truths[j].end_ms) for i,j,_ in selected])) if selected else None,
            'unmatched_fp_total_duration_s':float(sum(fp_durations)),
            'exact_boundary_pairs':boundary_pairs(predictions,truths)}


def compare_geometry(baseline,clipped,truths):
    a={j for _,j,_ in match_events(baseline,truths)}
    b={j for _,j,_ in match_events(clipped,truths)}
    common=a&b
    return {'baseline':event_diagnostics(baseline,truths),'clipped':event_diagnostics(clipped,truths),
            'baseline_strict':event_diagnostics(baseline,truths,strict=True),
            'clipped_strict':event_diagnostics(clipped,truths,strict=True),
            'matched_gt_gained':len(b-a),'matched_gt_lost':len(a-b),'common_matched_gt':len(common),
            'gained_gt':[asdict(truths[j]) for j in sorted(b-a)],'lost_gt':[asdict(truths[j]) for j in sorted(a-b)],
            'baseline_common_gt':event_diagnostics(baseline,truths,common=common),
            'clipped_common_gt':event_diagnostics(clipped,truths,common=common),
            'changed_intervals':sum(a!=b for a,b in zip(baseline,clipped))}


def run(root):
    root=Path(root); source=root/'outputs/challenger_70/representation_v2/amendment_a3_events'
    output=root/'outputs/challenger_70/representation_v2/geometry_policy_audit'
    if output.exists():raise RuntimeError('audit already exists; do not overwrite')
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
        raise RuntimeError('commit diagnostic code/spec first')
    started=time.perf_counter(); hashes={}; files={}; rows=[]
    for k in range(3):
        for name in [f'fold{k}_policies.json',*[f'fold{k}_policy{j}_scores.json' for j in range(3)]]:
            path=source/allowed_input(name); raw=path.read_bytes()
            committed=subprocess.check_output(['git','show',f'{SOURCE_COMMIT}:{path.relative_to(root).as_posix()}'],cwd=root)
            if raw!=committed:raise RuntimeError('sealed input changed')
            hashes[name]=hashlib.sha256(raw).hexdigest(); files[name]=json.loads(raw)
    boundary=release_boundary_check(root)
    write_json(output/'release_boundary_compatibility.json',boundary)
    write_json(output/'plan.json',{'id':'S6-GEOMETRY-OOF-DIAGNOSTIC','diagnostic_only':True,'exploratory':True,
        'source_commit':SOURCE_COMMIT,'input_sha256':hashes,'health_data_read':False,'outer_calls':0,'model_fits':0,
        'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'spec_sha256':hashlib.sha256((root/'docs/challenge_070_geometry_probe.md').read_bytes()).hexdigest()})
    for k in range(3):
        ds=[files[f'fold{k}_policy{j}_scores.json'] for j in range(3)]
        pol=files[f'fold{k}_policies.json']; validate_policy_graph(ds,pol)
        cs=[restore_candidate(c) for d in ds for c in d['candidates']]
        scores=np.asarray([s for d in ds for s in d['scores']['b']])
        groups=[g for d in ds for g in d['groups']]; policy=pol['policies']['b']
        truths=[EventRef(**t) for t in pol['policy_truths']]
        admitted=admit_candidates(cs,scores,groups,CandidateAdmissionConfig(**policy['admission']))
        selected=[i for i in admitted if scores[i]>=policy['threshold']]
        baseline=[cs[i].event for i in selected]; clipped=[clip_support(cs[i]) for i in selected]
        actual=asdict(compute_event_metrics(baseline,truths)); expected=policy['selection_metrics']
        if any(actual[key]!=expected[key] for key in ('n_tp','n_pred','n_true')):
            raise RuntimeError('frozen baseline selection replay failed')
        result=compare_geometry(baseline,clipped,truths)
        if result['baseline']['tp']!=actual['n_tp']:raise RuntimeError('integer matcher differs from incumbent')
        result.update({'wrapper':k,'diagnostic_only':True,'health_data_read':False,
            'candidate_count':len(cs),'dual_candidate_count':sum(c.macro is not None and c.micro is not None for c in cs),
            'raw_candidate_exact_boundary_pairs':boundary_pairs([c.event for c in cs],truths),
            'selected_raw_indices':selected,'original_policy':policy,
            'changed_examples':[{'raw_index':i,'original':asdict(cs[i].event),'clipped':asdict(clip_support(cs[i]))}
                                for i in selected if cs[i].event!=clip_support(cs[i])]})
        rows.append(result); write_json(output/f'fold{k}_audit.json',result)
    if any(hashlib.sha256((source/n).read_bytes()).hexdigest()!=h for n,h in hashes.items()):
        raise RuntimeError('source changed during audit')
    size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    if size>2*1024**2 or time.perf_counter()-started>900:raise RuntimeError('diagnostic budget exceeded')
    summary={'status':'S6_GEOMETRY_DIAGNOSTIC_COMPLETE','diagnostic_only':True,'exploratory':True,
        'health_data_read':False,'outer_calls':0,'model_fits':0,'input_unchanged':True,
        'changed_intervals':sum(r['changed_intervals'] for r in rows),
        'matched_gt_gained_occurrences':sum(r['matched_gt_gained'] for r in rows),
        'matched_gt_lost_occurrences':sum(r['matched_gt_lost'] for r in rows),
        'event_counts_unchanged_all_wrappers':all(r['baseline']['tp']==r['clipped']['tp'] for r in rows),
        'strict_boundary_counts_unchanged_all_wrappers':all(r['baseline']['tp']==r['baseline_strict']['tp'] for r in rows),
        'wall_seconds':time.perf_counter()-started,'artifact_bytes_before_summary':size,
        'note':'Repeated policy populations; occurrences are not independent unique GT. No new F1 headline.'}
    write_json(output/'summary.json',summary); print(json.dumps(summary),flush=True)


if __name__=='__main__':run(Path(__file__).resolve().parents[3])
