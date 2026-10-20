"""GT-aware explanation of frozen complete-label health misses; never a challenger."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from .day_budget_probe import restore_candidates
from .event_probe import decode, partition_truths, subject_events
from .scope_audit import _sha, match_events, protection
from src.pipeline.event_stack import EventRef


def audit_candidates(candidates,scores,groups,policy,mapping,truths,eligible,bounds):
    raw=subject_events([c.event for c in candidates],mapping)
    final=subject_events(decode(candidates,np.asarray(scores),groups,policy),mapping)
    matched={ti for _,ti in match_events(final,truths)}
    subject_bounds=defaultdict(list)
    for sid,(a,b) in bounds.items(): subject_bounds[mapping[sid]].append((a,b))
    rows=[]
    for index,gt in enumerate(truths):
        subject,start,end=gt
        reachable=[i for i,event in enumerate(raw) if match_events([event],[gt])]
        best_any_iou=0.
        for s,a,b in raw:
            if s==subject:
                best_any_iou=max(best_any_iou,max(0,min(b,end)-max(a,start))/(max(b,end)-min(a,start)))
        best=max((float(scores[i]) for i in reachable),default=None)
        if index in matched: mechanism='MATCHED_TP'
        elif not reachable: mechanism='NO_MATCHABLE_RAW_CANDIDATE'
        elif best<policy['threshold']: mechanism='SCORE_BELOW_FINAL_THRESHOLD'
        else: mechanism='CONTROL_OR_ONE_TO_ONE_COMPETITION'
        rows.append(dict(subject=subject,start_ms=start,end_ms=end,legacy_eligible=gt in eligible,
            raw_reachable=bool(reachable),raw_matching_candidates=len(reachable),max_raw_score=best,
            best_any_raw_iou=best_any_iou,event_threshold=policy['threshold'],
            max_score_margin=None if best is None else best-policy['threshold'],
            single_file_contains_gt=any(a<=start and b>=end for a,b in subject_bounds[subject]),
            mechanism=mechanism))
    return rows


def execute(root,source,policies,output):
    root,source,policies,output=map(Path,(root,source,policies,output))
    output.mkdir(parents=True,exist_ok=False)
    protected=protection(root)
    def read(path): return json.loads(path.read_text(encoding='utf-8'))
    source_summary=read(source/'summary.json');budget_summary=read(policies/'summary.json')
    if source_summary.get('state')!='COMPLETE' or budget_summary.get('state')!='COMPLETE':
        raise ValueError('completed frozen source/policy evidence required')
    population=read(source/'population.json');mapping=population['subject_by_session']
    bounds=population['session_bounds'];truths=[tuple(t) for t in population['complete_truths']]
    eligible=set(subject_events([EventRef(**t) for t in population['legacy_eligible_truths']],mapping))
    files=[source/'population.json',source/'summary.json',policies/'summary.json',Path(__file__).resolve()]
    files.extend(source/f'fold{i}_health_scores.json' for i in range(3))
    files.extend(policies/f'fold{i}_policies.json' for i in range(3))
    hashes={str(p.resolve().relative_to(root.resolve())):_sha(p) for p in files}
    rows=[]
    for fold in range(3):
        ledger=read(source/f'fold{fold}_health_scores.json')
        policy=read(policies/f'fold{fold}_policies.json')['policies']['subject']
        targets=partition_truths(truths,set(ledger['score']))
        rows.extend(dict(fold=fold,**r) for r in audit_candidates(restore_candidates(ledger['candidates']),
            ledger['scores']['old'],ledger['groups'],policy,mapping,targets,eligible,bounds))
    counts=dict(Counter(r['mechanism'] for r in rows))
    if (counts.get('MATCHED_TP',0)!=budget_summary['aggregate']['subject']['tp'] or
            len(rows)!=budget_summary['aggregate']['subject']['truths']):
        raise ValueError('diagnostic cannot reproduce the frozen policy counts')
    if protected!=protection(root) or hashes!={str(p.resolve().relative_to(root.resolve())):_sha(p) for p in files}:
        raise ValueError('source or protected artifacts changed')
    summary=dict(evidence_kind='DIAGNOSTIC_ONLY_USES_GT_NO_NEW_F1',source_sha256=hashes,
        truths=len(rows),mechanisms=counts,
        by_scope={name:dict(Counter(r['mechanism'] for r in rows if r['legacy_eligible']==flag))
                  for name,flag in [('legacy_eligible',True),('previously_excluded',False)]},
        fn_without_single_file_containment=sum(r['mechanism']!='MATCHED_TP' and not r['single_file_contains_gt'] for r in rows),
        subject_event_scores_frozen=True,model_fits=0,outer_calls=0,protected_unchanged=True)
    with (output/'gt_attribution.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('source','policies','output'): parser.add_argument('--'+flag,type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(execute(Path(__file__).resolve().parents[3],args.source,args.policies,args.output),ensure_ascii=False))
