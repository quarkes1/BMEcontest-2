"""Measure cached window support and recording-length pressure without signal copies."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import time

import numpy as np

from .scope_audit import _sha, normalized_id, protection, validated_session_subjects


def merge_intervals(intervals):
    merged=[]
    for start,end in sorted(intervals):
        if end<=start:
            raise ValueError('intervals must have positive duration')
        if merged and start<=merged[-1][1]:
            merged[-1]=(merged[-1][0],max(merged[-1][1],end))
        else:
            merged.append((start,end))
    return merged


def covered_ms(intervals,target):
    start,end=target
    if end<=start:
        raise ValueError('target must have positive duration')
    clipped=[(max(start,a),min(end,b)) for a,b in intervals if min(end,b)>max(start,a)]
    return sum(b-a for a,b in merge_intervals(clipped))


def interval_diagnostics(intervals,target):
    start,end=target
    covered=covered_ms(intervals,target)
    return dict(covered_ms=covered,coverage_fraction=covered/(end-start),
        overlapping_intervals=sum(min(end,b)>max(start,a) for a,b in intervals),
        single_interval_contains=any(a<=start and b>=end for a,b in intervals))


def active_days(intervals,*,utc_offset_hours=8):
    offset=int(utc_offset_hours*3600000)
    days=set()
    for start,end in merge_intervals(intervals):
        days.update(range((start+offset)//86400000,(end-1+offset)//86400000+1))
    return len(days)


def audit_coverage(root,scope_path):
    from src.data.manifests import load_sensor_index, INDEX_CSV
    root,scope_path=Path(root).resolve(),Path(scope_path).resolve()
    scope=json.loads(scope_path.read_text(encoding='utf-8'))
    if scope.get('evidence_kind')!='FROZEN_PREDICTION_SCOPE_DIAGNOSTIC_NOT_NEW_MODEL_PERFORMANCE':
        raise ValueError('scope audit identity missing')
    protected=protection(root)
    started=time.perf_counter()
    source_hashes={str(scope_path.relative_to(root)):_sha(scope_path),
                   str(Path(INDEX_CSV).relative_to(root)):_sha(Path(INDEX_CSV)),
                   str(Path(__file__).resolve().relative_to(root)):_sha(Path(__file__).resolve())}
    subjects={row['subject']:row['fold'] for row in scope['per_subject']}
    index=load_sensor_index()
    by_sid=validated_session_subjects(index.to_dict('records'))
    recorded=defaultdict(list)
    for r in index.to_dict('records'):
        a,b=int(r['timeStamp.startTime']),int(r['timeStamp.endTime'])
        if b>a:
            recorded[normalized_id(r['externalid'])].append((a,b))
    macro,micro,used_micro=defaultdict(list),defaultdict(list),defaultdict(list)
    micro_without_macro=defaultdict(list)
    counters=[]
    for fold in range(5):
        macro_sids=set()
        row_counts=Counter()
        for kind in ('slide','micro15'):
            path=root/'cache'/kind/f'fold{fold}_val.npz'
            source_hashes[str(path.relative_to(root))]=_sha(path)
            with np.load(path,allow_pickle=False) as arrays:
                refs=arrays['wid']
            for wid in refs:
                sid,start,end=json.loads(str(wid))
                subject=by_sid[sid]
                if subject not in subjects:
                    row_counts[kind+'_outside_audited_subjects']+=1
                    continue
                if subjects[subject]!=fold:
                    raise ValueError('cached window subject crosses its held-out fold')
                interval=(int(start),int(end))
                if interval[1]<=interval[0]:
                    raise ValueError('cached window interval not positive')
                if kind=='slide':
                    macro[subject].append(interval)
                    macro_sids.add(sid)
                else:
                    micro[subject].append(interval)
                    if sid in macro_sids:
                        used_micro[subject].append(interval)
                    else:
                        micro_without_macro[subject].append(interval)
                row_counts[kind]+=1
            del refs
        counters.append(dict(fold=fold,**row_counts))
    # Per-window arrays are reduced to unions before per-meal measurements.
    unions={name:{s:merge_intervals(values) for s,values in groups.items()}
            for name,groups in [('macro',macro),('all_micro',micro),('legacy_used_micro',used_micro),
                                 ('micro_without_macro_session',micro_without_macro)]}
    truths=[]
    for original in scope['truth_rows']:
        subject=original['subject']
        target=(original['start_ms'],original['end_ms'])
        row=dict(original)
        row['index_interval_union_fraction']=interval_diagnostics(recorded[subject],target)['coverage_fraction']
        for name,groups in unions.items():
            row[name+'_support_fraction']=interval_diagnostics(groups.get(subject,()),target)['coverage_fraction']
        row['micro_support_discarded_by_macro_session_filter']=(row['all_micro_support_fraction']>
                                                               row['legacy_used_micro_support_fraction'])
        truths.append(row)
    persons=[]
    for old in scope['per_subject']:
        subject=old['subject']
        intervals=merge_intervals(unions['macro'].get(subject,[])+unions['all_micro'].get(subject,[]))
        persons.append(dict(old,cached_window_support_hours=sum(b-a for a,b in intervals)/3600000,
            cached_window_active_days_utc8=active_days(intervals),
            valid_gt_over_deployment_cap6=max(old['truths']-6,0),
            predictions_at_or_above_deployment_cap6=old['predictions']>=6))
    missing=[r for r in truths if not r['legacy_eligible']]
    summary=dict(schema_version=1,evidence_kind='CACHED_WINDOW_SUPPORT_DIAGNOSTIC_NOT_MODEL_GAIN',
        truths=len(truths),legacy_excluded_truths=len(missing),
        excluded_no_index_overlap=sum(r['index_interval_union_fraction']==0 for r in missing),
        excluded_partial_index_union=sum(0<r['index_interval_union_fraction']<.999 for r in missing),
        excluded_with_80pct_macro_window_support=sum(r['macro_support_fraction']>=.8 for r in missing),
        excluded_with_80pct_micro_window_support=sum(r['all_micro_support_fraction']>=.8 for r in missing),
        excluded_no_cached_micro_support=sum(r['all_micro_support_fraction']==0 for r in missing),
        truths_with_discarded_micro_support=sum(r['micro_support_discarded_by_macro_session_filter'] for r in truths),
        micro_windows_without_macro_session=sum(len(v) for v in micro_without_macro.values()),
        subjects_with_more_than6_valid_labels=sum(r['truths']>6 for r in persons),
        theoretical_unavoidable_fn_with_six_cap=sum(r['valid_gt_over_deployment_cap6'] for r in persons),
        per_fold_windows=counters,per_subject=persons,truth_rows=truths,
        source_sha256=source_hashes,protected_sha256=protected,
        raw_signal_reads=0,training_fits=0,policy_selections=0,
        elapsed_seconds=time.perf_counter()-started)
    if protection(root)!=protected:
        raise ValueError('protected release changed')
    for relative,digest in source_hashes.items():
        if _sha(root/relative)!=digest:
            raise ValueError('audit input changed')
    summary['protected_unchanged']=True
    return summary


def write_report(output,report):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name in ('truth_rows','per_subject'):
        rows=report[name]
        with (output/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    lines=['# 记录长度与缓存窗口支持审计','',
        '不训练、不改变评分或策略、不产生挑战模型成绩。窗口覆盖是已缓存窗口区间的并集，不是原始信号有效率；索引覆盖也不等于有效IMU。','',
        '| 观察 | 数量 |','|---|---:|']
    for key in ('truths','legacy_excluded_truths','excluded_no_index_overlap',
                'excluded_partial_index_union','excluded_with_80pct_macro_window_support',
                'excluded_with_80pct_micro_window_support','excluded_no_cached_micro_support',
                'truths_with_discarded_micro_support','micro_windows_without_macro_session',
                'subjects_with_more_than6_valid_labels','theoretical_unavoidable_fn_with_six_cap'):
        lines.append(f'| {key} | {report[key]} |')
    lines.extend(['','每人6候选只作部署配置的理论上限检查；旧五折策略并非全部cap6，不能将此上限当作实际漏检归因。',
        '微窗支持被宏窗会话过滤丢弃，不等于恢复该支持一定形成正确事件；须在训练内重新验证候选、评分和FP成本。',
        '完整有效标签保留未覆盖餐；未知原始采样情况不作推断。所有输入和保护哈希见summary.json。'])
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    report=audit_coverage(Path(__file__).resolve().parents[3],args.scope)
    write_report(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if not isinstance(v,(list,dict))},ensure_ascii=False))


if __name__=='__main__':
    main()
