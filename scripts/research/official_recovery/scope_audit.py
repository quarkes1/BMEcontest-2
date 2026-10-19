"""Re-evaluate frozen predictions under explicit scopes, without fitting a model."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

RUN_KEY = '160afaf81debf1ee'


def normalized_id(value):
    result = ''.join(str(value).split()).upper()
    if not result:
        raise ValueError('empty subject ID')
    return result


def fold_subjects(records):
    result, originals = {}, {}
    for record in records:
        fold = int(record['fold'])
        for original in record['subjects']:
            subject = normalized_id(original)
            if subject in result and result[subject] != fold:
                raise ValueError('subject belongs to multiple folds')
            if subject in originals and originals[subject] != original:
                raise ValueError('normalized ID aliases require explicit resolution')
            if subject in result:
                raise ValueError('duplicate subject in fold')
            result[subject], originals[subject] = fold, original
    return result


def _validate_events(events):
    for subject, start, end in events:
        if not subject or not isinstance(start, int) or not isinstance(end, int) or end <= start:
            raise ValueError('events need nonempty IDs and positive integer intervals')


def match_events(predictions, truths, *, strict=True):
    """Stable IoU-descending greedy matching; equality tested using integer geometry."""
    _validate_events(predictions)
    _validate_events(truths)
    edges = []
    for pi, (ps, p0, p1) in enumerate(predictions):
        for ti, (ts, t0, t1) in enumerate(truths):
            if ps != ts:
                continue
            overlap = max(0, min(p1, t1) - max(p0, t0))
            union = max(p1, t1) - min(p0, t0)
            accepted = 4*overlap > union if strict else 4*overlap >= union
            if accepted:
                edges.append((overlap/union, pi, ti))
    edges.sort(key=lambda value: -value[0])
    used_p, used_t, pairs = set(), set(), []
    for _, pi, ti in edges:
        if pi not in used_p and ti not in used_t:
            used_p.add(pi)
            used_t.add(ti)
            pairs.append((pi, ti))
    return pairs


def event_metrics(predictions, truths, *, strict=True):
    tp = len(match_events(predictions, truths, strict=strict))
    npred, ntrue = len(predictions), len(truths)
    return dict(tp=tp, fp=npred-tp, fn=ntrue-tp, predictions=npred, truths=ntrue,
                recall=tp/ntrue if ntrue else 0.0, ppv=tp/npred if npred else 0.0,
                f1=2*tp/(npred+ntrue) if npred+ntrue else 0.0)


def validate_prediction_rows(rows):
    identities = [(int(r['fold']), int(r['final_index'])) for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError('duplicate final prediction identity')
    if len(rows) != 197 or Counter(r['group'] for r in rows) != {
            'tp_reference': 114, 'fp_reference': 83}:
        raise ValueError('frozen prediction counts do not match the release')
    expected = {0: 35, 1: 46, 2: 33, 3: 42, 4: 41}
    if dict(Counter(int(r['fold']) for r in rows)) != expected:
        raise ValueError('fold prediction counts do not match the release')
    for fold, count in expected.items():
        if {i for f, i in identities if f == fold} != set(range(count)):
            raise ValueError('frozen final indices are not a complete range')


def prepare_truths(rows, subject_fold):
    kept, outside, seen = [], [], set()
    for row in rows:
        event = (normalized_id(row['subject']), int(row['start_ms']), int(row['end_ms']))
        _validate_events([event])
        if event in seen:
            raise ValueError('duplicate truth identity requires explicit resolution')
        seen.add(event)
        (kept if event[0] in subject_fold else outside).append(event)
    return sorted(kept), sorted(outside)


def _read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def protection(root):
    files = [root/'release/event_stack_incumbent.json',
             root/'dist/submission/EatingDetection.exe']
    files.extend(sorted((root/'models/event_stack'/RUN_KEY).rglob('*')))
    return {p.relative_to(root).as_posix(): _sha(p) for p in files if p.is_file()}


def audit_scope(root):
    from src.data.manifests import load_meals, load_sensor_index, MEALS_CSV, INDEX_CSV
    from src.pipeline.artifacts import verify_promotion_attestation
    root = Path(root).resolve()
    # Existing manifest helpers intentionally use their configured repository.
    if Path(MEALS_CSV).resolve().parent.parent != root:
        raise ValueError('audit root differs from the configured data repository')
    protected = protection(root)
    problems = verify_promotion_attestation(root/'models/event_stack'/RUN_KEY,
                                          expected_run_key=RUN_KEY)
    if problems:
        raise ValueError('release attestation invalid: '+str(problems))
    started = time.perf_counter()
    paths = [Path(MEALS_CSV), Path(INDEX_CSV)]
    fold_records = []
    for fold in range(5):
        path = root/'models/event_stack'/RUN_KEY/f'outer-fold-{fold}'/'run_config.json'
        # Subject membership comes from the attested promotion summary, not an alias cache.
        paths.append(path)
    summary_path = root/'models/event_stack'/RUN_KEY/'promotion_summary.json'
    paths.append(summary_path)
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    for record in summary['folds']:
        fold_records.append(dict(fold=record['run_config']['outer_fold'],
                                 subjects=record['outer_subjects']))
    subject_fold = fold_subjects(fold_records)
    tp_path = root/'outputs/challenger_70/e04_verifier_audit/tp_reference.csv'
    fp_path = root/'outputs/challenger_70/e04_verifier_audit/fp_reference.csv'
    fn_path = root/'outputs/challenger_70/error_atlas/fn.csv'
    paths.extend([tp_path, fp_path, fn_path])
    sources_before = {str(p.relative_to(root)): _sha(p) for p in paths}
    tp, fp, fn = _read_csv(tp_path), _read_csv(fp_path), _read_csv(fn_path)
    rows = sorted(tp+fp, key=lambda row: (int(row['fold']), int(row['final_index'])))
    validate_prediction_rows(rows)
    for row in rows+fn:
        if subject_fold.get(normalized_id(row['subject_id'])) != int(row['outer_fold']):
            raise ValueError('ledger subject does not belong to its held-out fold')
    legacy_predictions = [(r['fold']+':'+r['session_id'], int(r['start_ms']), int(r['end_ms'])) for r in rows]
    legacy_truths = []
    eligible_subject_truths = []
    for row in tp+fn:
        key = row.get('matched_truth_key') or row['truth_key']
        session, start, end = key.rsplit(':', 2)
        if session != str(row['outer_fold'])+':'+row['session_id']:
            raise ValueError('truth key and session disagree')
        legacy_truths.append((session, int(start), int(end)))
        eligible_subject_truths.append((normalized_id(row['subject_id']), int(start), int(end)))
    if len(set(legacy_truths)) != 153 or len(legacy_truths) != 153:
        raise ValueError('eligible truth ledger is not exactly 153 unique events')
    if len(set(eligible_subject_truths)) != len(eligible_subject_truths):
        raise ValueError('eligible meals duplicate across sessions; resolve scope before matching')
    legacy = event_metrics(legacy_predictions, legacy_truths, strict=False)
    if (legacy['tp'], legacy['fp'], legacy['fn']) != (114, 83, 39):
        raise ValueError('frozen legacy counts do not replay')
    predictions = [(normalized_id(r['subject_id']), int(r['start_ms']), int(r['end_ms'])) for r in rows]
    meals = load_meals()
    complete, outside = prepare_truths([
        dict(subject=r.externalid, start_ms=r.before_ms, end_ms=r.after_ms)
        for r in meals.itertuples()], subject_fold)
    if not set(eligible_subject_truths).issubset(complete):
        raise ValueError('eligible truth absent from complete valid labels')
    matched = dict((ti, pi) for pi, ti in match_events(predictions, complete))
    eligible_set = set(eligible_subject_truths)
    index = load_sensor_index()
    bounds = {}
    for row in index.to_dict('records'):
        subject = normalized_id(row['externalid'])
        bounds.setdefault(subject, []).append((int(row['timeStamp.startTime']), int(row['timeStamp.endTime'])))
    truth_rows = []
    for ti, (subject, start, end) in enumerate(complete):
        intervals = bounds.get(subject, ())
        truth_rows.append(dict(subject=subject, fold=subject_fold[subject], start_ms=start,
            end_ms=end, duration_s=(end-start)/1000,
            legacy_eligible=(subject, start, end) in eligible_set,
            matched=ti in matched, matched_prediction_index=matched.get(ti),
            index_any_overlap=any(min(end,b)>max(start,a) for a,b in intervals),
            index_single_session_contains=any(a<=start and b>=end for a,b in intervals),
            raw_signal_coverage='NOT_MEASURED_IN_SCOPE_AUDIT'))
    full = event_metrics(predictions, complete)
    per_subject=[]
    for subject in sorted(subject_fold):
        subject_rows=[r for r in truth_rows if r['subject']==subject]
        subject_preds=[p for p in predictions if p[0]==subject]
        subject_truths=[t for t in complete if t[0]==subject]
        per_subject.append(dict(subject=subject, fold=subject_fold[subject],
            **event_metrics(subject_preds,subject_truths),
            eligible_truths=sum(r['legacy_eligible'] for r in subject_rows),
            index_span_days=(max(b for a,b in bounds[subject])-min(a for a,b in bounds[subject]))/86400000 if bounds.get(subject) else 0))
    cases = {
        'legacy_session_eligible_inclusive': legacy,
        'session_eligible_strict': event_metrics(legacy_predictions,legacy_truths),
        'subject_eligible_strict': event_metrics(predictions,eligible_subject_truths),
        'subject_complete_valid_strict': full,
    }
    report = dict(schema_version=1, run_key=RUN_KEY,
        evidence_kind='FROZEN_PREDICTION_SCOPE_DIAGNOSTIC_NOT_NEW_MODEL_PERFORMANCE',
        official_feedback=dict(source='user aggregate table, received 2026-10-19',
            tp=42,fp=26,fn=70,truths=112,predictions=68,recall=.375,ppv=42/68,f1=84/180),
        cases=cases, evaluated_subjects=len(subject_fold),
        valid_labels_all_subjects=len(meals), outside_evaluated_subjects=len(outside),
        outside_subject_ids=sorted({s for s,_,_ in outside}),
        additional_truths=len(complete)-len(eligible_subject_truths),
        additional_truths_matched=sum(r['matched'] and not r['legacy_eligible'] for r in truth_rows),
        complete_short_under120s=sum(r['duration_s']<120 for r in truth_rows),
        metadata_no_overlap_truths=sum(not r['index_any_overlap'] for r in truth_rows),
        metadata_no_containing_session_truths=sum(not r['index_single_session_contains'] for r in truth_rows),
        truth_rows=truth_rows, per_subject=per_subject,
        source_sha256=sources_before, protected_sha256=protected,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        training_fits=0, policy_selections=0, raw_signal_reads=0,
        elapsed_seconds=time.perf_counter()-started)
    if protected != protection(root) or sources_before != {str(p.relative_to(root)):_sha(p) for p in paths}:
        raise ValueError('protected model or source input changed during audit')
    report['protected_unchanged']=True
    return report


def write_audit(output, report):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name in ('truth_rows','per_subject'):
        rows=report[name]
        with (output/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
    lines=['# 官方反馈后的评价范围审计','',
        '仅重评封存预测；没有训练、阈值选择、模型改进或新正式outer成绩。官方测试仍为F1=0.466667；历史153餐结果保持不变。','',
        '| 口径 | TP | FP | FN | GT | F1 |','|---|---:|---:|---:|---:|---:|']
    for label,result in report['cases'].items():
        lines.append(f"| {label} | {result['tp']} | {result['fp']} | {result['fn']} | {result['truths']} | {result['f1']:.6f} |")
    lines.extend(['',f"外层实际留出{report['evaluated_subjects']}人；完整有效标签比eligible增加{report['additional_truths']}餐，其中{report['additional_truths_matched']}餐被已有预测匹配。",
        f"另有{report['outside_evaluated_subjects']}餐属于未被这套外层预测覆盖的主体，单列而不冒充已评估。",
        '', '完整有效标签口径不因没有窗口、候选或采集覆盖而删除餐次；其分母仍不是官方隐藏测试分母。',
        '本审计的会话范围只来自元数据索引，不能把有记录时段等同于有效IMU覆盖。',
        '新增匹配可能改变原FP的身份；必须分别看subject匹配与完整标签两项变化，不将计数变化称为算法提升。',
        '没有官方逐餐真值，不能由本地口径差异证明官方漏检的唯一原因。',
        '', '所有数据源和正式模型/EXE哈希保留于summary.json；开始与结束校验一致。'])
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    root=Path(__file__).resolve().parents[3]
    if args.output.exists():
        raise FileExistsError(args.output)
    report=audit_scope(root)
    write_audit(args.output,report)
    print(json.dumps({k:report[k] for k in ('cases','additional_truths','additional_truths_matched','protected_unchanged')},ensure_ascii=False))


if __name__=='__main__':
    main()
