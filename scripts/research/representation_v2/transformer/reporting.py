"""Paired event evidence, transparent subject tails and no outer headline."""
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score,log_loss

from scripts.research.representation_v2.validation_metrics import matched_indices
from src.pipeline.event_stack import compute_event_metrics
from .contract import ARMS


def metrics(predictions,truths):
    value=asdict(compute_event_metrics(predictions,truths))
    value.update(fp=value['n_pred']-value['n_tp'],fn=value['n_true']-value['n_tp'])
    return value


def make_record(wrapper,predictions,truths,subject_by_sid,candidates,labels,scores):
    if set(predictions)!=set(ARMS) or set(scores)!=set(ARMS): raise ValueError('four arms required')
    if any(p.sid not in subject_by_sid for ps in predictions.values() for p in ps):
        raise ValueError('prediction ownership missing')
    if any(t.sid not in subject_by_sid for t in truths): raise ValueError('truth ownership missing')
    subjects=[]; short={}; diagnostics={}
    for subject in sorted(set(subject_by_sid.values())):
        sids={sid for sid,owner in subject_by_sid.items() if owner==subject}
        ts=[t for t in truths if t.sid in sids]
        subjects.append({'subject':subject,'events':{
            a:metrics([p for p in predictions[a] if p.sid in sids],ts) for a in ARMS}})
    for a in ARMS:
        _,indices=matched_indices(predictions[a],truths)
        short[a]={'n_true':sum(t.end_ms-t.start_ms<600000 for t in truths),
                  'n_tp':sum(truths[j].end_ms-truths[j].start_ms<600000 for j in indices)}
        p=np.asarray(scores[a]); y=np.asarray(labels)
        if p.shape!=y.shape or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
            raise ValueError('candidate score alignment invalid')
        binary=set(y.tolist())=={0,1}
        diagnostics[a]={'ap':float(average_precision_score(y,p)) if binary else None,
                        'bce':float(log_loss(y,p,labels=[0,1])) if binary else None}
    reachable=compute_event_metrics([c.event for c in candidates],truths)
    return {'wrapper':wrapper,'events':{a:metrics(predictions[a],truths) for a in ARMS},
            'subject_events':subjects,'short_events':short,'candidate_metrics':diagnostics,
            'candidate_recall':{'n_true':reachable.n_true,'n_reachable':reachable.n_tp},
            'predictions':{a:[asdict(p) for p in predictions[a]] for a in ARMS},
            'truths':[asdict(t) for t in truths],'outer_calls':0}


def pool(rows):
    tp=sum(r['n_tp'] for r in rows); pred=sum(r['n_pred'] for r in rows); true=sum(r['n_true'] for r in rows)
    return {'n_tp':tp,'n_pred':pred,'n_true':true,'fp':pred-tp,'fn':true-tp,
            'f1':2*tp/(pred+true) if pred+true else 0.,'ppv':tp/pred if pred else 0.,
            'recall':tp/true if true else 0.}


def summarize(records):
    if len(records)!=3 or {r['wrapper'] for r in records}!=set(range(3)):
        raise ValueError('three complete wrappers required')
    subjects=[s for r in records for s in r['subject_events']]
    ids=[s['subject'] for s in subjects]
    if len(ids)!=24 or len(set(ids))!=24: raise ValueError('24 unique health subjects required')
    for r in records:
        if r.get('outer_calls')!=0 or set(r['events'])!=set(ARMS): raise ValueError('invalid development evidence')
        if len({r['events'][a]['n_true'] for a in ARMS})!=1: raise ValueError('GT denominator differs')
        for a in ARMS:
            totals=pool([s['events'][a] for s in r['subject_events']])
            if any(totals[k]!=r['events'][a][k] for k in ('n_tp','n_pred','n_true')):
                raise ValueError('subject counts differ from wrapper')
    events={a:pool([r['events'][a] for r in records]) for a in ARMS}
    gate={}
    for a in ('p','c','t'):
        delta=events[a]['f1']-events['b']['f1']
        nonworse=sum(r['events'][a]['f1']>=r['events']['b']['f1']-1e-12 for r in records)
        gate[a]={'passed':bool(delta>=.01-1e-12 and nonworse>=2),
                 'delta_f1_vs_b':delta,'nonworse_wrappers':nonworse}
    diagnostics={}
    for a in ARMS:
        values=[s['events'][a]['f1'] for s in subjects if s['events'][a]['n_true']>0]
        delta=[s['events'][a]['f1']-s['events']['b']['f1'] for s in subjects if s['events'][a]['n_true']>0]
        diagnostics[a]={'with_gt_subjects':len(values),'median':float(np.median(values)) if values else None,
                        'iqr':np.percentile(values,[25,75]).tolist() if values else None,
                        'p10':float(np.percentile(values,10)) if values else None,
                        'improved':sum(d>0 for d in delta),'worsened':sum(d<0 for d in delta),
                        'tied':sum(d==0 for d in delta),
                        'no_gt_subjects':sum(s['events'][a]['n_true']==0 for s in subjects),
                        'no_gt_fp':sum(s['events'][a]['fp'] for s in subjects if s['events'][a]['n_true']==0)}
    short={a:{k:sum(r['short_events'][a][k] for r in records) for k in ('n_true','n_tp')} for a in ARMS}
    for a in ARMS: short[a]['recall']=short[a]['n_tp']/short[a]['n_true'] if short[a]['n_true'] else 0.
    candidate={k:sum(r['candidate_recall'][k] for r in records) for k in ('n_true','n_reachable')}
    candidate['recall']=candidate['n_reachable']/candidate['n_true'] if candidate['n_true'] else 0.
    return {'events':events,'gate':gate,'subject_diagnostics':diagnostics,'short_events':short,
            'candidate_recall':candidate,'unique_subjects':24,'wrapper_count':3,
            'transformer_delta_vs_pool':events['t']['f1']-events['p']['f1'],
            'transformer_delta_vs_tcn':events['t']['f1']-events['c']['f1'],
            'development_only':True,'outer_calls':0,'dist_changed':False}


def write_report(output,summary):
    lines=['# A6 T1 — 四臂严格 inner 开发实验','',
           '本报告不是正式 outer 成绩。T24已反复用于开发；不更新dist，不代表达到正式0.70。','',
           '| 臂 | Event F1 | TP | FP | FN | PPV | Recall |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    for a in ARMS:
        r=summary['events'][a]
        lines.append(f"| {a} | {r['f1']:.6f} | {r['n_tp']} | {r['fp']} | {r['fn']} | {r['ppv']:.6f} | {r['recall']:.6f} |")
    lines+=['','## 归因与下一步资格','']
    for a in ('p','c','t'):
        r=summary['gate'][a]
        lines.append(f"- {a}: ΔF1={r['delta_f1_vs_b']:+.6f}，{r['nonworse_wrappers']}/3 wrapper非下降；扩大验证gate={'通过' if r['passed'] else '未通过'}。")
    lines += ['',f"Transformer vs pooling: {summary['transformer_delta_vs_pool']:+.6f}; vs TCN: {summary['transformer_delta_vs_tcn']:+.6f}。",'',
              '这只能区分当前固定配置的输入/读出收益；任何gate通过也不授权outer或发布。',
              '', '## 证据', '',
              'summary.json包含短餐召回、受试者median/IQR/P10、改善/恶化人数和无GT主体FP；逐wrapper health文件保留配对计数与事件。',
              'resources.json保留实际拟合次数、步数、训练/编码/CPU耗时和显存；manifest与registered_plan锁定身份、配置、源码和保护快照。']
    if 'resources' in summary:
        r=summary['resources']
        lines += ['',f"计算：producer {r['producer_fits']} fits，base {r['base_fits']} fits，neural {r['neural_fits']} fits / {r['steps']} steps；neural {r['neural_seconds']:.1f}s，encoding {r['encoding_seconds']:.1f}s，CPU {r['cpu_seconds']:.1f}s。"]
    Path(output,'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
