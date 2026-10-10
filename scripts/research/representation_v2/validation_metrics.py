"""A5 development-only event counts and descriptive error complementarity."""
from dataclasses import asdict

import numpy as np
from sklearn.metrics import average_precision_score, log_loss

from src.eval.metrics import event_iou
from src.pipeline.event_stack import EventRef, compute_event_metrics

ARMS = ('b', 'r', 'f')


def classify_gain(deltas):
    if len(deltas) != 2 or not np.isfinite(deltas).all():
        raise ValueError('two finite repeat deltas required')
    mean = float(np.mean(deltas))
    if mean <= 0:
        return 'NO_SUPPORTED_GAIN'
    if min(deltas) <= 0:
        return 'UNSTABLE_GAIN'
    return 'SUPPORTED_NEXT_DESIGN' if mean >= .005-1e-12 else 'SMALL_CONSISTENT_GAIN'


def matched_indices(predictions, truths):
    pairs = [(event_iou(p.interval, t.interval), i, j)
             for i, p in enumerate(predictions) for j, t in enumerate(truths)
             if p.sid == t.sid and event_iou(p.interval, t.interval) >= .25]
    pairs.sort(key=lambda row: -row[0])
    pi, ti = set(), set()
    for _, i, j in pairs:
        if i not in pi and j not in ti:
            pi.add(i)
            ti.add(j)
    if len(pi) != compute_event_metrics(predictions, truths).n_tp:
        raise RuntimeError('diagnostic matcher differs from canonical')
    return pi, ti


def metrics(predictions, truths):
    row = asdict(compute_event_metrics(predictions, truths))
    row.update(fp=row['n_pred']-row['n_tp'], fn=row['n_true']-row['n_tp'])
    return row


def make_record(wrapper, predictions, truths, subject_by_sid, labels, scores):
    if set(predictions) != set(ARMS) or set(scores) != set(ARMS):
        raise ValueError('three aligned arms required')
    if set(t.sid for t in truths)-set(subject_by_sid):
        raise ValueError('truth subject mapping incomplete')
    if any(set(p.sid for p in ps)-set(subject_by_sid) for ps in predictions.values()):
        raise ValueError('prediction subject mapping incomplete')
    subjects = []
    for subject in sorted(set(subject_by_sid.values())):
        sids = {sid for sid, owner in subject_by_sid.items() if owner == subject}
        ts = [t for t in truths if t.sid in sids]
        subjects.append({'subject': subject, 'events': {
            a: metrics([p for p in predictions[a] if p.sid in sids], ts) for a in ARMS}})
    short = {}
    candidate = {}
    for a in ARMS:
        _, ti = matched_indices(predictions[a], truths)
        short[a] = {'n_true': sum(t.end_ms-t.start_ms < 600000 for t in truths),
                    'n_tp': sum(truths[j].end_ms-truths[j].start_ms < 600000 for j in ti)}
        p = np.asarray(scores[a], dtype=float)
        y = np.asarray(labels)
        if p.shape != y.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError('invalid candidate score alignment')
        binary = len(y) and set(y.tolist()) == {0, 1}
        candidate[a] = {'ap': float(average_precision_score(y, p)) if binary else None,
                        'bce': float(log_loss(y, p, labels=[0, 1])) if binary else None,
                        'reason': None if binary else 'empty_or_single_class', 'scope': 'candidate_pooled'}
    return {'wrapper': wrapper['id'], 'repeat': wrapper['repeat'], 'fold': wrapper['fold'],
            'events': {a: metrics(predictions[a], truths) for a in ARMS},
            'subject_events': subjects, 'short_events': short, 'candidate_metrics': candidate,
            'predictions': {a: [asdict(p) for p in predictions[a]] for a in ARMS},
            'truths': [asdict(t) for t in truths], 'outer_calls': 0}


def pool(rows):
    tp = sum(r['n_tp'] for r in rows)
    pred = sum(r['n_pred'] for r in rows)
    true = sum(r['n_true'] for r in rows)
    return {'n_tp': tp, 'n_pred': pred, 'n_true': true, 'fp': pred-tp, 'fn': true-tp,
            'f1': 2*tp/(pred+true) if pred+true else 0.,
            'ppv': tp/pred if pred else 0., 'recall': tp/true if true else 0.}


def summarize_repeats(records):
    if len(records) != 6 or {r['wrapper'] for r in records} != set(range(6)):
        raise ValueError('six complete unique wrappers required')
    if any(r['repeat'] != r['wrapper']//3 or r['fold'] != r['wrapper']%3 for r in records):
        raise ValueError('wrapper identity differs')
    repeats, populations, subject_deltas = [], [], {}
    for repeat in range(2):
        rows = [r for r in records if r['repeat'] == repeat]
        subjects = [s for row in rows for s in row['subject_events']]
        owners = [s['subject'] for s in subjects]
        if len(owners) != 24 or len(set(owners)) != 24:
            raise ValueError('repeat must cover 24 unique subjects')
        populations.append(set(owners))
        events = {a: pool([r['events'][a] for r in rows]) for a in ARMS}
        diagnostics = {}
        for a in ARMS:
            values = [s['events'][a]['f1'] for s in subjects]
            delta = [s['events'][a]['f1']-s['events']['b']['f1'] for s in subjects]
            diagnostics[a] = {'median': float(np.median(values)), 'iqr': np.percentile(values, [25, 75]).tolist(),
                              'p10': float(np.percentile(values, 10)),
                              'improved': sum(d > 0 for d in delta), 'worsened': sum(d < 0 for d in delta),
                              'tied': sum(d == 0 for d in delta)}
            for s in subjects:
                subject_deltas.setdefault(s['subject'], {}).setdefault(a, []).append(
                    s['events'][a]['f1']-s['events']['b']['f1'])
        repeats.append({'repeat': repeat, 'events': events, 'subject_diagnostics': diagnostics,
                        'short_events': {a: {k: sum(r['short_events'][a][k] for r in rows)
                                              for k in ('n_true', 'n_tp')} for a in ARMS}})
    if populations[0] != populations[1] or repeats[0]['events']['b']['n_true'] != repeats[1]['events']['b']['n_true']:
        raise ValueError('repeat population or truth denominator differs')
    means = {a: float(np.mean([r['events'][a]['f1'] for r in repeats])) for a in ARMS}
    gains = {a: [r['events'][a]['f1']-r['events']['b']['f1'] for r in repeats] for a in ('r', 'f')}
    supported = [a for a in ('r', 'f') if classify_gain(gains[a]) == 'SUPPORTED_NEXT_DESIGN']
    preferred = None
    if supported:
        preferred = max(supported, key=lambda a: means[a])
        if len(supported) == 2 and abs(means['r']-means['f']) <= .001+1e-12:
            preferred = 'r'
    return {'repeats': repeats, 'mean_f1': means, 'repeat_deltas_vs_b': gains,
            'mean_f_minus_r': means['f']-means['r'], 'unique_subjects': 24,
            'classification': {a: classify_gain(gains[a]) for a in gains},
            'preferred_for_next_design': preferred,
            'subject_mean_deltas': {s: {a: float(np.mean(v)) for a, v in arms.items()}
                                    for s, arms in subject_deltas.items()}}


def complementarity(records):
    result = []
    for repeat in range(2):
        outcomes = []
        fps = {a: [] for a in ARMS}
        for row in records:
            if row['repeat'] != repeat:
                continue
            truths = [EventRef(**t) for t in row['truths']]
            hits = {}
            for a in ARMS:
                pred = [EventRef(**p) for p in row['predictions'][a]]
                pi, hits[a] = matched_indices(pred, truths)
                fps[a].extend(asdict(p) for i, p in enumerate(pred) if i not in pi)
            for j, truth in enumerate(truths):
                outcomes.append({'wrapper': row['wrapper'], 'truth_index': j, 'truth': asdict(truth),
                                 'detected': {a: j in hits[a] for a in ARMS}})
        flags = [r['detected'] for r in outcomes]
        result.append({'repeat': repeat, 'gt_outcomes': outcomes, 'fp_events': fps,
                       'counts': {'r_only_vs_f': sum(v['r'] and not v['f'] for v in flags),
                                  'f_only_vs_r': sum(v['f'] and not v['r'] for v in flags),
                                  'common_fn': sum(not any(v.values()) for v in flags),
                                  **{a+'_gained_vs_b': sum(v[a] and not v['b'] for v in flags) for a in ('r', 'f')},
                                  **{a+'_lost_vs_b': sum(v['b'] and not v[a] for v in flags) for a in ('r', 'f')}}})
    return {'repeats': result, 'diagnostic_only': True, 'no_merged_model_metric': True}
