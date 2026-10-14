"""Deterministic same-session candidate pooling; no labels or learned transforms."""
import numpy as np


def phases(event):
    s, e = event['start_ms'], event['end_ms']
    if e <= s:
        raise ValueError('nonpositive candidate duration')
    return [(s-1200000, s-300000), (s, e), (s+600000, s+2700000)]


def union_length(intervals):
    total, right = 0, None
    for start, end in sorted(intervals):
        if end <= start:
            continue
        total += end-max(start, right if right is not None else start) if right is None or end > right else 0
        right = max(end, right) if right is not None else end
    return total


def weighted_median(values, weights):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not ok.any():
        return float('nan')
    values, weights = values[ok], weights[ok]
    order = np.argsort(values, kind='stable')
    return float(values[order][np.searchsorted(np.cumsum(weights[order]), weights.sum()/2)])


def aggregate_candidate(event, windows, shift_ms=0):
    if shift_ms not in (0, -560):
        raise ValueError('unregistered anchor shift')
    q, summaries = [], []
    for start, end in phases(event):
        selected = []
        for w in windows:
            if w['sid'] != event['sid']:
                continue
            lo, hi = max(start, w['start_ms']+shift_ms), min(end, w['end_ms']+shift_ms)
            if hi > lo:
                selected.append((w, (lo, hi), hi-lo))
        valid = [v for v in selected if v[0]['valid']]
        paired = [v for v in valid if v[0]['pair_pass']]
        coverage = union_length([v[1] for v in valid])/(end-start)
        pair_coverage = union_length([v[1] for v in paired])/(end-start)
        motion = weighted_median([v[0]['motion'] for v in selected], [v[2] for v in selected])
        q.extend([coverage, pair_coverage, motion])
        proxies, reasons, proxy_coverage, proxy_counts = [], [], [], []
        for name in ('frequency', 'rms3_dc', 'rms5_dc'):
            finite = [v for v in paired if np.isfinite(v[0][name])]
            cov = union_length([v[1] for v in finite])/(end-start)
            reason = 'ok' if cov >= .5 and len(finite) >= 6 else 'coverage_or_window_count'
            proxies.append(weighted_median([v[0][name] for v in finite], [v[2] for v in finite])
                           if reason == 'ok' else float('nan'))
            reasons.append(reason); proxy_coverage.append(cov); proxy_counts.append(len(finite))
        summaries.append(dict(start_ms=start, end_ms=end, requested_ms=end-start,
                              coverage=coverage, pair_coverage=pair_coverage,
                              windows=len(selected), pair_windows=len(paired), proxies=proxies,
                              proxy_coverage=proxy_coverage, proxy_counts=proxy_counts, reasons=reasons))
    stage = np.array([p['proxies'] for p in summaries])
    return dict(q=q, p=np.concatenate([stage[1]-stage[0], stage[2]-stage[0]]).tolist(), phases=summaries)
