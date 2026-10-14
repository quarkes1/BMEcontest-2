"""Fixed nested B/Q/P controls. Policy commits precede health evaluation."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import multiprocessing
import os
from pathlib import Path
import shutil
import time

import numpy as np

from ..ppg.audit import (write_json, require_committed, assert_state, verify_hashes,
                         stage, owned_rss, output_size)
from ..ppg.contract import T1, read_json, file_hash
from .cache import OUTPUT, registration, candidate_features, read_windows


def sufficient_support(labels, valid):
    y = np.asarray(labels)[valid]
    return bool((y == 1).sum() >= 10 and (y == 0).sum() >= 30)


def fallback_scores(base, prediction, valid):
    base, prediction, valid = np.asarray(base), np.asarray(prediction), np.asarray(valid, bool)
    if base.shape != prediction.shape or base.shape != valid.shape or not np.isfinite(prediction).all():
        raise ValueError('invalid conditional score alignment')
    return np.where(valid, prediction, base)


def reproduce(reference, candidates, groups, labels, base):
    canonical = lambda v: json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if (canonical(candidates) != canonical(reference['candidates']) or list(groups) != reference['groups'] or
            list(labels) != reference['labels'] or np.asarray(base).shape != np.asarray(reference['scores']['b']).shape
            or not np.allclose(base, reference['scores']['b'], atol=1e-6, rtol=1e-5)):
        raise RuntimeError('incumbent candidate/base reproduction failed')


def gate(b, q, p, directions):
    passed = (p['f1']-b['f1'] >= .015-1e-12 and p['f1']-q['f1'] >= .005-1e-12
              and sum(directions) >= 2 and p['n_tp'] >= b['n_tp']-1
              and (p['fp'] < b['fp'] or p['n_tp']-b['n_tp'] > p['fp']-b['fp']))
    return dict(passed=bool(passed), delta_vs_base=p['f1']-b['f1'], delta_vs_quality=p['f1']-q['f1'],
                nonworse_wrappers=sum(directions), outer_calls=0, next_validation_authorized=bool(passed))


def deserialize(rows):
    from src.pipeline.event_stack import EventRef, CandidateEvent, MultiScaleCandidate
    def stream(raw):
        if raw is None:
            return None
        raw = dict(raw); raw['event'] = EventRef(**raw['event']); raw['probabilities'] = tuple(raw['probabilities'])
        return CandidateEvent(**raw)
    return [MultiScaleCandidate(event=EventRef(**r['event']), macro=stream(r['macro']), micro=stream(r['micro'])) for r in rows]


def load_runtime(root, manifest):
    from .. import event_probe as ep
    from ..transformer.v7_models import SubsetSource
    config = ep.registered_e03_config(0, workers=4)
    if asdict(config) != manifest['config']:
        raise RuntimeError('candidate producer configuration differs')
    allowed = {sid for sid, owner in manifest['subject_by_session'].items() if owner in manifest['identities']['t24']}
    data = SubsetSource(root, allowed).load_outer_train_only(config)
    if data.subject_by_session != manifest['subject_by_session']:
        raise RuntimeError('filtered T24 ownership differs')
    return data, config


def split_scores(root, data, config, train, score, excluded, windows, owners, prefix):
    from .. import event_probe as ep
    train, score = set(train), set(score)
    if train & (score | set(excluded)):
        raise ValueError('fusion train/score subject overlap')
    started = time.perf_counter(); graph = {}
    t, v = ep.build_candidate_ledgers(data, config, sorted(train), sorted(score), fit_audit=graph)
    if set(t.groups)-train or set(v.groups)-score:
        raise ValueError('candidate ledger subject isolation failed')
    base = ep.verifier_probabilities(t.features, t.labels, v.features, seed=config.seed)
    reference = read_json(Path(root)/T1/f'{prefix}_scores.json')
    reproduce(reference, [asdict(c) for c in v.candidates], v.groups, v.labels, base)
    # Preserve original baseline values exactly, including every fallback row.
    base = np.asarray(reference['scores']['b'], float)
    tq, tp = candidate_features(t.candidates, windows, owners, train)
    vq, vp = candidate_features(v.candidates, windows, owners, score)
    eligible_t, eligible_v = np.isfinite(tp).any(axis=1), np.isfinite(vp).any(axis=1)
    support = sufficient_support(t.labels, eligible_t)
    scores = {'b': base}; fitted = 0
    for arm, x, z in [('q', np.column_stack([t.features, tq]), np.column_stack([v.features, vq])),
                      ('p', np.column_stack([t.features, tq, tp]), np.column_stack([v.features, vq, vp]))]:
        prediction = base.copy()
        if support:
            parts = []
            for model in (ep.runner._verifier_estimator(config.seed+20, .1), ep.runner._verifier_lgbm_estimator(config.seed+21)):
                model.fit(x[eligible_t], t.labels[eligible_t])
                parts.append(ep.runner._positive_probability(model, z))
                fitted += 1
            prediction = np.mean(parts, axis=0)
        scores[arm] = fallback_scores(base, prediction, eligible_v)
        graph[arm] = dict(fit_subjects=sorted(train) if support else [], parents=['candidates'])
    graph['base'] = dict(fit_subjects=sorted(train), parents=['candidates'])
    graph['score'] = dict(fit_subjects=[], parents=['base', 'q', 'p'])
    ep.check_graph(graph, 'score', score | set(excluded), train)
    evidence = dict(train=sorted(train), score=sorted(score), excluded=sorted(excluded), graph=graph,
        candidates=[asdict(c) for c in v.candidates], groups=v.groups.tolist(), labels=v.labels.tolist(),
        scores={a: s.tolist() for a, s in scores.items()}, eligibility=eligible_v.tolist(),
        train_support=dict(eligible_positive=int((eligible_t & (t.labels == 1)).sum()),
                           eligible_negative=int((eligible_t & (t.labels == 0)).sum()), passed=support),
        fits=dict(producer=10, baseline=2, branches=fitted), elapsed_seconds=time.perf_counter()-started, outer_calls=0)
    write_json(Path(root)/OUTPUT/f'{prefix}_scores.json', evidence)
    return v, scores, evidence


def worker(root, wrapper, phase):
    root = Path(root); out = root/OUTPUT
    if phase == 'health':
        require_committed(root, out/'policy_seal.json')
        verify_hashes(out, read_json(out/'policy_seal.json'))
    from threadpoolctl import threadpool_limits
    from .. import event_probe as ep
    reg, inv = registration(root); manifest = reg['manifest']
    verify_hashes(out, read_json(out/'cache_seal.json'))
    windows = read_windows(out/'windows.npz')
    resources = read_json(out/'cache_resources.json')
    if resources['completed_files'] != 696:
        raise RuntimeError('incomplete full-session cache')
    for sid in inv['subject_by_sid']:
        windows.setdefault(sid, [])  # legal zero-window session, verified complete extraction
    data, config = load_runtime(root, manifest)
    owners = manifest['subject_by_session']
    excluded = set(manifest['identities']['v7']) | set(manifest['identities']['outer'])
    k = wrapper['id']
    with threadpool_limits(limits=4):
        if phase == 'policy':
            candidates, groups = [], []
            scores = {a: [] for a in ('b', 'q', 'p')}
            inner_data = ep.slice_outer_train(data, wrapper['train'])
            for j, part in enumerate(wrapper['policy_splits']):
                ledger, prediction, _ = split_scores(root, inner_data, config, part['train'], part['score'],
                    excluded | set(wrapper['health']), windows, owners, f'wrapper{k}_policy{j}')
                candidates.extend(ledger.candidates); groups.extend(ledger.groups)
                for a in scores:
                    scores[a].extend(prediction[a])
            policies = {'b': read_json(root/T1/f'wrapper{k}_policies.json')['policies']['b']}
            for a in ('q', 'p'):
                policies[a] = ep.choose_policy(candidates, np.asarray(scores[a]), groups, inner_data.train_truths, config)
            write_json(out/f'wrapper{k}_policies.json', dict(wrapper=k, policies=policies, outer_calls=0))
            return dict(wrapper=k, completed=3)
        require_committed(root, out/'policy_seal.json')
        verify_hashes(out, read_json(out/'policy_seal.json'))
        ledger, scores, evidence = split_scores(root, data, config, wrapper['train'], wrapper['health'], excluded,
                                               windows, owners, f'wrapper{k}_health')
        policies = read_json(out/f'wrapper{k}_policies.json')['policies']
        truths = ep.slice_outer_train(data, wrapper['health']).train_truths
        record = dict(wrapper=k, events={}, predictions={}, subject_events={}, short_recall={}, outer_calls=0)
        from src.pipeline.validation_metrics import matched_indices
        for a in scores:
            predictions = ep.decode_policy(ledger.candidates, scores[a], ledger.groups, policies[a])
            record['predictions'][a] = [asdict(p) for p in predictions]
            record['events'][a] = asdict(ep.compute_event_metrics(predictions, truths))
            record['subject_events'][a] = {s: asdict(ep.compute_event_metrics(
                [p for p in predictions if owners[p.sid] == s], [t for t in truths if owners[t.sid] == s]))
                for s in wrapper['health']}
            short = [t for t in truths if t.end_ms-t.start_ms <= 600000]
            _, matched = matched_indices(predictions, short)
            record['short_recall'][a] = dict(gt=len(short), detected=len(matched))
        reference = read_json(root/T1/f'wrapper{k}_health.json')
        if record['predictions']['b'] != reference['predictions']['b']:
            raise RuntimeError('frozen baseline decoded events differ')
        record['eligibility'] = evidence['eligibility']; record['train_support'] = evidence['train_support']
        write_json(out/f'wrapper{k}_health.json', record)
        return record


def execute(root, phase):
    from .. import event_probe as ep
    root = Path(root); out = root/OUTPUT
    reg, _ = registration(root)
    for n in ('state.json', 'registry.json', 'inventory.json', 'windows.npz', 'cache_seal.json'):
        require_committed(root, out/n)
    if phase == 'health':
        require_committed(root, out/'policy_seal.json')
        verify_hashes(out, read_json(out/'policy_seal.json'))
    expected, success = ('CACHE_SEALED', 'POLICIES_FROZEN') if phase == 'policy' else ('POLICIES_FROZEN', 'HEALTH_COMPLETE')
    start = time.perf_counter(); res = dict(phase=phase, completed_wrappers=0, peak_rss_bytes=0, outer_calls=0)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '4'
    with stage(out, expected, success):
        pool = ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn'))
        records = []; last_check = 0
        try:
            pending = {pool.submit(worker, str(root), w, phase) for w in reg['manifest']['wrappers']}
            while pending:
                now = time.perf_counter()
                if now-last_check >= 1:
                    rss = owned_rss([os.getpid()]+[p.pid for p in multiprocessing.active_children()])
                    res['peak_rss_bytes'] = max(res['peak_rss_bytes'], rss); res['elapsed_seconds'] = now-start
                    if now-start > 6*3600 or rss > 32*2**30 or output_size(out) > 128*2**20 or shutil.disk_usage(root).free < 20*2**30:
                        raise RuntimeError('PPG fusion resource budget exceeded')
                    write_json(out/f'{phase}_resources.json', res); last_check = now
                done, pending = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                for future in done:
                    records.append(future.result()); res['completed_wrappers'] += 1
                    print(f'P1 {phase} wrapper complete {res["completed_wrappers"]}/3', flush=True)
        except BaseException:
            for child in multiprocessing.active_children():
                child.terminate()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        if phase == 'policy':
            write_json(out/'policy_seal.json', {f'wrapper{k}_policies.json': file_hash(out/f'wrapper{k}_policies.json') for k in range(3)})
        else:
            records.sort(key=lambda r: r['wrapper'])
            metrics = {a: ep.counts_metrics([r['events'][a] for r in records]) for a in ('b', 'q', 'p')}
            b = metrics['b']
            if (b['n_tp'], b['fp'], b['fn']) != (66, 20, 31):
                raise RuntimeError('original T1 pooled baseline differs')
            directions = [r['events']['p']['f1'] >= r['events']['b']['f1'] for r in records]
            result = dict(metrics=metrics, gate=gate(metrics['b'], metrics['q'], metrics['p'], directions), records=records)
            write_json(out/'result.json', result)
            text = '# A7 P1 固定PPG融合结果\n\nT24开发证据；outer_calls=0。\n\n|arm|F1|TP|FP|FN|\n|---|---:|---:|---:|---:|\n'
            for a, m in metrics.items():
                text += f'|{a}|{m["f1"]:.6f}|{m["n_tp"]}|{m["fp"]}|{m["fn"]}|\n'
            text += '\nGate: '+str(result['gate'])+'\n\n正式模型F1仍.6514285714。T24结果不能直接更新dist。\n'
            (out/'report.md').write_text(text, encoding='utf-8')
        res['elapsed_seconds'] = time.perf_counter()-start
        write_json(out/f'{phase}_resources.json', res)


def main():
    from .cache import register, build_cache
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['register', 'cache', 'policy', 'health'])
    args = parser.parse_args(); root = Path.cwd()
    if args.phase == 'register':
        register(root)
    elif args.phase == 'cache':
        build_cache(root)
    else:
        execute(root, args.phase)


if __name__ == '__main__':
    main()
