"""Nested upstream-OOF offset model with immutable B policies and priors."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import multiprocessing
import os
import shutil
import subprocess
import time

import numpy as np

from ..ppg.audit import (write_json, require_committed, verify_hashes, protected_snapshot,
                         create_output, stage, owned_rss, output_size)
from ..ppg.contract import T1, file_hash, read_json
from ..ppg_fusion.cache import OUTPUT as SOURCE, registration as source_registration, read_windows, candidate_features
from ..ppg_fusion.probe import load_runtime, deserialize, short_recall, gate
from .model import fit, predict

OUTPUT = Path('outputs/challenger_70/representation_v2/amendment_a7_ppg_residual/attempt_02')
SPEC = 'docs/challenge_070_amendment_a7_p2_prior_residual.md'


def seal_policy(out):
    names = [f'wrapper{k}_policies.json' for k in range(3)] + ['policy_resources.json']
    write_json(out/'policy_seal.json', {name: file_hash(out/name) for name in names})


def gt_migration(predictions, truths):
    from ..validation_metrics import matched_indices
    hits = {a: matched_indices(ps, truths)[1] for a, ps in predictions.items()}
    return {a: dict(matched_indices=sorted(h), gained_indices=sorted(h-hits['b']),
                    lost_indices=sorted(hits['b']-h)) for a, h in hits.items()}


def register(root):
    root = Path(root)
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root).strip():
        raise RuntimeError('commit before residual registration')
    old, _ = source_registration(root)
    verify_hashes(root/SOURCE, read_json(root/SOURCE/'cache_seal.json'))
    files = dict(old['files'])
    for name in ('registry.json', 'inventory.json', 'windows.npz', 'cache_seal.json', 'cache_resources.json', 'result.json', 'score_diagnostic.json'):
        files[(SOURCE/name).as_posix()] = file_hash(root/SOURCE/name)
    for p in (root/'scripts/research/representation_v2/ppg_residual').glob('*.py'):
        files[p.relative_to(root).as_posix()] = file_hash(p)
    files[SPEC] = file_hash(root/SPEC)
    out = root/OUTPUT; create_output(out)
    write_json(out/'registry.json', dict(files=files, manifest=old['manifest'], source=SOURCE.as_posix(),
                                       protected_index=protected_snapshot(root), ridge=16, inner_splits=3, outer_calls=0))
    write_json(out/'state.json', dict(status='REGISTERED', outer_calls=0))


def registration(root):
    root = Path(root); out = root/OUTPUT
    r = read_json(out/'registry.json'); verify_hashes(root, r['files'])
    if r['ridge'] != 16 or r['inner_splits'] != 3 or protected_snapshot(root) != r['protected_index']:
        raise RuntimeError('registered residual contract differs')
    return r


def check_partition(train, score, excluded):
    if not set(train) or not set(score) or set(train) & (set(score) | set(excluded)):
        raise ValueError('residual partition overlap or empty subjects')


def base_graph(reference):
    original = reference['graph']; result = {}
    def visit(name):
        if name in result:
            return
        result[name] = original[name]
        for parent in original[name]['parents']:
            visit(parent)
    visit('base')
    result['prior_score'] = dict(fit_subjects=[], parents=['base'])
    return result


def fit_split(root, data, config, train, score, excluded, windows, owners, prefix):
    from .. import event_probe as ep
    from dataclasses import asdict
    check_partition(train, score, excluded)
    train, score = set(train), set(score)
    inner_data = ep.slice_outer_train(data, train)
    blocks, graphs = [], []
    started = time.perf_counter()
    for j, (child_train, child_score) in enumerate(ep.inner_subject_partitions(inner_data, n_splits=3)):
        graph = {}
        t, v = ep.build_candidate_ledgers(inner_data, config, child_train, child_score, fit_audit=graph)
        base = ep.verifier_probabilities(t.features, t.labels, v.features, seed=config.seed)
        graph['base'] = dict(fit_subjects=list(child_train), parents=['candidates'])
        graph['score'] = dict(fit_subjects=[], parents=['base'])
        ep.check_graph(graph, 'score', set(child_score) | score | set(excluded), set(child_train))
        q, p = candidate_features(v.candidates, windows, owners, set(child_score))
        blocks.append((q, p, v.labels, base, v.groups)); graphs.append(graph)
        print(f'P2 {prefix} upstream OOF {j+1}/3', flush=True)
    q, p, y, prior, groups = [np.concatenate([b[i] for b in blocks]) for i in range(5)]
    eligible = np.isfinite(p).any(axis=1)
    models = {'q': fit(q[eligible], y[eligible], prior[eligible], groups[eligible]),
              'p': fit(np.column_stack([q, p])[eligible], y[eligible], prior[eligible], groups[eligible])}
    if models['q']['active'] != models['p']['active']:
        raise RuntimeError('paired support gate differs')
    if any(set(m['fit_subjects'])-train for m in models.values()):
        raise RuntimeError('residual model learned held-out subjects')
    reference = read_json(Path(root)/T1/f'{prefix}_scores.json')
    if set(reference['train']) != train or set(reference['score']) != score or set(reference['groups'])-score:
        raise RuntimeError('frozen prior partition differs')
    ep.check_graph(base_graph(reference), 'prior_score', score | set(excluded), train)
    candidates = deserialize(reference['candidates']); prior = np.asarray(reference['scores']['b'])
    q, p = candidate_features(candidates, windows, owners, score)
    eligible = np.isfinite(p).any(axis=1)
    scores = dict(b=prior, q=predict(models['q'], q, prior, eligible),
                  p=predict(models['p'], np.column_stack([q, p]), prior, eligible))
    record = dict(reference=(T1/f'{prefix}_scores.json').as_posix(), scores={a: v.tolist() for a, v in scores.items()},
                  models=models, eligibility=eligible.tolist(), train=sorted(train), score=sorted(score),
                  training_graphs=graphs, prior_graph=base_graph(reference), outer_calls=0,
                  fits=dict(producer=30, baseline=6, residual=sum(m['active'] for m in models.values())),
                  elapsed_seconds=time.perf_counter()-started)
    write_json(Path(root)/OUTPUT/f'{prefix}_scores.json', record)
    return candidates, reference['groups'], scores, models


def worker(root, wrapper, phase):
    root = Path(root); out = root/OUTPUT
    if phase == 'health':
        require_committed(root, out/'policy_seal.json')
        require_committed(root, out/'policy_resources.json')
        verify_hashes(out, read_json(out/'policy_seal.json'))
    from threadpoolctl import threadpool_limits
    from .. import event_probe as ep
    from dataclasses import asdict
    r = registration(root); manifest = r['manifest']
    data, config = load_runtime(root, manifest)
    verify_hashes(root/SOURCE, read_json(root/SOURCE/'cache_seal.json'))
    windows = read_windows(root/SOURCE/'windows.npz')
    if read_json(root/SOURCE/'cache_resources.json')['completed_files'] != 696:
        raise RuntimeError('incomplete residual signal cache')
    owners = manifest['subject_by_session']
    for sid in owners:
        windows.setdefault(sid, [])
    excluded = set(manifest['identities']['v7']) | set(manifest['identities']['outer'])
    k = wrapper['id']
    with threadpool_limits(limits=4):
        if phase == 'policy':
            inner_data = ep.slice_outer_train(data, wrapper['train'])
            for j, part in enumerate(wrapper['policy_splits']):
                fit_split(root, inner_data, config, part['train'], part['score'],
                          excluded | set(wrapper['health']), windows, owners, f'wrapper{k}_policy{j}')
            original = read_json(root/T1/f'wrapper{k}_policies.json')['policies']['b']
            write_json(out/f'wrapper{k}_policies.json', dict(wrapper=k, policies={a: original for a in ('b', 'q', 'p')}, outer_calls=0))
            return dict(wrapper=k)
        candidates, groups, scores, models = fit_split(root, data, config, wrapper['train'], wrapper['health'], excluded,
                                                     windows, owners, f'wrapper{k}_health')
        truths = ep.slice_outer_train(data, wrapper['health']).train_truths
        policies = read_json(out/f'wrapper{k}_policies.json')['policies']
        result = dict(wrapper=k, events={}, predictions={}, subject_events={}, short_recall={}, models=models,
                      truths=[asdict(t) for t in truths], outer_calls=0)
        predictions = {}
        for a in ('b', 'q', 'p'):
            ps = ep.decode_policy(candidates, scores[a], groups, policies[a])
            predictions[a] = ps
            result['events'][a] = asdict(ep.compute_event_metrics(ps, truths))
            result['predictions'][a] = [asdict(p) for p in ps]
            result['short_recall'][a] = short_recall(ps, truths)
            result['subject_events'][a] = {s: asdict(ep.compute_event_metrics(
                [p for p in ps if owners[p.sid] == s], [t for t in truths if owners[t.sid] == s])) for s in wrapper['health']}
        result['gt_migration'] = gt_migration(predictions, truths)
        result['availability'] = read_json(out/f'wrapper{k}_health_scores.json')['eligibility']
        if result['predictions']['b'] != read_json(root/T1/f'wrapper{k}_health.json')['predictions']['b']:
            raise RuntimeError('residual B decoded predictions differ')
        write_json(out/f'wrapper{k}_health.json', result)
        return result


def execute(root, phase):
    from .. import event_probe as ep
    root = Path(root); out = root/OUTPUT; r = registration(root)
    for n in ('registry.json', 'state.json'):
        require_committed(root, out/n)
    if phase == 'health':
        require_committed(root, out/'policy_seal.json')
        require_committed(root, out/'policy_resources.json')
        verify_hashes(out, read_json(out/'policy_seal.json'))
    expected, success = ('REGISTERED', 'POLICIES_FROZEN') if phase == 'policy' else ('POLICIES_FROZEN', 'HEALTH_COMPLETE')
    prior_time = read_json(out/'policy_resources.json')['elapsed_seconds'] if phase == 'health' else 0
    start = time.perf_counter(); res = dict(phase=phase, completed_wrappers=0, peak_rss_bytes=0, outer_calls=0)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '4'
    with stage(out, expected, success):
        pool = ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn'))
        records, last_check = [], 0
        try:
            pending = {pool.submit(worker, str(root), w, phase) for w in r['manifest']['wrappers']}
            while pending:
                now = time.perf_counter()
                if now-last_check >= 1:
                    rss = owned_rss([os.getpid()]+[p.pid for p in multiprocessing.active_children()])
                    res['peak_rss_bytes'] = max(res['peak_rss_bytes'], rss); res['elapsed_seconds'] = now-start
                    if prior_time+now-start > 21600 or rss > 32*2**30 or output_size(out) > 64*2**20 or shutil.disk_usage(root).free < 20*2**30:
                        raise RuntimeError('residual total budget exceeded')
                    write_json(out/f'{phase}_resources.json', res); last_check = now
                done, pending = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                for future in done:
                    records.append(future.result()); res['completed_wrappers'] += 1
        except BaseException:
            for child in multiprocessing.active_children():
                child.terminate()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        if phase == 'health':
            records.sort(key=lambda r: r['wrapper'])
            metrics = {a: ep.counts_metrics([r['events'][a] for r in records]) for a in ('b', 'q', 'p')}
            if (metrics['b']['n_tp'], metrics['b']['fp'], metrics['b']['fn']) != (66, 20, 31):
                raise RuntimeError('original residual baseline differs')
            verdict = gate(metrics['b'], metrics['q'], metrics['p'],
                           [r['events']['p']['f1'] >= r['events']['b']['f1'] for r in records])
            subjects = {s: {a: r['subject_events'][a][s]['f1'] for a in ('b', 'q', 'p')}
                        for r in records for s in r['subject_events']['b']}
            diagnostics = {}
            for a in ('b', 'q', 'p'):
                values = [s[a] for s in subjects.values()]
                delta = [s[a]-s['b'] for s in subjects.values()]
                diagnostics[a] = dict(median=float(np.median(values)), iqr=np.percentile(values, [25, 75]).tolist(),
                                      p10=float(np.percentile(values, 10)), improved=sum(d > 0 for d in delta),
                                      worsened=sum(d < 0 for d in delta), tied=sum(d == 0 for d in delta))
            migration = {a: {key: sum(len(r['gt_migration'][a][key+'_indices']) for r in records)
                             for key in ('gained', 'lost', 'matched')} for a in ('b', 'q', 'p')}
            availability = dict(eligible=sum(sum(r['availability']) for r in records),
                                total=sum(len(r['availability']) for r in records))
            write_json(out/'result.json', dict(metrics=metrics, gate=verdict, records=records,
                                               gt_migration=migration, subject_diagnostics=diagnostics, availability=availability))
            text = '# A7 P2 冻结先验残差结果\n\nT24开发证据，政策固定；outer_calls=0。\n\n|arm|F1|TP|FP|FN|\n|---|---:|---:|---:|---:|\n'
            for a, m in metrics.items():
                text += f'|{a}|{m["f1"]:.6f}|{m["n_tp"]}|{m["fp"]}|{m["fn"]}|\n'
            text += '\n## GT迁移与可用性\n\n'+str(migration)+'\n\n'+str(availability)+'\n'
            for row in records:
                text += f'\nWrapper {row["wrapper"]}: '+str(row['gt_migration'])+'\n'
            text += '\n## 逐人诊断\n\n'+str(diagnostics)+'\n'
            text += '\nGate: '+str(verdict)+'\n\n正式F1仍.6514285714，未变更dist。\n'
            (out/'report.md').write_text(text, encoding='utf-8')
        res['elapsed_seconds'] = time.perf_counter()-start
        write_json(out/f'{phase}_resources.json', res)
        if output_size(out) > 64*2**20 or prior_time+res['elapsed_seconds'] > 21600:
            raise RuntimeError('residual post-write budget exceeded')
        if phase == 'policy':
            seal_policy(out)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['register', 'policy', 'health'])
    args = parser.parse_args(); root = Path.cwd()
    register(root) if args.phase == 'register' else execute(root, args.phase)


if __name__ == '__main__':
    main()
