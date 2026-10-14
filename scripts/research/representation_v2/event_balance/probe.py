"""T24-only weighted verifier experiment with sealed policy/health boundary."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from threadpoolctl import threadpool_limits
from .. import event_probe as ep
from ..ppg.audit import (write_json, require_committed, verify_hashes, protected_snapshot,
                         create_output, stage, owned_rss, output_size)
from ..ppg.contract import T1, file_hash, read_json
from ..ppg_fusion.cache import registration as source_registration
from ..ppg_fusion.probe import load_runtime, reproduce, short_recall
from ..ppg_residual.probe import gt_migration, check_partition
from .model import sample_weights, probabilities

OUTPUT = Path('outputs/challenger_70/representation_v2/amendment_a9_event_balance')
SPEC = 'docs/challenge_070_amendment_a9_event_balanced_loss.md'


def register(root):
    root = Path(root)
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root).strip():
        raise RuntimeError('commit before A9 registration')
    old, _ = source_registration(root)
    files = dict(old['files']); files[SPEC] = file_hash(root/SPEC)
    for p in (root/'scripts/research/representation_v2/event_balance').glob('*.py'):
        files[p.relative_to(root).as_posix()] = file_hash(p)
    for path in ('scripts/research/representation_v2/ppg_residual/probe.py', 'scripts/research/representation_v2/ppg_fusion/probe.py'):
        files[path] = file_hash(root/path)
    out = root/OUTPUT; create_output(out)
    write_json(out/'registry.json', dict(files=files, manifest=old['manifest'], protected_index=protected_snapshot(root), outer_calls=0))
    write_json(out/'state.json', dict(status='REGISTERED', outer_calls=0))


def registration(root):
    r = read_json(Path(root)/OUTPUT/'registry.json'); verify_hashes(root, r['files'])
    if protected_snapshot(root) != r['protected_index']:
        raise RuntimeError('protected A9 release changed')
    return r


def score_split(root, data, config, train, score, excluded, prefix):
    check_partition(train, score, excluded); graph = {}; start = time.perf_counter()
    t, v = ep.build_candidate_ledgers(data, config, train, score, fit_audit=graph)
    if set(t.groups)-set(train) or set(v.groups)-set(score):
        raise RuntimeError('A9 ledger subject mismatch')
    base = ep.verifier_probabilities(t.features, t.labels, v.features, seed=config.seed)
    reference = read_json(Path(root)/T1/f'{prefix}_scores.json')
    reproduce(reference, [asdict(c) for c in v.candidates], v.groups, v.labels, base)
    if set(reference['train']) != set(train) or set(reference['score']) != set(score):
        raise RuntimeError('A9 prior partition differs')
    truths = ep.slice_outer_train(data, train).train_truths
    weights = sample_weights(t.candidates, t.labels, t.groups, truths, data.subject_by_session)
    w = probabilities(t.features, t.labels, v.features, weights, seed=config.seed)
    graph['weighted_verifier'] = dict(fit_subjects=list(train), parents=['candidates'])
    ep.check_graph(graph, 'weighted_verifier', set(score)|set(excluded), set(train))
    scores = dict(b=np.asarray(reference['scores']['b']), w=w)
    write_json(Path(root)/OUTPUT/f'{prefix}_scores.json', dict(candidates=[asdict(c) for c in v.candidates],
        groups=v.groups.tolist(), labels=v.labels.tolist(), train=list(train), score=list(score), graph=graph,
        scores={a: p.tolist() for a, p in scores.items()}, weight_diagnostic=dict(min=float(weights.min()),
        max=float(weights.max()), mean=float(weights.mean()), effective_n=float(weights.sum()**2/np.dot(weights, weights)),
        n_candidates=len(weights)), elapsed_seconds=time.perf_counter()-start, outer_calls=0))
    return v.candidates, v.groups, scores


def check_policy(root, out):
    for n in ('policy_seal.json', 'policy_resources.json'):
        require_committed(root, out/n)
    verify_hashes(out, read_json(out/'policy_seal.json'))


def worker(root, wrapper, phase):
    root = Path(root); out = root/OUTPUT
    if phase == 'health': check_policy(root, out)
    r = registration(root); manifest = r['manifest']; data, config = load_runtime(root, manifest)
    excluded = set(manifest['identities']['v7'])|set(manifest['identities']['outer']); k = wrapper['id']
    with threadpool_limits(limits=4):
        if phase == 'policy':
            candidates, groups, scores = [], [], {'b': [], 'w': []}
            for j, part in enumerate(wrapper['policy_splits']):
                cs, gs, ps = score_split(root, data, config, part['train'], part['score'],
                                        excluded|set(wrapper['health']), f'wrapper{k}_policy{j}')
                candidates.extend(cs); groups.extend(gs)
                for a in scores: scores[a].extend(ps[a])
                print(f'A9 wrapper{k} policy {j+1}/3', flush=True)
            truths = ep.slice_outer_train(data, wrapper['train']).train_truths
            policy = ep.choose_policy(candidates, scores['w'], groups, truths, config)
            old = read_json(root/T1/f'wrapper{k}_policies.json')['policies']['b']
            write_json(out/f'wrapper{k}_policies.json', dict(wrapper=k, policies=dict(b=old, w=policy), outer_calls=0))
            return dict(wrapper=k)
        cs, gs, scores = score_split(root, data, config, wrapper['train'], wrapper['health'], excluded, f'wrapper{k}_health')
        truths = ep.slice_outer_train(data, wrapper['health']).train_truths
        policies = read_json(out/f'wrapper{k}_policies.json')['policies']
        predictions = {a: ep.decode_policy(cs, scores[a], gs, policies[a]) for a in scores}
        record = dict(wrapper=k, events={a: asdict(ep.compute_event_metrics(ps, truths)) for a, ps in predictions.items()},
            truths=[asdict(t) for t in truths], predictions={a: [asdict(p) for p in ps] for a, ps in predictions.items()},
            short_recall={a: short_recall(ps, truths) for a, ps in predictions.items()},
            gt_migration=gt_migration(predictions, truths), outer_calls=0)
        record['subject_events'] = {a: {s: asdict(ep.compute_event_metrics(
            [p for p in ps if manifest['subject_by_session'][p.sid] == s],
            [t for t in truths if manifest['subject_by_session'][t.sid] == s])) for s in wrapper['health']} for a, ps in predictions.items()}
        if record['predictions']['b'] != read_json(root/T1/f'wrapper{k}_health.json')['predictions']['b']:
            raise RuntimeError('A9 B decoded predictions differ')
        write_json(out/f'wrapper{k}_health.json', record); return record


def execute(root, phase):
    root = Path(root); out = root/OUTPUT; r = registration(root)
    for name in ('registry.json', 'state.json'): require_committed(root, out/name)
    if phase == 'health': check_policy(root, out)
    prior = read_json(out/'policy_resources.json')['elapsed_seconds'] if phase == 'health' else 0
    expected, success = ('REGISTERED', 'POLICIES_FROZEN') if phase == 'policy' else ('POLICIES_FROZEN', 'HEALTH_COMPLETE')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[name] = '4'
    start = time.perf_counter(); resources = dict(elapsed_seconds=0, peak_rss_bytes=0, completed_wrappers=0, outer_calls=0)
    def budget():
        resources['elapsed_seconds'] = time.perf_counter()-start
        rss = owned_rss([os.getpid()]+[p.pid for p in multiprocessing.active_children()])
        resources['peak_rss_bytes'] = max(resources['peak_rss_bytes'], rss)
        if prior+resources['elapsed_seconds'] > 3600 or rss > 16*2**30 or output_size(out) > 64*2**20 or shutil.disk_usage(root).free < 20*2**30:
            raise RuntimeError('A9 total budget exceeded')
    with stage(out, expected, success):
        pool = ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn'))
        records, last = [], 0
        try:
            pending = {pool.submit(worker, str(root), w, phase) for w in r['manifest']['wrappers']}
            while pending:
                if time.perf_counter()-last >= 1:
                    budget(); write_json(out/f'{phase}_resources.json', resources); last = time.perf_counter()
                done, pending = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                for f in done: records.append(f.result()); resources['completed_wrappers'] += 1
        except BaseException:
            for child in multiprocessing.active_children(): child.terminate()
            raise
        finally: pool.shutdown(wait=True, cancel_futures=True)
        if phase == 'health':
            records.sort(key=lambda row: row['wrapper'])
            metrics = {a: ep.counts_metrics([r['events'][a] for r in records]) for a in ('b', 'w')}
            b, w = metrics['b'], metrics['w']
            if (b['n_tp'], b['fp'], b['fn']) != (66, 20, 31): raise RuntimeError('A9 baseline differs')
            nonworse = sum(row['events']['w']['f1'] >= row['events']['b']['f1'] for row in records)
            gate = dict(passed=bool(w['f1']-b['f1'] >= .01-1e-12 and nonworse >= 2 and w['n_tp'] >= b['n_tp']-1
                        and (w['fp'] < b['fp'] or w['n_tp']-b['n_tp'] > w['fp']-b['fp'])), nonworse_wrappers=nonworse,
                        delta=w['f1']-b['f1'], outer_calls=0)
            write_json(out/'result.json', dict(metrics=metrics, gate=gate, records=records))
            text = '# A9 主体×事件均衡训练结果\n\nT24开发证据，outer_calls=0，dist未改变。\n\n|arm|F1|TP|FP|FN|\n|---|---:|---:|---:|---:|\n'
            for a, m in metrics.items(): text += f'|{a}|{m["f1"]:.6f}|{m["n_tp"]}|{m["fp"]}|{m["fn"]}|\n'
            text += '\nGate: '+str(gate)+'\n\nGT得失、逐人、短餐与权重诊断见result及各split scores。正式F1仍.6514285714。\n'
            (out/'report.md').write_text(text, encoding='utf-8')
        budget(); write_json(out/f'{phase}_resources.json', resources)
        if phase == 'policy':
            names = [f'wrapper{k}_policies.json' for k in range(3)]+['policy_resources.json']
            write_json(out/'policy_seal.json', {n: file_hash(out/n) for n in names})
        budget()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=('register', 'policy', 'health'))
    args = parser.parse_args(); register(Path.cwd()) if args.phase == 'register' else execute(Path.cwd(), args.phase)


if __name__ == '__main__': main()
