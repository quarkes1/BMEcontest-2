"""Registered, budgeted P0 extraction. CLI has no training or held-out access."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from .candidate import aggregate_candidate, phases
from .contract import (OUTPUT, T1, SPEC, PLAN, SCHEMA, Q_NAMES, P_NAMES,
                       candidate_inventory, file_hash, read_json, validate_inventory)
from .window import extract_file

PROTECTED = ['src', 'dist', str(T1), 'scripts/research/representation_v2/transformer']


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    try:
        temporary.write_text(json.dumps(clean_json(value), ensure_ascii=False,
                                       sort_keys=True, allow_nan=False, indent=2)+'\n', encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def create_output(path):
    Path(path).mkdir(parents=True, exist_ok=False)


def assert_state(state, expected):
    if state['status'] != expected:
        raise RuntimeError('unexpected audit state; no overwrite or implicit retry')


def require_committed(root, path):
    relative = Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    try:
        committed = subprocess.check_output(['git', 'rev-parse', 'HEAD:'+relative], cwd=root,
                                            text=True, stderr=subprocess.DEVNULL).strip()
        working = subprocess.check_output(['git', 'hash-object', '--path='+relative, str(path)],
                                          cwd=root, text=True).strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError('artifact must be committed: '+relative) from exc
    if committed != working:
        raise RuntimeError('artifact differs from committed seal: '+relative)


def verify_hashes(root, hashes):
    for relative, expected in hashes.items():
        if file_hash(Path(root)/relative) != expected:
            raise RuntimeError('registered input changed: '+relative)


def protected_snapshot(root):
    if subprocess.check_output(['git', 'status', '--porcelain', '--', *PROTECTED], cwd=root):
        raise RuntimeError('protected paths modified')
    return subprocess.check_output(['git', 'ls-files', '-s', '--', *PROTECTED], cwd=root).decode()


def check_budget(elapsed, rss, output_bytes, free_bytes):
    if elapsed > 7200 or rss > 16*2**30 or output_bytes > 128*2**20 or free_bytes < 20*2**30:
        raise RuntimeError('P0 resource budget exceeded')


def output_size(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def owned_rss(pids):
    ids = ','.join(str(int(p)) for p in pids)
    command = f'(Get-Process -Id {ids} -ErrorAction SilentlyContinue | Measure-Object WorkingSet64 -Sum).Sum'
    out = subprocess.check_output(['powershell', '-NoProfile', '-Command', command], text=True)
    return int(float(out.strip() or '0'))


def validate_arrays(q, p, n):
    if q.shape != (n, 9) or p.shape != (n, 6):
        raise ValueError('feature width or identity differs')
    cov = q[:, [0, 1, 3, 4, 6, 7]]
    if not np.isfinite(cov).all() or np.any((cov < 0) | (cov > 1)) or np.isinf(q).any() or np.isinf(p).any():
        raise ValueError('invalid feature coverage/value')


def register(root):
    root = Path(root)
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root).strip():
        raise RuntimeError('commit code before registration')
    inventory = candidate_inventory(root)
    paths = [SPEC, PLAN] + [p.relative_to(root).as_posix() for p in
                           (root/'scripts/research/representation_v2/ppg').glob('*.py')]
    from src.data.manifests import INDEX_CSV, BLACKLIST_FILE, MEALS_CSV
    paths.extend(p.relative_to(root).as_posix() for p in (INDEX_CSV, BLACKLIST_FILE, MEALS_CSV) if p.exists())
    paths.extend((T1/f'wrapper{k}_health.json').as_posix() for k in range(3))
    hashes = {**inventory['input_hashes'], **{p: file_hash(root/p) for p in paths}}
    output = root/OUTPUT
    check_budget(0, owned_rss([os.getpid()]), 0, shutil.disk_usage(root).free)
    create_output(output)
    write_json(output/'state.json', {'status': 'REGISTERING', 'outer_calls': 0, 'training_fits': 0})
    started = time.perf_counter()
    try:
        for index, f in enumerate(inventory['files']):
            f['sha256'] = file_hash(root/f['path'])
            if index % 25 == 0:
                print(f'hash {index+1}/{len(inventory["files"])}', flush=True)
        write_json(output/'inventory.json', inventory)
        write_json(output/'registry.json', dict(schema=SCHEMA, q_names=Q_NAMES, p_names=P_NAMES,
                   files=hashes, inventory_sha256=file_hash(output/'inventory.json'),
                   protected_index=protected_snapshot(root), git_head=subprocess.check_output(
                       ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                   registration_hash_bytes=sum(f['bytes'] for f in inventory['files']),
                   registration_hash_seconds=time.perf_counter()-started))
        write_json(output/'state.json', {'status': 'REGISTERED', 'outer_calls': 0, 'training_fits': 0})
    except BaseException as exc:
        write_json(output/'state.json', {'status': 'INCOMPLETE', 'error': repr(exc)})
        raise


def load_registration(root):
    root = Path(root); output = root/OUTPUT
    reg, inventory = read_json(output/'registry.json'), read_json(output/'inventory.json')
    verify_hashes(root, reg['files'])
    if reg['schema'] != SCHEMA or file_hash(output/'inventory.json') != reg['inventory_sha256']:
        raise RuntimeError('schema or inventory identity changed')
    if protected_snapshot(root) != reg['protected_index']:
        raise RuntimeError('protected index changed')
    validate_inventory(inventory)
    return reg, inventory


@contextmanager
def stage(output, expected, completed):
    state = read_json(output/'state.json'); assert_state(state, expected)
    write_json(output/'state.json', {**state, 'status': completed+'_RUNNING'})
    try:
        yield
    except BaseException as exc:
        write_json(output/'state.json', {**state, 'status': 'INCOMPLETE', 'error': repr(exc)})
        raise
    else:
        write_json(output/'state.json', {**state, 'status': completed})


def extract(root):
    root = Path(root); output = root/OUTPUT
    reg, inventory = load_registration(root)
    for name in ('registry.json', 'inventory.json', 'state.json'):
        require_committed(root, output/name)
    resources = dict(started_epoch=time.time(), extraction_bytes=0, completed_files=0,
                     peak_sampled_rss_bytes=0, elapsed_seconds=0, outer_calls=0, training_fits=0,
                     registration_hash_bytes=reg['registration_hash_bytes'])
    start = time.perf_counter(); last_check = 0.
    windows, file_stats = [], []
    futures = {}
    for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[variable] = '1'

    def monitor(force=False):
        nonlocal last_check
        now = time.perf_counter()
        if force or now-last_check >= 1:
            rss = owned_rss([os.getpid()] + [p.pid for p in multiprocessing.active_children()])
            resources['peak_sampled_rss_bytes'] = max(rss, resources['peak_sampled_rss_bytes'])
            resources['elapsed_seconds'] = now-start
            check_budget(now-start, rss, output_size(output), shutil.disk_usage(root).free)
            write_json(output/'resources.json', resources)
            last_check = now

    with stage(output, 'REGISTERED', 'EXTRACTED'):
        pool = ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn'))
        try:
            futures = {pool.submit(extract_file, root/f['path'], f['sid']): f for f in inventory['files']}
            while futures:
                monitor()
                done, _ = wait(futures, timeout=.1, return_when=FIRST_COMPLETED)
                for future in done:
                    f = futures.pop(future); result = future.result()
                    if result['sha256'] != f['sha256'] or result['bytes'] != f['bytes']:
                        raise RuntimeError('raw input changed during extraction')
                    needed = [phase for key in inventory['keys'] if key['sid'] == f['sid'] for phase in phases(key)]
                    kept = [w for w in result.pop('windows') if any(
                        w['end_ms'] > s and w['start_ms']-560 < e for s, e in needed)]
                    windows.extend(kept); file_stats.append(result)
                    resources['extraction_bytes'] += result['bytes']; resources['completed_files'] += 1
                    print(f'extract {resources["completed_files"]}/227 retained_windows={len(windows)}', flush=True)
        except BaseException:
            for p in multiprocessing.active_children():
                p.terminate()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        windows.sort(key=lambda w: (w['sid'], w['start_ms'], w['end_ms']))
        by_sid = {}
        for w in windows:
            by_sid.setdefault(w['sid'], []).append(w)
        aggregates = {shift: [] for shift in (0, -560)}
        for key in inventory['keys']:
            monitor()
            for shift in aggregates:
                aggregates[shift].append(aggregate_candidate(key, by_sid.get(key['sid'], []), shift))
        values = {}
        for shift, rows in aggregates.items():
            suffix = 'main' if shift == 0 else 'shift'
            for name in ('q', 'p'):
                values[name+'_'+suffix] = np.asarray([r[name] for r in rows], dtype=float)
            for name in ('proxies', 'proxy_coverage', 'proxy_counts'):
                values[name+'_'+suffix] = np.asarray([[p[name] for p in r['phases']] for r in rows], dtype=float)
            values['requested_ms_'+suffix] = np.asarray([[p['requested_ms'] for p in r['phases']] for r in rows])
            validate_arrays(values['q_'+suffix], values['p_'+suffix], len(inventory['keys']))
        np.savez_compressed(output/'candidate_features.npz', **values)
        columns = ['start_ms', 'end_ms', 'valid', 'pair_pass', 'motion', 'frequency',
                   'rms3_dc', 'rms5_dc', 'peak3', 'peak5', 'concentration3', 'concentration5',
                   'implied_hz', 'packet_span_ms']
        np.savez_compressed(output/'window_stats.npz', sid=np.asarray([w['sid'] for w in windows]),
                            columns=np.asarray(columns), values=np.asarray([[w[c] for c in columns] for w in windows]))
        write_json(output/'file_stats.json', sorted(file_stats, key=lambda f: f['sid']))
        monitor(force=True)
        seal_names = ['candidate_features.npz', 'window_stats.npz', 'file_stats.json', 'resources.json']
        write_json(output/'extraction_seal.json', {name: file_hash(output/name) for name in seal_names})
    verify(root)


def verify(root):
    root = Path(root); output = root/OUTPUT
    _, inv = load_registration(root)
    state = read_json(output/'state.json')
    if state['status'] not in ('EXTRACTED', 'REPORTED') or state.get('outer_calls') != 0 or state.get('training_fits') != 0:
        raise RuntimeError('not a completed zero-training audit')
    verify_hashes(output, read_json(output/'extraction_seal.json'))
    with np.load(output/'candidate_features.npz', allow_pickle=False) as features:
        for suffix in ('main', 'shift'):
            validate_arrays(features['q_'+suffix], features['p_'+suffix], len(inv['keys']))
    if output_size(output) > 128*2**20:
        raise RuntimeError('persistent audit budget exceeded')
    print('A7 seal verified', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['register', 'extract', 'verify'])
    args = parser.parse_args()
    globals()[args.command](Path.cwd())


if __name__ == '__main__':
    main()
