"""Complete deterministic T24 waveform cache; unknown cache is an error."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

from ..ppg.audit import (write_json, require_committed, verify_hashes, protected_snapshot,
                         create_output, stage, owned_rss, output_size)
from ..ppg.contract import T1, OUTPUT as P0, SCHEMA, read_json, file_hash, build_inventory
from ..ppg.candidate import aggregate_candidate
from ..ppg.window import extract_file

OUTPUT = Path('outputs/challenger_70/representation_v2/amendment_a7_ppg_fusion/attempt_02')
SPEC = 'docs/challenge_070_amendment_a7_p1_ppg_fusion.md'
COLUMNS = ['start_ms', 'end_ms', 'valid', 'pair_pass', 'motion', 'frequency', 'rms3_dc', 'rms5_dc',
           'peak3', 'peak5', 'concentration3', 'concentration5', 'implied_hz', 'packet_span_ms']


def resource_gate(elapsed, rss, size, free):
    if elapsed > 1800 or rss > 16*2**30 or size > 128*2**20 or free < 20*2**30:
        raise RuntimeError('PPG complete-cache budget exceeded')


def complete_cache(counts, retained):
    n = counts['complete_valid_packets']-counts['tail_packets']
    return n >= 0 and n % 50 == 0 and retained == n//50


def read_windows(path):
    with np.load(path, allow_pickle=False) as z:
        if z['columns'].tolist() != COLUMNS or z['values'].shape != (len(z['sid']), len(COLUMNS)):
            raise ValueError('window cache schema differs')
        result = {}
        for sid, row in zip(z['sid'], z['values']):
            w = dict(zip(COLUMNS, row.tolist())); w['sid'] = str(sid)
            w['valid'] = bool(w['valid']); w['pair_pass'] = bool(w['pair_pass'])
            result.setdefault(str(sid), []).append(w)
    return result


def candidate_features(candidates, windows, owners, allowed):
    q, p = [], []
    for c in candidates:
        e = c.event
        if e.sid not in windows or e.sid not in owners or owners[e.sid] not in allowed:
            raise ValueError('unregistered candidate session/subject; not missing PPG')
        r = aggregate_candidate(asdict(e), windows[e.sid])
        q.append(r['q']); p.append(r['p'])
    return np.asarray(q, float).reshape(-1, 9), np.asarray(p, float).reshape(-1, 6)


def registration(root):
    root = Path(root); out = root/OUTPUT
    reg = read_json(out/'registry.json'); inv = read_json(out/'inventory.json')
    verify_hashes(root, reg['files'])
    if reg['schema'] != SCHEMA or file_hash(out/'inventory.json') != reg['inventory_sha256']:
        raise RuntimeError('registered fusion schema changed')
    if protected_snapshot(root) != reg['protected_index']:
        raise RuntimeError('protected source changed')
    return reg, inv


def register(root):
    import subprocess
    from src.config import SENSOR_DIR
    from ..ppg.audit import verify as verify_p0
    from src.pipeline.runner import FilesystemDataSource
    from src.data.manifests import INDEX_CSV, MEALS_CSV, BLACKLIST_FILE
    root = Path(root); out = root/OUTPUT
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root).strip():
        raise RuntimeError('commit before fusion registration')
    verify_p0(root)
    verify_hashes(root/P0, read_json(root/P0/'report_seal.json'))
    manifest = read_json(root/T1/'manifest.json')
    events = [dict(event=dict(sid=sid, start_ms=b[0], end_ms=b[1]))
              for sid, b in manifest['session_bounds_by_sid'].items()]
    inv = build_inventory(manifest, {'sessions': {'candidates': events}}, SENSOR_DIR)
    if len(inv['files']) != 696 or len(inv['subject_by_sid']) != 696:
        raise ValueError('full T24 session universe differs')
    p0_windows = read_windows(root/P0/'window_stats.npz')
    stats = {s['sid']: s for s in read_json(root/P0/'file_stats.json')}
    paths = [SPEC, (T1/'manifest.json').as_posix()]
    paths += [p.relative_to(root).as_posix() for directory in ('ppg', 'ppg_fusion')
              for p in (root/'scripts/research/representation_v2'/directory).glob('*.py')]
    paths += [f'scripts/research/representation_v2/{name}.py' for name in ('crossfit', 'event_probe')]
    paths += [(P0/name).as_posix() for name in ('window_stats.npz', 'file_stats.json', 'extraction_seal.json', 'report_seal.json')]
    paths += [(T1/f'{part}_scores.json').as_posix() for part in
              [f'wrapper{k}_{p}' for k in range(3) for p in ('health', 'policy0', 'policy1', 'policy2')]]
    paths += [(T1/f'wrapper{k}_health.json').as_posix() for k in range(3)]
    paths += [(T1/f'wrapper{k}_policies.json').as_posix() for k in range(3)]
    source = FilesystemDataSource(root)
    paths += [p.relative_to(root).as_posix() for role in ('train', 'meal_train', 'no_meal_train')
              for p in (source._split_path(0, role), source._micro_split_path(0, role))]
    paths += [(Path('cache/sessions')/f'{sid}.npz').as_posix() for sid in manifest['subject_by_session']]
    paths += ['cache/splits/fold0.json']
    paths += [p.relative_to(root).as_posix() for p in (INDEX_CSV, MEALS_CSV, BLACKLIST_FILE) if p.exists()]
    hashes = {p: file_hash(root/p) for p in paths}
    create_output(out); write_json(out/'state.json', dict(status='REGISTERING', outer_calls=0))
    started = time.perf_counter()
    try:
        reused = 0
        for i, f in enumerate(inv['files']):
            raw = Path(f['path']); f['path'] = raw.relative_to(root).as_posix()
            f['sha256'] = file_hash(raw)
            f['reuse_p0'] = (f['sid'] in stats and complete_cache(stats[f['sid']]['counts'], len(p0_windows.get(f['sid'], []))))
            if f['reuse_p0'] and f['sha256'] != stats[f['sid']]['sha256']:
                raise RuntimeError('P0 raw identity changed')
            reused += f['reuse_p0']
            if i % 100 == 0:
                print(f'P1 hash {i+1}/696', flush=True)
        if reused != 165:
            raise ValueError('registered full-cache reuse count differs')
        write_json(out/'inventory.json', inv)
        write_json(out/'registry.json', dict(files=hashes, schema=SCHEMA, manifest=manifest,
            inventory_sha256=file_hash(out/'inventory.json'), protected_index=protected_snapshot(root),
            reuse_files=reused, raw_hash_bytes=sum(f['bytes'] for f in inv['files']),
            raw_hash_seconds=time.perf_counter()-started))
        write_json(out/'state.json', dict(status='REGISTERED', outer_calls=0))
    except BaseException as exc:
        write_json(out/'state.json', dict(status='INCOMPLETE', error=repr(exc), outer_calls=0))
        raise


def build_cache(root):
    import shutil
    root = Path(root); out = root/OUTPUT
    reg, inv = registration(root)
    for name in ('registry.json', 'inventory.json', 'state.json'):
        require_committed(root, out/name)
    old = read_windows(root/P0/'window_stats.npz')
    windows, stats, started = [], [], time.perf_counter()
    res = dict(completed_files=0, reused_files=0, extracted_bytes=0, peak_rss_bytes=0, outer_calls=0)
    for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[variable] = '1'
    with stage(out, 'REGISTERED', 'CACHE_SEALED'):
        for f in inv['files']:
            if f['reuse_p0']:
                windows.extend(old.get(f['sid'], [])); res['completed_files'] += 1; res['reused_files'] += 1
        pool = ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn'))
        last_check = 0
        try:
            pending = {pool.submit(extract_file, root/f['path'], f['sid']): f for f in inv['files'] if not f['reuse_p0']}
            while pending:
                now = time.perf_counter()
                if now-last_check >= 1:
                    rss = owned_rss([os.getpid()]+[p.pid for p in multiprocessing.active_children()])
                    res['peak_rss_bytes'] = max(res['peak_rss_bytes'], rss)
                    res['elapsed_seconds'] = now-started
                    resource_gate(now-started, rss, output_size(out), shutil.disk_usage(root).free)
                    write_json(out/'cache_resources.json', res); last_check = now
                done, _ = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                for future in done:
                    f = pending.pop(future); row = future.result()
                    if row['sha256'] != f['sha256'] or row['bytes'] != f['bytes']:
                        raise RuntimeError('PPG raw changed during cache build')
                    windows.extend(row.pop('windows')); stats.append(row)
                    res['completed_files'] += 1; res['extracted_bytes'] += row['bytes']
                    if res['completed_files'] % 50 == 0:
                        print(f'P1 complete cache {res["completed_files"]}/696', flush=True)
        except BaseException:
            for child in multiprocessing.active_children():
                child.terminate()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        windows.sort(key=lambda w: (w['sid'], w['start_ms']))
        np.savez_compressed(out/'windows.npz', sid=np.asarray([w['sid'] for w in windows]), columns=np.asarray(COLUMNS),
            values=np.asarray([[w[c] for c in COLUMNS] for w in windows], float).reshape(-1, len(COLUMNS)))
        write_json(out/'file_stats.json', stats)
        res['elapsed_seconds'] = time.perf_counter()-started; res['windows'] = len(windows)
        write_json(out/'cache_resources.json', res)
        resource_gate(time.perf_counter()-started, owned_rss([os.getpid()]), output_size(out), shutil.disk_usage(root).free)
        write_json(out/'cache_seal.json', {n: file_hash(out/n) for n in ('windows.npz', 'file_stats.json', 'cache_resources.json')})
