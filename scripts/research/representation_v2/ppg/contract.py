"""Fixed audit schema and label-free candidate identity projection."""
import hashlib
import json
from pathlib import Path

T1 = Path('outputs/challenger_70/representation_v2/amendment_a6_transformer/t1')
OUTPUT = Path('outputs/challenger_70/representation_v2/amendment_a7_ppg')
SPEC = 'docs/challenge_070_amendment_a7_ppg.md'
PLAN = 'docs/superpowers/plans/2026-10-14-a7-ppg-p0-plan.md'
PARTITIONS = tuple(f'wrapper{k}_{part}' for k in range(3)
                   for part in ('health', 'policy0', 'policy1', 'policy2'))
SCHEMA = {'version': 'a7-p0-v1', 'hz': 25, 'packet_rows': 15, 'window_packets': 50,
          'packet_gap_ms': [450, 750], 'band': [.7, 3.3], 'peak_band': [.7, 3.],
          'peak_radius': .12, 'concentration_min': .15, 'peak_difference_max': 1/6,
          'coverage_min': .5, 'windows_min': 6, 'anchor_shifts_ms': [0, -560]}
Q_NAMES = [f'{phase}_{stat}' for phase in ('pre', 'during', 'late')
           for stat in ('coverage', 'pair_coverage', 'motion')]
P_NAMES = [f'{phase}_minus_pre_{stat}' for phase in ('during', 'late')
           for stat in ('frequency', 'rms3_dc', 'rms5_dc')]


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def build_inventory(manifest, partitions, sensor_dir):
    ids = manifest['identities']
    t24 = set(ids['t24'])
    if not t24 or t24 & (set(ids['v7']) | set(ids['outer'])):
        raise ValueError('invalid subject isolation')
    owners = manifest['subject_by_session']
    keys, occurrences, lookup = [], [], {}
    for name, payload in sorted(partitions.items()):
        for index, row in enumerate(payload['candidates']):
            e = row['event']
            sid = e['sid']
            if (sid not in owners or owners[sid] not in t24 or
                    not sid or any(c in sid for c in '/\\:') or sid in ('.', '..')):
                raise ValueError('invalid candidate ownership/path')
            start, end = int(e['start_ms']), int(e['end_ms'])
            if start != e['start_ms'] or end != e['end_ms'] or end <= start:
                raise ValueError('invalid candidate interval')
            key = (sid, start, end)
            if key not in lookup:
                lookup[key] = len(keys)
                keys.append(dict(sid=sid, start_ms=start, end_ms=end))
            occurrences.append(dict(partition=name, row=index, key_index=lookup[key]))
    # Resolve only after *all* candidate ownership checks have passed.
    sensor_dir = Path(sensor_dir).resolve()
    files = []
    for sid in sorted({k['sid'] for k in keys}):
        directory = (sensor_dir/sid).resolve()
        if not directory.is_relative_to(sensor_dir):
            raise ValueError('raw directory escapes sensor root')
        found = sorted(p.resolve() for p in directory.glob('collect_data*') if p.is_file())
        if len(found) != 1 or not found[0].is_relative_to(sensor_dir):
            raise ValueError('expected exactly one contained collect_data file')
        files.append(dict(sid=sid, path=str(found[0]), bytes=found[0].stat().st_size))
    return dict(keys=keys, occurrences=occurrences, files=files,
                subject_by_sid={sid: owners[sid] for sid in sorted({k['sid'] for k in keys})})


def validate_inventory(inventory, expected=(2034, 2411, 227)):
    actual = tuple(len(inventory.get(k, [])) for k in ('keys', 'occurrences', 'files'))
    if actual != expected:
        raise ValueError(f'candidate inventory differs: {actual} != {expected}')
    for o in inventory['occurrences']:
        if not 0 <= o['key_index'] < actual[0]:
            raise ValueError('invalid occurrence mapping')


def candidate_inventory(root):
    from src.config import SENSOR_DIR
    root = Path(root)
    paths = [root/T1/'manifest.json'] + [root/T1/f'{p}_scores.json' for p in PARTITIONS]
    result = build_inventory(read_json(paths[0]),
                             {p: read_json(root/T1/f'{p}_scores.json') for p in PARTITIONS}, SENSOR_DIR)
    for f in result['files']:
        f['path'] = Path(f['path']).relative_to(root).as_posix()
    result['input_hashes'] = {p.relative_to(root).as_posix(): file_hash(p) for p in paths}
    validate_inventory(result)
    return result
