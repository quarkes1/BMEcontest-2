"""A5: partition-expanded (not population-expanded) inner development."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import time

import numpy as np
import torch

from scripts.research.representation_v2 import adapt_probe as ap
from src.pipeline.challenger.e03 import _e03_outer_subject_ids

ep = ap.ep
IDENTITY = 'RV2-A5-S15-001'
OUTPUT = 'outputs/challenger_70/representation_v2/amendment_a5_validation'
SPEC = 'docs/challenge_070_amendment_a5_inner_validation_design.md'
SALTS = (20261012, 20261013)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def partition_signature(folds):
    return tuple(sorted(tuple(sorted(h)) for _, h in folds))


def registered_partitions(subjects, historical):
    if (len(subjects) != 24 or any(not isinstance(s, str) or not s for s in subjects)
            or len(set(subjects)) != 24):
        raise ValueError('exactly 24 unique nonempty subjects required')
    repeats = []
    signatures = {partition_signature(historical)}
    for salt in SALTS:
        order = sorted(subjects, key=lambda s: (hashlib.sha256(f'{salt}:{s}'.encode()).hexdigest(), s))
        folds = tuple((tuple(sorted(set(subjects)-set(order[k::3]))), tuple(sorted(order[k::3])))
                      for k in range(3))
        signature = partition_signature(folds)
        if signature in signatures:
            raise ValueError('equivalent partition; do not replace registered salt')
        signatures.add(signature)
        repeats.append(folds)
    return tuple(repeats)


def validate_identities(identities):
    if set(identities) != {'t24', 'v7', 'outer', 'source_train', 'source_val'}:
        raise ValueError('identity fields differ')
    previous = set()
    for key, values in identities.items():
        if not values or len(set(values)) != len(values) or previous & set(values):
            raise ValueError('identity overlap or empty/duplicate population: '+key)
        previous.update(values)
    if len(identities['t24']) != 24:
        raise ValueError('T24 population changed')


def build_manifest(data, payload, config, *, root):
    train, validation = ep.inner_subject_partitions(data, n_splits=4)[0]
    sliced = ep.slice_outer_train(data, train)
    historical = ep.inner_subject_partitions(sliced, n_splits=3)
    identities = {'t24': list(train), 'v7': list(validation),
                  'outer': sorted(_e03_outer_subject_ids(root, config)),
                  'source_train': [f'FD-R-{s}' for s in payload['provenance']['source_train']],
                  'source_val': [f'FD-R-{s}' for s in payload['provenance']['source_validation']]}
    validate_identities(identities)
    if len(identities['v7']) != 7 or len(identities['source_train']) != 27 or len(identities['source_val']) != 7:
        raise ValueError('registered validation/source population changed')
    repeats = registered_partitions(train, historical)
    wrappers = [{'id': r*3+k, 'repeat': r, 'fold': k, 'train': list(t), 'health': list(h)}
                for r, folds in enumerate(repeats) for k, (t, h) in enumerate(folds)]
    return {'id': IDENTITY, 'identities': identities, 'identity_sha256': digest(identities),
            'salts': list(SALTS), 'historical_partitions': historical, 'wrappers': wrappers,
            'wrapper_ids': list(range(6)), 'config': asdict(config), 'development_only': True,
            'outer_calls': 0, 'source_sha256': ep.REGISTERED_SOURCE_SHA}


def require_clean(root):
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip():
        raise RuntimeError('commit code and evidence first')


def load_manifest(root):
    root = Path(root)
    output = root/OUTPUT
    plan = json.loads((output/'plan.json').read_text())
    if plan.get('id') != IDENTITY or file_hash(output/'manifest.json') != plan.get('manifest_sha256'):
        raise RuntimeError('registered manifest changed')
    manifest = json.loads((output/'manifest.json').read_text())
    validate_identities(manifest['identities'])
    if manifest['id'] != IDENTITY or manifest['source_sha256'] != ep.REGISTERED_SOURCE_SHA:
        raise RuntimeError('manifest contract changed')
    if file_hash(root/SPEC) != plan['spec_sha256']:
        raise RuntimeError('registered spec changed')
    for name, expected in plan['code_files'].items():
        if file_hash(root/name) != expected:
            raise RuntimeError('registered code changed: '+name)
    rebuilt = registered_partitions(tuple(manifest['identities']['t24']), manifest['historical_partitions'])
    expected = [{'id': r*3+k, 'repeat': r, 'fold': k, 'train': list(t), 'health': list(h)}
                for r, folds in enumerate(rebuilt) for k, (t, h) in enumerate(folds)]
    if manifest['wrappers'] != expected or manifest['wrapper_ids'] != list(range(6)):
        raise RuntimeError('registered wrapper topology changed')
    return manifest


def manifest_stage(root):
    root = Path(root)
    output = root/OUTPUT
    if output.exists():
        raise RuntimeError('A5 attempt exists; no overwrite')
    require_clean(root)
    snapshot = ep.protected_snapshot(root)
    code = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    files = subprocess.check_output(['git', 'ls-files', 'src', 'scripts'], cwd=root, text=True).splitlines()
    hashes = {p: file_hash(root/p) for p in files if p.endswith('.py')}
    try:
        config = ep.registered_e03_config(0, workers=4)
        payload = ep.load_source_artifact(root/'outputs/challenger_70/representation_v2/amendment_a3_source',
                                          ep.REGISTERED_SOURCE_SHA)
        data = ep.runner.FilesystemDataSource(root).load_outer_train_only(config)
        manifest = build_manifest(data, payload, config, root=root)
        ep.write_json(output/'manifest.json', manifest)
        ep.write_json(output/'plan.json', {'id': IDENTITY, 'code_sha': code, 'code_files': hashes,
                      'spec_sha256': file_hash(root/SPEC), 'manifest_sha256': file_hash(output/'manifest.json'),
                      'protected_digest': digest(snapshot), 'outer_calls': 0, 'development_only': True,
                      'producer_fits': 240, 'verifier_fits': 144, 'adaptation_readout_fits': 24,
                      'neural_optimizer_steps': 24000})
        if ep.protected_snapshot(root) != snapshot:
            raise RuntimeError('protected release changed')
        ep.write_json(output/'state.json', {'status': 'S15_MANIFEST_READY', 'outer_calls': 0,
                                           'health_files_read': 0})
        print('S15 manifest ready: commit before event stage', flush=True)
    except Exception as exc:
        ep.write_json(output/'state.json', {'status': 'S15_INCOMPLETE', 'error': str(exc), 'outer_calls': 0})
        raise


def freeze_policies(output, manifest):
    output = Path(output)
    if manifest['wrapper_ids'] != list(range(6)):
        raise RuntimeError('six registered wrappers required')
    files = {}
    for k in range(6):
        path = output/f'wrapper{k}_policy_selection.json'
        if not path.is_file():
            raise RuntimeError('all policies must exist before health')
        row = json.loads(path.read_text())
        if row['wrapper'] != k or set(row['policies']) != {'b', 'r', 'f'}:
            raise RuntimeError('policy topology changed')
        files[path.name] = file_hash(path)
    marker = {'wrapper_ids': list(range(6)), 'health_files_read': 0, 'policy_file_sha256': files,
              'manifest_sha256': file_hash(output/'manifest.json'), 'plan_sha256': file_hash(output/'plan.json')}
    ep.write_json(output/'policy_phase_frozen.json', marker)
    return file_hash(output/'policy_phase_frozen.json')


def load_all_policies(output, marker_sha, manifest):
    output = Path(output)
    marker = output/'policy_phase_frozen.json'
    if file_hash(marker) != marker_sha:
        raise RuntimeError('policy marker changed')
    value = json.loads(marker.read_text())
    expected = {f'wrapper{k}_policy_selection.json' for k in range(6)}
    if (value['wrapper_ids'] != manifest['wrapper_ids'] or value['health_files_read'] != 0
            or set(value['policy_file_sha256']) != expected
            or value['manifest_sha256'] != file_hash(output/'manifest.json')
            or value['plan_sha256'] != file_hash(output/'plan.json')):
        raise RuntimeError('all-policy freeze identity mismatch')
    result = {}
    for k in range(6):
        name = f'wrapper{k}_policy_selection.json'
        if file_hash(output/name) != value['policy_file_sha256'][name]:
            raise RuntimeError('frozen policy changed')
        row = json.loads((output/name).read_text())
        if row['wrapper'] != k or set(row['policies']) != {'b', 'r', 'f'}:
            raise RuntimeError('frozen policy contract changed')
        result[k] = row['policies']
    return result


def execute_phases(output, manifest, policy_callback, health_callback):
    for wrapper in manifest['wrappers']:
        ep.write_json(Path(output)/f"wrapper{wrapper['id']}_policy_selection.json", policy_callback(wrapper))
    marker_sha = freeze_policies(output, manifest)
    records = []
    for wrapper in manifest['wrappers']:
        policies = load_all_policies(output, marker_sha, manifest)
        records.append(health_callback(wrapper, policies[wrapper['id']]))
    return records, marker_sha


def check_resources(root, output, started, work_seconds, initial_free_cuda):
    size = sum(p.stat().st_size for p in Path(output).rglob('*') if p.is_file())
    if (time.perf_counter()-started > 28800 or work_seconds > 14400
            or size > 128*1024**2 or shutil.disk_usage(root).free < 20*1024**3):
        raise RuntimeError('registered time/disk/output budget exceeded')
    if initial_free_cuda is not None:
        if initial_free_cuda < 1280*1024**2 or torch.cuda.max_memory_allocated() > .8*initial_free_cuda:
            raise RuntimeError('registered CUDA budget exceeded')


def validate_fit_counts(counts):
    if counts != {'splits': 24, 'producer_fits': 240, 'verifier_fits': 144, 'readout_fits': 24, 'steps': 24000}:
        raise RuntimeError('actual fit counts differ from registered budget')


def validate_split_evidence(root, name, row):
    train, score, excluded = (set(row[k]) for k in ('train', 'score', 'excluded'))
    if train & (score | excluded) or score & excluded:
        raise ValueError('split ownership overlap')
    n = len(row['candidates'])
    if len(row['groups']) != n or len(row['labels']) != n or set(row['groups'])-score:
        raise ValueError('candidate ownership mismatch')
    for arm in ('b', 'r', 'f'):
        p = np.asarray(row['scores'][arm])
        if p.shape != (n,) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError('invalid aligned probability')
    # Source population uses a separate FD namespace, checked again by the caller.
    ep.check_graph(row['graph'], 'score', score | excluded,
                   train | set(row['graph'].get('source', {}).get('fit_subjects', [])))
    return digest({k: row[k] for k in ('train', 'score', 'excluded', 'candidates', 'groups', 'scores', 'graph')})


def run_event(root):
    from scripts.research.representation_v2 import validation_metrics as vm
    root = Path(root)
    output = root/OUTPUT
    require_clean(root)
    manifest = load_manifest(root)
    plan = json.loads((output/'plan.json').read_text())
    state = json.loads((output/'state.json').read_text())
    if state['status'] != 'S15_MANIFEST_READY':
        raise RuntimeError('event attempt already started; no overwrite')
    started = time.perf_counter()
    counts = {'splits': 0, 'producer_fits': 0, 'verifier_fits': 0, 'readout_fits': 0, 'steps': 0}
    records = []
    completed = []
    snapshot = ep.protected_snapshot(root)
    work = [0., None]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    free_cuda = torch.cuda.mem_get_info()[0] if device == 'cuda' else None
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    def budget():
        current = 0. if work[1] is None else time.perf_counter()-work[1]
        check_resources(root, output, started, work[0]+current, free_cuda)
    def training_budget():
        if work[1] is None:
            work[1] = time.perf_counter()
        budget()
    def state_write(status, **extra):
        ep.write_json(output/'state.json', {'status': status, 'completed_wrappers': completed,
                      'fit_counts': counts, 'outer_calls': 0, **extra})
    try:
        budget()
        if digest(snapshot) != plan['protected_digest']:
            raise RuntimeError('protected release changed since manifest')
        subprocess.run(['git', 'merge-base', '--is-ancestor', plan['code_sha'], 'HEAD'], cwd=root, check=True)
        for name in ('plan.json', 'manifest.json'):
            stored = subprocess.check_output(['git', 'show', f'HEAD:{OUTPUT}/{name}'], cwd=root)
            if stored != (output/name).read_bytes():
                raise RuntimeError('manifest and plan must be committed')
        state_write('S15_IN_PROGRESS')
        torch.set_num_threads(4)
        config = ep.registered_e03_config(0, workers=4)
        if digest(asdict(config)) != digest(manifest['config']):
            raise RuntimeError('registered producer/policy config changed')
        payload = ep.load_source_artifact(root/'outputs/challenger_70/representation_v2/amendment_a3_source',
                                          ep.REGISTERED_SOURCE_SHA)
        data = ep.runner.FilesystemDataSource(root).load_outer_train_only(config)
        if digest(build_manifest(data, payload, config, root=root)) != digest(manifest):
            raise RuntimeError('loaded population differs from manifest')
        data = ep.slice_outer_train(data, manifest['identities']['t24'])
        ids = manifest['identities']
        source_train, source_val = set(ids['source_train']), set(ids['source_val'])
        protected = set(ids['v7']) | set(ids['outer'])
        encoder = ep.MotionBranch().token_encoder
        encoder.load_state_dict(payload['encoder'], strict=True)
        encoders = {'f': encoder.requires_grad_(False).to(device).eval()}
        splitter = ap.prepare(root, payload, output, device, adaptation_options={'batch_size': 32, 'clip_norm': 1.},
                              replay=validate_split_evidence, resource_check=training_budget)
        def split(partition_data, train, score, excluded, prefix):
            budget()
            ledger, scores, graph, seconds = splitter(root, partition_data, config, train, score, excluded,
                encoders, source_train, source_val, output, prefix, device)
            work[1] = None
            work[0] += seconds
            ep.check_graph(graph, 'score', set(score) | set(excluded) | source_val, set(train) | source_train)
            report = json.loads((output/(prefix+'_adaptation.json')).read_text())
            if report['batch_size'] != 32 or report['clip_norm'] != 1. or report['steps'] != 1000:
                raise RuntimeError('adaptation contract changed')
            counts['splits'] += 1
            counts['producer_fits'] += sum(k.startswith(('macro_oof', 'micro_oof')) or k in ('macro_refit', 'micro_refit') for k in graph)
            counts['verifier_fits'] += 2*sum(k in graph for k in ('base_verifier', 'r_verifier', 'f_verifier'))
            counts['readout_fits'] += int('initial_readout' in report)
            counts['steps'] += report['steps']
            state_write('S15_IN_PROGRESS')
            budget()
            return ledger, scores, graph
        def policy(wrapper):
            train, health = wrapper['train'], wrapper['health']
            tdata = ep.slice_outer_train(data, train)
            pc, pg, ps, graph = [], [], {a: [] for a in ('b', 'r', 'f')}, {}
            for inner, (fit, valid) in enumerate(ep.inner_subject_partitions(tdata, n_splits=3)):
                print(f"S15 wrapper{wrapper['id']} policy{inner}: {len(fit)} train/{len(valid)} val", flush=True)
                ledger, scores, g = split(tdata, fit, valid, set(health) | protected,
                                          f"wrapper{wrapper['id']}_policy{inner}")
                pc.extend(ledger.candidates)
                pg.extend(ledger.groups)
                for a in ps:
                    ps[a].extend(scores[a])
                for name, node in g.items():
                    graph[f'p{inner}_{name}'] = {'fit_subjects': node['fit_subjects'],
                                                'parents': [f'p{inner}_{p}' for p in node['parents']]}
            if set(pg) != set(train):
                raise RuntimeError('policy OOF subject coverage mismatch')
            graph['policy'] = {'fit_subjects': train, 'parents': [f'p{i}_score' for i in range(3)]}
            ep.check_graph(graph, 'policy', set(health) | protected | source_val, set(train) | source_train)
            policies = {a: ep.choose_policy(pc, np.asarray(ps[a]), pg, tdata.train_truths, config) for a in ps}
            return {'wrapper': wrapper['id'], 'policies': policies, 'graph': graph}
        def health(wrapper, policies):
            budget()
            k = wrapper['id']
            print(f'S15 health wrapper{k}: all six policies frozen', flush=True)
            ledger, scores, graph = split(data, wrapper['train'], wrapper['health'], protected, f'wrapper{k}_health')
            hdata = ep.slice_outer_train(data, wrapper['health'])
            predictions = {a: ep.decode_policy(ledger.candidates, scores[a], ledger.groups, policies[a]) for a in scores}
            row = vm.make_record(wrapper, predictions, hdata.train_truths, hdata.subject_by_session, ledger.labels, scores)
            ep.write_json(output/f'wrapper{k}_report.json', row)
            completed.append(k)
            state_write('S15_IN_PROGRESS')
            print(f'S15 wrapper{k} health complete (not an interim selection)', flush=True)
            return row
        records, marker_sha = execute_phases(output, manifest, policy, health)
        validate_fit_counts(counts)
        summary = vm.summarize_repeats(records)
        ep.write_json(output/'complementarity.json', vm.complementarity(records))
        summary.update({'id': IDENTITY, 'outer_calls': 0, 'development_only': True,
                        'event_f1': None, 'fit_counts': counts, 'wall_seconds': time.perf_counter()-started,
                        'adaptation_and_encoding_seconds': work[0], 'policy_marker_sha256': marker_sha,
                        'persistent_token_cache_bytes': 0, 'copied_checkpoint_bytes': 0})
        if ep.protected_snapshot(root) != snapshot:
            raise RuntimeError('protected release changed')
        budget()
        ep.write_json(output/'summary.json', summary)
        state_write('S15_COMPLETE', protected_unchanged=True, resource_ok=True,
                    wall_seconds=time.perf_counter()-started, peak_cuda_allocated_bytes=(
                        torch.cuda.max_memory_allocated() if device == 'cuda' else None))
        print(json.dumps(summary), flush=True)
    except Exception as exc:
        state_write('S15_INCOMPLETE', error=str(exc), wall_seconds=time.perf_counter()-started)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('manifest', 'event'), required=True)
    args = parser.parse_args()
    with ep.threadpool_limits(limits=4):
        (manifest_stage if args.stage == 'manifest' else run_event)(Path(__file__).resolve().parents[3])
