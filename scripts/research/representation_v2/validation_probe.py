"""A5: partition-expanded (not population-expanded) inner development."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

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
