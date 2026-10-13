"""Registered partition-contained three-seed experiment, never a V7 entrypoint."""
import argparse
from dataclasses import asdict
import gc
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .. import event_probe as ep
from ..crossfit import build_candidate_ledgers
from .cache import encoder_hash, file_hash
from .checkpoint_audit import profile, read_tokens, score_checkpoint
from .contract import SOURCE_SHA, cache_key, digest
from .probe import (Resources, load_registration, phase, read_json, require_clean,
                    require_committed, write_json, freeze_policies, read_frozen_policies,
                    verify_registered_files)
from .reporting import make_record, summarize
from .timeline import candidate_chunks
from .training import fit_head, fit_moments, predict_head, fuse_scores
from .v7_models import SubsetSource

T1 = 'outputs/challenger_70/representation_v2/amendment_a6_transformer/t1'
OUTPUT = 'outputs/challenger_70/representation_v2/amendment_a6_transformer/seed_ensemble'
SPEC = 'docs/challenge_070_amendment_a6_seed_ensemble.md'
EXPECTED = {'producer_fits': 120, 'base_fits': 24, 'neural_fits': 48, 'steps': 48000}


def cache_snapshot(directory):
    directory = Path(directory)
    headers = {p.stem for p in directory.glob('*.json')}
    arrays = {p.stem for p in directory.glob('*.npz')}
    if not headers or headers != arrays:
        raise ValueError('empty or partial source token cache')
    return {f'{stem}{suffix}': file_hash(directory/f'{stem}{suffix}')
            for stem in sorted(headers) for suffix in ('.json', '.npz')}


def policy_coverage(partitions, expected):
    seen = set()
    if len(partitions) != 3:
        raise ValueError('three OOF partitions required')
    for values in partitions:
        subjects = set(values)
        if not subjects or subjects & seen or subjects-set(expected):
            raise ValueError('policy OOF overlap or unknown subject')
        seen.update(subjects)
    if seen != set(expected):
        raise ValueError('policy OOF subject coverage differs')


def wrapper_directions(records, originals):
    if len(records) != 3 or len(originals) != 3:
        raise ValueError('three paired wrappers required')
    return {a: [r['events'][a]['f1'] >= o['events'][a]['f1']-1e-12
                for r, o in zip(records, originals)] for a in ('c', 't')}


def seeds_for(split_id):
    if not isinstance(split_id, int) or not 0 <= split_id < 12:
        raise ValueError('unknown registered partition')
    return tuple(20261012 + split_id + offset for offset in (0, 100000, 200000))


def average_three(values):
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('unaligned ensemble') from exc
    if (array.ndim != 2 or array.shape[0] != 3 or not array.shape[1]
            or not np.isfinite(array).all() or np.any((array < 0) | (array > 1))):
        raise ValueError('three aligned probability members required')
    return array.mean(axis=0)


def verify_reproduction(reference, candidates, labels, groups, base):
    if (digest(reference['candidates']) != digest(candidates)
            or not np.array_equal(reference['labels'], labels)
            or not np.array_equal(reference['groups'], groups)
            or np.asarray(base).shape != np.asarray(reference['scores']['b']).shape
            or not np.allclose(base, reference['scores']['b'], atol=1e-6, rtol=1e-5)):
        raise ValueError('original candidate/base evidence not reproduced')


def ensemble_gate(base_f1, single_f1, ensemble_f1, nonworse):
    if (len(nonworse) != 3 or not np.isfinite([base_f1, single_f1, ensemble_f1]).all()
            or any(not 0 <= v <= 1 for v in (base_f1, single_f1, ensemble_f1))):
        raise ValueError('invalid primary gate inputs')
    return {'primary': 'c', 'passed': bool(ensemble_f1-base_f1 >= .01-1e-12
        and ensemble_f1 >= single_f1-1e-12 and sum(nonworse) >= 2),
        'delta_vs_incumbent': ensemble_f1-base_f1, 'delta_vs_single': ensemble_f1-single_f1,
        'nonworse_wrappers_vs_single': sum(nonworse), 'outer_authorized': False}


class SeedResources(Resources):
    def check(self, *, active_neural_seconds=0., active_encoding_seconds=0.):
        super().check(active_neural_seconds=active_neural_seconds, active_encoding_seconds=active_encoding_seconds)
        if (time.time()-self.row['started_epoch'] > 4*3600
                or self.row['neural_seconds']+active_neural_seconds > 2*3600
                or self.row['cpu_seconds'] > 3600
                or self.row['encoding_seconds']+active_encoding_seconds > 3600):
            raise RuntimeError('seed ensemble resource budget exceeded')


class FrozenSequences:
    """Read source-only T1 tokens; never write to or rebuild the old cache."""
    def __init__(self, root, manifest):
        self.root, self.manifest = Path(root), manifest
        self.raw = read_json(self.root/T1/'raw_checksums.json')
        self.index, self.loaded, self.hashes = {}, {}, {}
        source = ep.load_source_artifact(self.root/'outputs/challenger_70/representation_v2/amendment_a3_source', SOURCE_SHA)
        encoder = ep.MotionBranch().token_encoder
        encoder.load_state_dict(source['encoder'], strict=True)
        expected_encoder = encoder_hash(encoder)
        for header in sorted((self.root/T1/'tokens').glob('*.json')):
            meta = read_json(header)['metadata']; sid = meta['sid']
            if (sid in self.index or sid not in manifest['subject_by_session']
                    or header.stem != cache_key(meta) or meta['encoder_sha256'] != expected_encoder
                    or meta['fit_subjects'] != manifest['identities']['source_train']
                    or meta['normalization'] != 'unitfree'
                    or meta['preprocessing'] != {'sampling_hz': 25, 'token_ms': 5000,
                        'max_gap_ms': 200, 'min_coverage': .8, 'gravity_align': False}):
                raise ValueError('T1 token identity differs')
            self.index[sid] = header

    def sequences(self, ledger, resources):
        started = time.perf_counter(); result = []
        allowed = {s for s, p in self.manifest['subject_by_session'].items() if p in ledger.score_subjects}
        for candidate, group in zip(ledger.candidates, ledger.groups):
            resources.check(active_encoding_seconds=time.perf_counter()-started)
            e = candidate.event
            if e.sid not in allowed or self.manifest['subject_by_session'][e.sid] != group:
                raise ValueError('sequence partition ownership differs')
            if e.sid not in self.loaded:
                header = self.index[e.sid]  # Missing cache fails, no raw fallback.
                self.loaded[e.sid] = read_tokens(header, allowed_sids=allowed, raw_hashes=self.raw, source_sha=SOURCE_SHA)
                for path in (header, header.with_suffix('.npz')):
                    self.hashes[path.relative_to(self.root).as_posix()] = file_hash(path)
            tokens = self.loaded[e.sid]; chunks = []
            for c in candidate_chunks(tokens, e.start_ms, e.end_ms):
                ix = c.token_indices
                chunks.append({'region': c.region, 'indices': ix, 'embeddings': tokens.embeddings[ix],
                               'coverage': tokens.coverage[ix], 'times': tokens.center_ms[ix], 'valid': tokens.valid[ix]})
            result.append({'sid': e.sid, 'subject': str(group), 'start_ms': e.start_ms, 'chunks': chunks})
        resources.row['encoding_seconds'] += time.perf_counter()-started
        return result

    def verify(self):
        if any(file_hash(self.root/name) != sha for name, sha in self.hashes.items()):
            raise ValueError('source token cache changed during training')


def registered(root):
    output = Path(root)/OUTPUT; registry = read_json(output/'registry.json')
    verify_registered_files(root, registry)
    if cache_snapshot(Path(root)/T1/'tokens') != registry['token_cache']:
        raise ValueError('registered source token inventory changed')
    if digest(ep.protected_snapshot(root)) != registry['protected_digest']:
        raise ValueError('protected release changed')
    return read_json(output/'manifest.json')


def run_register(root):
    root = Path(root); output = root/OUTPUT
    if output.exists():
        raise RuntimeError('attempt exists; no overwrite')
    require_clean(root); manifest = load_registration(root)
    files = subprocess.check_output(['git', 'ls-files', 'scripts', 'src'], cwd=root, text=True).splitlines()
    names = [n for n in files if n.endswith('.py')] + [SPEC, f'{T1}/manifest.json',
             f'{T1}/raw_checksums.json', f'{T1}/summary.json']
    for k in range(3):
        names += [f'{T1}/wrapper{k}_policies.json', f'{T1}/wrapper{k}_health.json']
        for part in [f'policy{j}' for j in range(3)]+['health']:
            prefix = f'wrapper{k}_{part}'
            names += [f'{T1}/{prefix}_scores.json']+[f'{T1}/{prefix}_{a}.pt' for a in ('c', 't')]
    for name in names:
        require_committed(root, root/name)
    features = [f'cache/{folder}/fold0_{role}.npz' for folder in ('slide', 'micro15')
                for role in ('train', 'meal_train', 'no_meal_train')]
    token_cache = cache_snapshot(root/T1/'tokens')
    output.mkdir(parents=True)
    write_json(output/'manifest.json', manifest)
    names += features+[f'{OUTPUT}/manifest.json']+[f'{T1}/tokens/{name}' for name in token_cache]
    write_json(output/'registry.json', {'id': 'RV2-A6-SEED-001', 'expected_fits': EXPECTED,
        'token_cache': token_cache,
        'code_files': {n: file_hash(root/n) for n in names}, 'protected_digest': digest(ep.protected_snapshot(root)),
        'code_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()})
    write_json(output/'state.json', {'status': 'REGISTERED', 'outer_calls': 0})


def runtime(root, manifest):
    config = ep.registered_e03_config(0, workers=4)
    data = SubsetSource(root, set(manifest['subject_by_session'])).load_outer_train_only(config)
    if data.subject_by_session != manifest['subject_by_session'] or digest(asdict(config)) != digest(manifest['config']):
        raise ValueError('T24 runtime identity/config differs')
    return data, config, FrozenSequences(root, manifest)


def split(root, data, config, train, score, excluded, manifest, cache, resources, prefix, split_id):
    output = Path(root)/OUTPUT; reference = read_json(Path(root)/T1/f'{prefix}_scores.json')
    if set(reference['train']) != set(train) or set(reference['score']) != set(score) or set(train)&set(excluded):
        raise ValueError('reference training partition differs')
    ep.check_graph(reference['graph'], 'score', set(score)|set(excluded)|set(manifest['identities']['source_val']),
                   set(train)|set(manifest['identities']['source_train']))
    graph = {}; started = time.perf_counter()
    training, valid = build_candidate_ledgers(data, config, train, score, fit_audit=graph)
    base = ep.verifier_probabilities(training.features, training.labels, valid.features, seed=config.seed)
    resources.row['cpu_seconds'] += time.perf_counter()-started
    resources.row['producer_fits'] += 10; resources.row['base_fits'] += 2
    verify_reproduction(reference, [asdict(c) for c in valid.candidates], valid.labels, valid.groups, base)
    training_sequences = cache.sequences(training, resources); validation_sequences = cache.sequences(valid, resources)
    mask = np.asarray([any(c['valid'].any() for c in s['chunks']) for s in validation_sequences])
    if not np.array_equal(mask, reference['motion_valid']):
        raise ValueError('validity differs')
    center, scale = fit_moments(training_sequences, set(training.groups))
    scores = {'b': np.asarray(reference['scores']['b']), 'p': np.asarray(reference['scores']['p'])}
    components, histories, diagnostics = {}, {}, {}
    graph['source'] = {'fit_subjects': manifest['identities']['source_train'], 'parents': []}
    graph['base'] = {'fit_subjects': list(train), 'parents': ['candidates']}
    paired = {}
    for arm in ('c', 't'):
        checkpoint = Path(root)/T1/f'{prefix}_{arm}.pt'
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if set(payload['provenance']['fit_subjects']) != set(train):
            raise ValueError('checkpoint training subjects differ')
        if (not np.allclose(center, payload['center'].numpy(), atol=1e-6, rtol=1e-5)
                or not np.allclose(scale, payload['scale'].numpy(), atol=1e-6, rtol=1e-5)):
            raise ValueError('training normalization differs')
        motion, _ = score_checkpoint(checkpoint, arm, validation_sequences, base, mask, resources.device)
        if not np.allclose(motion, reference['components'][arm], atol=1e-6, rtol=1e-5):
            raise ValueError('original checkpoint not reproduced')
        members = [np.asarray(reference['components'][arm])]; history = []
        for seed in seeds_for(split_id)[1:]:
            print(f'ENSEMBLE {prefix} {arm} seed={seed} train={len(training.labels)}', flush=True)
            started = time.perf_counter()
            fitted = fit_head(arm, training_sequences, training.labels, training.groups, seed=seed, device=resources.device,
                resource_callback=lambda: resources.check(active_neural_seconds=time.perf_counter()-started))
            members.append(predict_head(fitted, validation_sequences,
                resource_callback=lambda: resources.check(active_neural_seconds=time.perf_counter()-started)))
            info = {'seed': seed, **{k: v for k, v in fitted.items() if k not in {'model', 'center', 'scale'}}}
            if seed in paired and paired[seed] != info['sampling_sha256']:
                raise ValueError('paired arm sampling differs')
            paired[seed] = info['sampling_sha256']; history.append({'seed': seed, **info})
            torch.save({'state_dict': {k: v.cpu() for k, v in fitted['model'].state_dict().items()},
                        'center': torch.tensor(fitted['center']), 'scale': torch.tensor(fitted['scale']),
                        'provenance': info}, output/f'{prefix}_{arm}_{seed}.pt')
            resources.row['neural_fits'] += 1; resources.row['steps'] += fitted['steps']
            resources.row['neural_seconds'] += time.perf_counter()-started
            resources.row['fits'].append({'split': prefix, 'arm': arm, 'seed': seed, **info})
            resources.check(); resources.save(); del fitted; gc.collect()
        components[arm] = [m.tolist() for m in members]
        scores[arm] = fuse_scores(base, average_three(members), mask)
        diagnostics[arm] = {
            'valid_mean_seed_std': float(np.asarray(members)[:, mask].std(0).mean()) if mask.any() else None,
            'member_profiles': [profile(valid.labels, fuse_scores(base, m, mask), m, mask) for m in members],
            'ensemble_profile': profile(valid.labels, scores[arm], average_three(members), mask)}
        histories[arm] = history
        graph[arm] = {'fit_subjects': list(train), 'parents': ['source', 'candidates']}
    graph['score'] = {'fit_subjects': [], 'parents': ['base', 'c', 't']}
    ep.check_graph(graph, 'score', set(score)|set(excluded)|set(manifest['identities']['source_val']),
                   set(train)|set(manifest['identities']['source_train']))
    write_json(output/f'{prefix}_scores.json', {'train': list(train), 'score': list(score), 'graph': graph,
        'candidates': [asdict(c) for c in valid.candidates], 'groups': valid.groups.tolist(),
        'labels': valid.labels.tolist(), 'scores': {a: s.tolist() for a, s in scores.items()},
        'motion_valid': mask.tolist(), 'members': components, 'histories': histories, 'diagnostics': diagnostics,
        'seeds': list(seeds_for(split_id)), 'outer_calls': 0})
    return valid, scores


def run_policy(root):
    root = Path(root); output = root/OUTPUT; require_clean(root)
    require_committed(root, output/'registry.json'); manifest = registered(root)
    with phase(output, 'REGISTERED', 'POLICIES_FROZEN'):
        resources = SeedResources(root, output, new=True); resources.save()
        data, config, cache = runtime(root, manifest)
        protected = set(manifest['identities']['v7'])|set(manifest['identities']['outer'])
        with threadpool_limits(limits=4):
            for wrapper in manifest['wrappers']:
                k = wrapper['id']; candidates = []; groups = []; scores = {a: [] for a in ('c', 't')}
                policy_coverage([part['score'] for part in wrapper['policy_splits']], wrapper['train'])
                partition = ep.slice_outer_train(data, wrapper['train'])
                for j, part in enumerate(wrapper['policy_splits']):
                    ledger, values = split(root, partition, config, part['train'], part['score'], protected|set(wrapper['health']),
                        manifest, cache, resources, f'wrapper{k}_policy{j}', k*4+j)
                    candidates.extend(ledger.candidates); groups.extend(ledger.groups)
                    for arm in scores:
                        scores[arm].extend(values[arm])
                if set(groups) != set(wrapper['train']):
                    raise ValueError('policy candidate subject coverage differs')
                original = read_json(root/T1/f'wrapper{k}_policies.json')['policies']
                policies = {a: original[a] for a in ('b', 'p')}
                policies.update({a: ep.choose_policy(candidates, np.asarray(scores[a]), groups, partition.train_truths, config)
                                 for a in ('c', 't')})
                write_json(output/f'wrapper{k}_policies.json', {'wrapper': k, 'policies': policies, 'outer_calls': 0})
        cache.verify(); freeze_policies(output); registered(root); resources.save()


def run_health(root):
    root = Path(root); output = root/OUTPUT
    policies = read_frozen_policies(output); require_clean(root)
    require_committed(root, output/'policy_phase_frozen.json'); manifest = registered(root)
    with phase(output, 'POLICIES_FROZEN', 'COMPLETE'):
        resources = SeedResources(root, output); data, config, cache = runtime(root, manifest)
        protected = set(manifest['identities']['v7'])|set(manifest['identities']['outer']); records = []
        with threadpool_limits(limits=4):
            for wrapper in manifest['wrappers']:
                k = wrapper['id']; read_frozen_policies(output)
                ledger, scores = split(root, data, config, wrapper['train'], wrapper['health'], protected,
                    manifest, cache, resources, f'wrapper{k}_health', k*4+3)
                predictions = {a: ep.decode_policy(ledger.candidates, p, ledger.groups, policies[k][a]) for a, p in scores.items()}
                health = ep.slice_outer_train(data, wrapper['health'])
                record = make_record(k, predictions, health.train_truths, health.subject_by_session, ledger.candidates, ledger.labels, scores)
                records.append(record); write_json(output/f'wrapper{k}_health.json', record)
        cache.verify(); registered(root); resources.check(); resources.save()
        if {k: resources.row[k] for k in EXPECTED} != EXPECTED:
            raise ValueError('actual fit budget differs')
        result = summarize(records)
        singles = [read_json(root/T1/f'wrapper{k}_health.json') for k in range(3)]
        original = read_json(root/T1/'summary.json')
        directions = wrapper_directions(records, singles)
        result['wrapper_directions_vs_single'] = directions
        result['gate'] = ensemble_gate(result['events']['b']['f1'], original['events']['c']['f1'], result['events']['c']['f1'], directions['c'])
        result['secondary'] = {'arm': 't', 'nonworse_wrappers_vs_single': sum(directions['t']),
            'delta_vs_single': result['events']['t']['f1']-original['events']['t']['f1'],
            'delta_vs_incumbent': result['events']['t']['f1']-result['events']['b']['f1']}
        result['health_seed_diagnostics'] = {str(k): read_json(output/f'wrapper{k}_health_scores.json')['diagnostics'] for k in range(3)}
        result['single_model_events'] = original['events']; result['resources'] = resources.row
        write_json(output/'records.json', records); write_json(output/'summary.json', result)
        lines = ['# A6 fixed three-seed ensemble', '', 'T24 nested development evidence only. No V7/outer/dist/release change.', '',
                 '| Arm | F1 | TP | FP | FN |', '| --- | ---: | ---: | ---: | ---: |']
        for arm, values in result['events'].items():
            lines.append(f"| {arm} | {values['f1']:.6f} | {values['n_tp']} | {values['fp']} | {values['fn']} |")
        for arm in ('c', 't'):
            values = original['events'][arm]
            lines.append(f"| single_{arm} | {values['f1']:.6f} | {values['n_tp']} | {values['fp']} | {values['fn']} |")
        lines += ['', f"Primary TCN gate: {result['gate']}", '',
                  f"Secondary Transformer: {result['secondary']}",
                  f"Wrapper directions versus single: {directions}", '',
                  'b/p are unchanged original controls; c/t are within-partition three-seed probability means.',
                  'This does not isolate train-size/composition effects or establish pristine generalization.']
        lines += ['', '## Health candidate / seed diagnostics', '',
                  '| Wrapper | Arm | AP | BCE | Valid mean seed std | True motion mean | Negative motion mean |',
                  '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
        for k in range(3):
            for arm in ('c', 't'):
                diag = result['health_seed_diagnostics'][str(k)][arm]; p = diag['ensemble_profile']
                lines.append(f"| {k} | {arm} | {p['candidate_ap']:.6f} | {p['candidate_bce']:.6f} | "
                    f"{diag['valid_mean_seed_std']:.6f} | {p['positive_motion']['mean']:.6f} | {p['negative_motion']['mean']:.6f} |")
        (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('register', 'policy', 'health'), required=True)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    globals()['run_'+args.stage](args.root.resolve())


if __name__ == '__main__':
    main()
