"""Diagnostic-only paired checkpoint inference: no training, policy search or F1."""
import argparse
from dataclasses import asdict
from pathlib import Path
import subprocess
import time

import numpy as np
from sklearn.metrics import average_precision_score, log_loss
import torch

from .. import event_probe as ep
from ..stability_audit import restore_candidate
from .cache import encoder_hash, file_hash
from .contract import SOURCE_SHA, cache_key, digest
from .heads import build_head
from .probe import read_json, require_clean, require_committed, write_json
from .timeline import SessionTokens, candidate_chunks
from .training import predict_head, fuse_scores
from .v7_contract import OUTPUT as V7, validate_oof, verify_seal
from .v7_probe import FROZEN_FILES, T1, _registered

OUTPUT = 'outputs/challenger_70/representation_v2/amendment_a6_transformer/checkpoint_audit'


def read_tokens(header, *, allowed_sids, raw_hashes, source_sha):
    """Read verified cached arrays only; never call raw loading or encode fallback."""
    header = Path(header)
    saved = read_json(header)
    metadata = saved['metadata']
    sid = metadata['sid']
    if (sid not in allowed_sids or sid not in raw_hashes
            or metadata.get('raw_sha256') != raw_hashes[sid]
            or metadata.get('source_checkpoint') != source_sha):
        raise ValueError('unauthorized cache identity')
    path = header.with_suffix('.npz')
    if file_hash(path) != saved['sha256']:
        raise ValueError('token checksum mismatch')
    with np.load(path, allow_pickle=False) as data:
        return SessionTokens(sid, *(data[k] for k in
            ('center_ms', 'span_id', 'coverage', 'valid', 'embeddings')),
            int(data['start_ms']), int(data['end_ms']))


def score_checkpoint(path, arm, sequences, base, valid, device):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    center, scale = payload['center'].numpy(), payload['scale'].numpy()
    if (center.shape != (16,) or scale.shape != (16,) or not np.isfinite(center).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError('invalid checkpoint normalization')
    model = build_head(arm).to(device)
    model.load_state_dict(payload['state_dict'], strict=True)
    fitted = {'model': model.eval(), 'center': center, 'scale': scale, 'device': device}
    motion = predict_head(fitted, sequences)
    return motion, fuse_scores(base, motion, valid)


def profile(labels, fused, motion, valid):
    y, p, m, v = (np.asarray(x) for x in (labels, fused, motion, valid))
    if (y.ndim != 1 or not len(y) or any(x.shape != y.shape for x in (p, m, v))
            or not np.isin(y, [0, 1]).all() or not np.isin(v, [False, True]).all()
            or not np.isfinite(p).all() or not np.isfinite(m).all()
            or np.any((p < 0) | (p > 1) | (m < 0) | (m > 1))):
        raise ValueError('invalid aligned diagnostic scores')
    row = {'n': len(y), 'positive_count': int(y.sum()), 'invalid_count': int((~v.astype(bool)).sum()),
           'candidate_ap': float(average_precision_score(y, p)) if len(set(y)) == 2 else None,
           'candidate_bce': float(log_loss(y, p, labels=[0, 1])),
           'candidate_brier': float(np.mean((p-y)**2))}
    for name, label in [('positive_motion', 1), ('negative_motion', 0)]:
        values = m[(y == label) & v.astype(bool)]
        row[name] = {'n': len(values), 'mean': float(values.mean()) if len(values) else None,
                     'median': float(np.median(values)) if len(values) else None,
                     'p90': float(np.quantile(values, .9)) if len(values) else None}
    return row


def run(root, device):
    root = Path(root)
    output = root / OUTPUT
    if output.exists():
        raise RuntimeError('diagnostic attempt already exists; no overwrite')
    require_clean(root)
    marker = verify_seal(root / V7)
    if set(marker['files']) != set(FROZEN_FILES):
        raise ValueError('incomplete final-model seal')
    manifest = _registered(root)
    if read_json(root / V7 / 'state.json')['status'] != 'COMPLETE':
        raise ValueError('completed V7 evidence required')
    ids = manifest['identities']
    rows = [read_json(root / T1 / f'wrapper{k}_health_scores.json') for k in range(3)]
    validate_oof(rows, set(ids['t24']), set(ids['source_train']),
                 set(ids['v7']) | set(ids['outer']) | set(ids['source_val']))
    checkpoints = {f'{name}_{arm}': (root / folder / filename, arm)
        for name, folder, template in [('refit', V7, '{arm}.pt')]
        + [(f'wrapper{k}', T1, f'wrapper{k}_health_{{arm}}.pt') for k in range(3)]
        for arm in ('c', 't') for filename in [template.format(arm=arm)]}
    names = [root / V7 / n for n in ('v7_scores.json', 'summary.json', 'policies.json', 'frozen.json')]
    names += [root / T1 / f'wrapper{k}_health_scores.json' for k in range(3)]
    names += [p for p, _ in checkpoints.values()]
    names += [Path(__file__).resolve()]
    for path in names:
        require_committed(root, path)
    hashes = {p.relative_to(root).as_posix(): file_hash(p) for p in names}
    protected = digest(ep.protected_snapshot(root))
    evidence = read_json(root / V7 / 'v7_scores.json')
    summary = read_json(root / V7 / 'summary.json')
    if file_hash(root / V7 / 'v7_scores.json') != summary['score_sha256']:
        raise ValueError('score evidence checksum mismatch')
    candidates = [restore_candidate(c) for c in evidence['candidates']]
    groups = np.asarray(evidence['groups'])
    if len(groups) != len(candidates) or set(groups) != set(ids['v7']):
        raise ValueError('V7 candidate coverage differs')
    allowed = {s for s, p in manifest['subject_by_session'].items() if p in ids['v7']}
    if any(c.event.sid not in allowed or manifest['subject_by_session'][c.event.sid] != g
           for c, g in zip(candidates, groups)):
        raise ValueError('V7 candidate ownership differs')
    # Source-only encoder identity check; no encoding/gradient/target fitting.
    source = ep.load_source_artifact(root / 'outputs/challenger_70/representation_v2/amendment_a3_source', SOURCE_SHA)
    encoder = ep.MotionBranch().token_encoder
    encoder.load_state_dict(source['encoder'], strict=True)
    expected_encoder = encoder_hash(encoder)
    raw_hashes = read_json(root / V7 / 'raw_checksums.json')
    tokens = {}; cache_hashes = {}
    needed = {c.event.sid for c in candidates}
    for header in sorted((root / V7 / 'tokens').glob('*.json')):
        meta = read_json(header)['metadata']
        sid = meta['sid']
        if sid not in needed:
            continue
        if (sid in tokens or header.stem != cache_key(meta)
                or meta['encoder_sha256'] != expected_encoder
                or meta['fit_subjects'] != ids['source_train']
                or meta['normalization'] != 'unitfree'
                or meta['preprocessing'] != {'sampling_hz': 25, 'token_ms': 5000,
                    'max_gap_ms': 200, 'min_coverage': .8, 'gravity_align': False}):
            raise ValueError('cache preprocessing/encoder identity differs')
        tokens[sid] = read_tokens(header, allowed_sids=allowed, raw_hashes=raw_hashes, source_sha=SOURCE_SHA)
        cache_hashes[header.relative_to(root).as_posix()] = file_hash(header)
        cache_hashes[header.with_suffix('.npz').relative_to(root).as_posix()] = file_hash(header.with_suffix('.npz'))
    if set(tokens) != needed:
        raise ValueError('missing frozen tokens; no encoding fallback permitted')
    sequences = []
    for c, g in zip(candidates, groups):
        e = c.event; t = tokens[e.sid]; chunks = []
        for chunk in candidate_chunks(t, e.start_ms, e.end_ms):
            ix = chunk.token_indices
            chunks.append({'region': chunk.region, 'indices': ix, 'embeddings': t.embeddings[ix],
                           'coverage': t.coverage[ix], 'times': t.center_ms[ix], 'valid': t.valid[ix]})
        sequences.append({'sid': e.sid, 'subject': str(g), 'start_ms': e.start_ms, 'chunks': chunks})
    valid = np.asarray([any(c['valid'].any() for c in s['chunks']) for s in sequences])
    if not np.array_equal(valid, evidence['motion_valid']):
        raise ValueError('candidate validity reconstruction differs')
    torch.set_num_threads(4)
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output.mkdir(parents=True)
    write_json(output / 'state.json', {'status': 'RUNNING', 'diagnostic_only': True})
    started = time.perf_counter()
    try:
        write_json(output / 'manifest.json', {'diagnostic_only': True, 'uses_gt': True,
            'training_fits': 0, 'threshold_searches': 0, 'outer_calls': 0,
            'code_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
            'inputs': hashes, 'cache_inputs': cache_hashes, 'protected_digest': protected,
            'comparison': 'same V7 candidates, base scores, tokens; own train-only normalization per head',
            'interpretation_limit': 'seed, train size and composition differ; no causal isolation of those factors'})
        base = np.asarray(evidence['scores']['b']); predictions = {}; parity = {}
        for name, (path, arm) in checkpoints.items():
            print(f'frozen diagnostic {name}: {len(candidates)} candidates', flush=True)
            motion, fused = score_checkpoint(path, arm, sequences, base, valid, device)
            predictions[name] = {'motion': motion.tolist(), 'fused': fused.tolist()}
            if name.startswith('refit'):
                delta = float(np.max(np.abs(fused - np.asarray(evidence['scores'][arm]))))
                parity[name] = delta
                if not np.allclose(fused, evidence['scores'][arm], atol=1e-6, rtol=1e-5):
                    raise ValueError('refit scores do not reproduce frozen evidence')
        # Persist scores before GT summaries. Never decode or optimize events here.
        write_json(output / 'scores.json', {'candidates': evidence['candidates'], 'groups': groups.tolist(),
            'motion_valid': valid.tolist(), 'base': base.tolist(), 'checkpoints': predictions})
        truths = [ep.runner.EventRef(**t) for t in summary['truths']]
        labels = ep.runner._candidate_labels(candidates, truths)
        profiles = {name: profile(labels, row['fused'], row['motion'], valid)
                    for name, row in predictions.items()}
        per_subject = {name: {g: profile(labels[groups == g], np.asarray(row['fused'])[groups == g],
            np.asarray(row['motion'])[groups == g], valid[groups == g]) for g in sorted(set(groups))}
            for name, row in predictions.items()}
        previous = {}
        for arm in ('c', 't'):
            previous[arm] = profile(np.concatenate([r['labels'] for r in rows]),
                np.concatenate([r['scores'][arm] for r in rows]),
                np.concatenate([r['components'][arm] for r in rows]),
                np.concatenate([r['motion_valid'] for r in rows]))
        for name, sha in {**hashes, **cache_hashes}.items():
            if file_hash(root / name) != sha:
                raise ValueError('diagnostic input mutated: ' + name)
        verify_seal(root / V7)
        if digest(ep.protected_snapshot(root)) != protected:
            raise ValueError('protected release changed')
        result = {'diagnostic_only': True, 'uses_gt': True, 'profiles': profiles,
                  'per_subject': per_subject, 't24_historical_oof': previous,
                  'refit_max_abs_fused_error': parity, 'training_fits': 0, 'outer_calls': 0,
                  'device': device, 'wall_seconds': time.perf_counter() - started,
                  'candidate_count': len(candidates), 'score_sha256': file_hash(output / 'scores.json')}
        write_json(output / 'summary.json', result)
        lines = ['# A6 Frozen checkpoint attribution audit', '',
            'DIAGNOSTIC ONLY / USES GT. No training, threshold search, event decoding, new F1 or release change.', '',
            'All rows use the same 223 V7 candidates/base scores/tokens. Own train-only normalization per checkpoint.', '',
            '| Checkpoint | True motion mean | True motion median | Negative motion mean | Fused AP | Fused BCE |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
        for name, row in profiles.items():
            lines.append(f"| {name} | {row['positive_motion']['mean']:.6f} | {row['positive_motion']['median']:.6f} | "
                         f"{row['negative_motion']['mean']:.6f} | {row['candidate_ap']:.6f} | {row['candidate_bce']:.6f} |")
        lines += ['', 'Comparing checkpoints controls evaluation population and input, but not training seed, size or composition.',
                  'T24 reference uses different subjects, candidates and fold-specific fitted models; it is descriptive, not paired.',
                  'V7 must not select checkpoints, fusion weights, thresholds or rescue rules. Future changes require train-only OOF validation.']
        (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        write_json(output / 'state.json', {'status': 'COMPLETE', 'diagnostic_only': True, 'training_fits': 0})
    except BaseException as exc:
        write_json(output / 'state.json', {'status': 'INCOMPLETE', 'error': repr(exc), 'diagnostic_only': True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args()
    run(args.root.resolve(), args.device)


if __name__ == '__main__':
    main()
