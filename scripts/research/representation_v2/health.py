"""A1 R0: observational training health on a split strictly inside original T."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.special import expit, logit
import torch

from scripts.research.representation_v2.crossfit import (
    build_candidate_ledgers, inner_subject_partitions, slice_outer_train,
)
from scripts.research.representation_v2.dataset import load_candidate_bags
from scripts.research.representation_v2.evaluate import probe_metrics
from scripts.research.representation_v2.models import MotionBranch
from scripts.research.representation_v2.train import fit_motion_branch, predict_motion_logits, save_checkpoint
from src.pipeline.challenger.e03 import registered_e03_config
from src.pipeline.challenger.protection import protected_snapshot
from src.pipeline.runner import FilesystemDataSource


def validate_health_partition(original_train, original_validation, train, health):
    t, v, a, b = map(set, (original_train, original_validation, train, health))
    if not a or not b or a & b or (a | b) != t or t & v:
        raise ValueError('health split must partition original T and exclude original V')


def logit_diagnostics(labels, logits):
    y, z = np.asarray(labels), np.asarray(logits, dtype=np.float64)
    if y.shape != z.shape or not np.isfinite(z).all():
        raise ValueError('invalid health logits')
    p = expit(z)
    metrics = probe_metrics(y, p)
    # No candidate F1 headline: this is a training-health diagnostic only.
    metrics = {k: v for k, v in metrics.items() if 'f1' not in k and not k.endswith('_05')}
    return {**metrics, 'raw_logit_bce': float(np.mean(np.logaddexp(0, z) - y * z)),
            'positive_below_001': int(np.sum((y == 1) & (p < .01))),
            'negative_above_099': int(np.sum((y == 0) & (p > .99))),
            'exact_probability_endpoints': int(np.sum((p == 0) | (p == 1))),
            'logit_min': float(z.min()), 'logit_max': float(z.max())}


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False,
                                         allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run(root: Path):
    output = root / 'outputs/challenger_70/representation_v2/amendment_a1'
    if (output / 'plan.json').exists():
        raise RuntimeError('R0 is single attempt: existing plan cannot be overwritten or silently resumed')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip():
        raise RuntimeError('commit implementation and registration before R0')
    started = time.perf_counter()
    before = protected_snapshot(root)
    code_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    spec_sha = hashlib.sha256((root / 'docs/challenge_070_amendment_rv2_a1.md').read_bytes()).hexdigest()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    plan = {'experiment': 'RV2-A1-R0-001', 'code_sha': code_sha, 'spec_sha256': spec_sha,
            'outer_wrapper': 0, 'original_inner': 0, 'health_group_folds': 3, 'health_fold': 0,
            'steps': 2000, 'seed': 20260924, 'batch_size': 8, 'device': device,
            'schema': 'motion-input-rv2-a1', 'outer_calls': 0, 'release_changes': False,
            'purpose': 'TRAINING HEALTH ONLY; no epoch selection or full P2 gate',
            'max_wall_seconds': 10800, 'max_training_seconds': 7200,
            'protected_digest': hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()}
    _write(output / 'plan.json', plan)
    plan_sha = hashlib.sha256((output / 'plan.json').read_bytes()).hexdigest()
    _write(output / 'state.json', {'status': 'R0_IN_PROGRESS', 'outer_calls': 0})
    try:
        config = registered_e03_config(0, workers=4)
        data = FilesystemDataSource(root).load_outer_train_only(config)
        original_t, original_v = inner_subject_partitions(data, n_splits=4)[0]
        train_data = slice_outer_train(data, original_t)
        train, health = inner_subject_partitions(train_data, n_splits=3)[0]
        validate_health_partition(original_t, original_v, train, health)
        subjects = {key: list(map(str, value)) for key, value in
                    [('original_train', original_t), ('excluded_original_validation', original_v),
                     ('fit_subjects', train), ('health_subjects', health)]}
        _write(output / 'subjects.json', subjects)
        print('R0 subject ancestry validated; generating train-contained candidate producers', flush=True)
        t, h = build_candidate_ledgers(train_data, config, train, health)
        t_bags, h_bags = load_candidate_bags(root, t.candidates), load_candidate_bags(root, h.candidates)
        t_valid = np.array([bool(b.token_mask.any()) for b in t_bags])
        h_valid = np.array([bool(b.token_mask.any()) for b in h_bags])
        fit_bags = [b for b, keep in zip(t_bags, t_valid) if keep]
        health_bags = [b for b, keep in zip(h_bags, h_valid) if keep]
        if not fit_bags or not health_bags:
            raise RuntimeError('no valid train or health candidates')
        prevalence = float(t.labels.mean())
        if not 0 < prevalence < 1:
            raise RuntimeError('training prevalence cannot support finite fallback logits')
        rows = [{'sid': c.event.sid, 'start_ms': int(c.event.start_ms), 'end_ms': int(c.event.end_ms),
                 'subject': str(s), 'label': int(y), 'valid': bool(v)}
                for c, s, y, v in zip(h.candidates, h.groups, h.labels, h_valid)]
        _write(output / 'population.json', {'health_rows': rows, 'train_candidate_count': len(t_bags),
               'train_valid_count': int(t_valid.sum()), 'train_prevalence': prevalence,
               'micro_threshold_fit_on_train_only': float(t.micro_threshold),
               'candidate_producer_fits': 2 * (config.inner_splits + 1),
               'health_baseline': logit_diagnostics(h.labels, np.full(len(h_bags), logit(prevalence)))})
        torch.manual_seed(20260924)
        initial = MotionBranch()
        reports, training_seconds = {}, 0.
        print(f'R0 input ready: train={len(t_bags)}, health={len(h_bags)}, device={device}', flush=True)
        for arm, frozen in [('q0', True), ('q1', False)]:
            observations = []

            def observe(model, center, scale, step):
                if time.perf_counter() - started > 10800:
                    raise TimeoutError('R0 wall budget exhausted')
                z = np.full(len(h_bags), logit(prevalence))
                z[h_valid] = predict_motion_logits({'model': model, 'center': center, 'scale': scale},
                                                   health_bags, device=device)
                record = {'step': step, 'all': logit_diagnostics(h.labels, z),
                          'valid': logit_diagnostics(h.labels[h_valid], z[h_valid]), 'logits': z.tolist()}
                observations.append(record)
                _write(output / f'{arm}_observations.json', observations)
                print(f'{arm} step={step}: valid BCE={record["valid"]["raw_logit_bce"]:.4f}', flush=True)

            result = fit_motion_branch(fit_bags, t.labels[t_valid], t.groups[t_valid], steps=2000,
                                       seed=20260924, device=device, initial_model=deepcopy(initial),
                                       freeze_token_encoder=frozen, observation_callback=observe,
                                       max_wall_seconds=min(7200 - training_seconds, 10800 - (time.perf_counter() - started)))
            training_seconds += result['wall_seconds']
            save_checkpoint(output / f'{arm}_final.pt', result, provenance={
                **subjects, 'code_sha': code_sha, 'plan_sha256': plan_sha, 'arm': arm,
                'outer_calls': 0, 'source': 'target_only_original_T_health_split'})
            reports[arm] = {key: result[key] for key in
                            ('steps', 'training_history', 'convergence', 'convergence_summary', 'wall_seconds', 'sample_counts')}
            reports[arm]['final_health'] = observations[-1]
            reports[arm]['normalization'] = {'center': result['center'].tolist(), 'scale': result['scale'].tolist()}
            _write(output / f'{arm}_training.json', reports[arm])
        unchanged = protected_snapshot(root) == before
        if not unchanged:
            raise RuntimeError('protected release snapshot changed')
        summary = {'status': 'R0_COMPLETE_REVIEW_REQUIRED', 'outer_calls': 0,
                   'protected_unchanged': unchanged, 'code_sha': code_sha, 'plan_sha256': plan_sha,
                   'wall_seconds': time.perf_counter() - started, 'training_seconds': training_seconds,
                   'optimizer_steps': 4000, 'persistent_token_cache_bytes': 0,
                   'arms': {key: {name: value[name] for name in ('convergence', 'convergence_summary', 'wall_seconds')} |
                            {'final_valid': value['final_health']['valid']} for key, value in reports.items()}}
        _write(output / 'summary.json', summary)
        _write(output / 'state.json', {'status': summary['status'], 'outer_calls': 0})
        print(json.dumps(summary, sort_keys=True), flush=True)
        return summary
    except Exception as exc:
        _write(output / 'state.json', {'status': 'R0_FAILED_REVIEW_REQUIRED', 'outer_calls': 0,
               'error_type': type(exc).__name__, 'error': str(exc), 'wall_seconds': time.perf_counter() - started})
        raise


if __name__ == '__main__':
    run(Path(__file__).resolve().parents[3])
