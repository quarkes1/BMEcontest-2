"""Single registered cap6 -> cap8 replay against the official-best policy."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from .consistent_validation import aggregate
from .dense_mil_probe import _score_file
from .event_probe import check_graph, complete_metrics, partition_truths
from .geometry_probe import load_context, inputs, T24_SOURCE, FIVE_SOURCE
from .negative_clean_probe import _read, _write, protected_sources
from .relative_score_probe import decode_relative, gate as r3_gate
from .scope_audit import _sha
from src.pipeline.artifacts import load_event_stack_bundle
from src.pipeline.runner import peak_working_set_bytes

SPEC = 'docs/official_recovery_cap8_v4.md'
MAIN = 'official_deployment_f1_policy_v3'
REFERENCES = dict(t24='outputs/official_recovery/deployment_f1_policy_v3_t24_attempt2',
                  five='outputs/official_recovery/deployment_f1_policy_v3_five_attempt2')
OUTPUTS = {k: f'outputs/official_recovery/cap8_policy_v4_{k}' for k in REFERENCES}
USER_PATHS = ('dist/official_trials/deployment_f1_policy_v3/README.md',
              'dist/official_trials/submission.zip')


def controlled_revision(root):
    root = Path(root)
    status = subprocess.check_output(['git', 'status', '--porcelain', '-z'], cwd=root).decode('utf-8')
    dirty = [row[3:] for row in status.split('\0') if row]
    if any(name not in USER_PATHS for name in dirty):
        raise RuntimeError('uncommitted changes outside the two existing user edits')
    edits = {name: _sha(root/name) if (root/name).is_file() else None for name in USER_PATHS}
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(), edits


def protected_baseline(root):
    root = Path(root)
    folder = root/'models/event_stack'/MAIN/'deployment'
    exported = _read(root/'outputs/official_recovery/deployment_f1_policy_v3_export/summary.json')
    built = _read(root/'outputs/official_recovery/deployment_f1_policy_v3_exe/summary.json')
    accepted = _read(root/'outputs/official_recovery/deployment_f1_policy_v3_acceptance/summary.json')
    actual = {p.name: _sha(p) for p in folder.iterdir() if p.is_file()}
    exe = root/'dist/official_trials/deployment_f1_policy_v3/EatingDetection.exe'
    if (exported['state'] != 'COMPLETE' or built['state'] != 'COMPLETE' or accepted['state'] != 'PASS'
        or exported['version'] != MAIN or built['version'] != MAIN or accepted['version'] != MAIN
        or actual != exported['bundle_sha256'] or actual != built['bundle_sha256']
        or _sha(exe) != built['exe_sha256'] or _sha(exe) != accepted['exe_sha256']):
        raise ValueError('official-best bundle or executable differs from acceptance')
    bundle = load_event_stack_bundle(folder, expected_role='deployment')
    policy = bundle.policy
    challenger_policy(dict(admission=dict(nms_iou=policy['nms_iou'],threshold=policy['admission_threshold'],
        max_candidates_per_subject=policy['max_candidates_per_subject']), threshold=policy['threshold']))
    if (policy['threshold'] != .4582664763345077 or policy['blend_weight'] != .25
        or policy['max_events_per_group'] is not None or bundle.run_config['trial_version'] != MAIN):
        raise ValueError('official-best policy identity differs')
    protected = protected_sources(root)
    protected.update({p.relative_to(root).as_posix(): _sha(p) for p in folder.iterdir() if p.is_file()})
    protected[exe.relative_to(root).as_posix()] = _sha(exe)
    return protected


def challenger_policy(original):
    if original['admission'] != dict(nms_iou=.3, threshold=.2, max_candidates_per_subject=6):
        raise ValueError('reference must be the frozen cap6 admission')
    threshold = original['threshold']
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('invalid frozen final threshold')
    result = deepcopy(original)
    result['admission']['max_candidates_per_subject'] = 8
    return result


def cap_gate(records, cohort):
    if cohort not in ('t24','five'): raise ValueError('unknown cap cohort')
    n, truths, majority = (3,160,2) if cohort == 't24' else (5,254,3)
    if len(records) != n or {r['fold'] for r in records} != set(range(n)):
        raise ValueError('complete unique folds required')
    for r in records:
        if r['events']['A']['truths'] != r['events']['G']['truths']:
            raise ValueError('paired denominator differs')
        for m in r['events'].values():
            if (any(m[k] < 0 or int(m[k]) != m[k] for k in ('tp','fp','fn','truths','predictions'))
                or m['tp']+m['fn'] != m['truths'] or m['tp']+m['fp'] != m['predictions']):
                raise ValueError('inconsistent event counts')
    total = {a: aggregate([r['events'][a] for r in records]) for a in ('A','G')}
    if total['A']['truths'] != truths: raise ValueError('complete truth population differs')
    delta = total['G']['f1'] - total['A']['f1']
    tp = total['G']['tp'] - total['A']['tp']; fp = total['G']['fp'] - total['A']['fp']
    nonworse = sum(aggregate([r['events']['G']])['f1'] >= aggregate([r['events']['A']])['f1'] for r in records)
    return dict(passed=bool(delta >= .005-1e-12 and tp >= 0 and fp <= 2*tp and nonworse >= majority),
                aggregate=total, delta_f1=delta, delta_tp=tp, delta_fp=fp, nonworse_folds=nonworse)


def execute(root, cohort='t24'):
    if cohort not in REFERENCES: raise ValueError('unknown cap cohort')
    root = Path(root).resolve(); output = root/OUTPUTS[cohort]
    from .policy_trial_export import ensure_new
    ensure_new(root, output)
    revision, edits = controlled_revision(root); protected = protected_baseline(root)
    def fingerprints():
        hashes = inputs(root, cohort, parent_directory=REFERENCES['t24'])
        for kind in ('t24', cohort):
            hashes.update({p.relative_to(root).as_posix(): _sha(p) for p in (root/REFERENCES[kind]).glob('*.json')})
        hashes[SPEC] = _sha(root/SPEC)
        return hashes
    if cohort == 'five':
        parent = root/OUTPUTS['t24']
        summary = _read(parent/'summary.json')
        records = [_read(parent/f'fold{k}_report.json') for k in range(3)]
        result = cap_gate(records,'t24')
        if summary['state'] != 'COMPLETE' or not result['passed'] or any(summary[k] != v for k,v in result.items()):
            raise ValueError('passed complete T24 cap8 evidence required')
        hashes = _read(parent/'plan.json')['source_sha256']
        if any(_sha(root/p) != h for p,h in hashes.items()): raise ValueError('parent cap8 source drift')
    output.mkdir(parents=True); started = time.perf_counter(); completed = 0
    try:
        hashes = fingerprints()
        _write(output/'plan.json', dict(id='CAP8-POLICY-V4', cohort=cohort, code_sha=revision,
            source_sha256=hashes, protected=protected, user_edits=edits, model_fits=0,
            score_recomputations=0, actual_exe_cv=False, comparison='frozen R3 main cap6 vs cap8'))
        count = 3 if cohort == 't24' else 5
        for k in range(count):
            old = _read(root/REFERENCES[cohort]/f'fold{k}_policy.json')
            _write(output/f'fold{k}_policy.json',dict(reference=old['policy'],challenger=challenger_policy(old['policy']),graph=old['graph']))
        _write(output/'policy_seal.json',dict(sha256={p.name:_sha(p) for p in output.glob('*.json')}))
        truths, mapping, _, wrappers = load_context(root, cohort, parent_directory=REFERENCES['t24'],
            parent_gate=lambda rows,kind:r3_gate(rows,kind,'deploy_f1'))
        source = root/(T24_SOURCE if cohort == 't24' else FIVE_SOURCE)
        if cohort == 't24':
            from .geometry_probe import MANIFEST
            manifest = _read(root/MANIFEST)
            forbidden = set(manifest['identities']['v7']) | set(manifest['identities']['outer'])
        else: forbidden = set()
        records = []
        for w in wrappers:
            k = w['id']; train = set(w['train']); score = set(w['score'])
            selected = _read(output/f'fold{k}_policy.json')
            check_graph(selected['graph'],'policy',score|forbidden,train)
            role = 'health' if cohort == 't24' else 'outer'
            cs, ps, gs, graph = _score_file(source/f'fold{k}_{role}_A_scores.json',mapping,train,score,forbidden)
            targets = partition_truths(truths,score)
            events = {a: decode_relative(cs,ps,gs,selected[key],use_relative=False)
                      for a,key in (('A','reference'),('G','challenger'))}
            original = _read(root/REFERENCES[cohort]/f'fold{k}_report.json')
            if ([asdict(e) for e in events['A']] != original['predictions']['G']
                or complete_metrics(events['A'],mapping,targets) != original['events']['G']):
                raise ValueError('official-best fold reference replay differs')
            row = dict(fold=k, events={a:complete_metrics(es,mapping,targets) for a,es in events.items()},
                predictions={a:[asdict(e) for e in es] for a,es in events.items()},
                policy=selected,score_graph=graph,raw_candidate_count=len(cs),
                subject_events=[dict(subject=s,events={a:complete_metrics([e for e in es if mapping[e.sid] == s],mapping,
                    partition_truths(targets,{s})) for a,es in events.items()}) for s in sorted(score)])
            records.append(row); completed += 1; _write(output/f'fold{k}_report.json',row)
            print(f'cap8 {cohort} fold{k}: '+str(row['events']),flush=True)
        if hashes != fingerprints() or protected != protected_baseline(root): raise ValueError('source/protected drift')
        if edits != {name:_sha(root/name) if (root/name).is_file() else None for name in USER_PATHS}:
            raise ValueError('user edits changed')
        if any(_sha(output/name) != h for name,h in _read(output/'policy_seal.json')['sha256'].items()):
            raise ValueError('sealed cap8 policies changed')
        peak = peak_working_set_bytes()
        if peak is None or peak > 12*1024**3 or time.perf_counter()-started > 600 or shutil.disk_usage(root).free < 10*1024**3:
            raise RuntimeError('cap8 resource budget exceeded')
        if sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > 16*1024**2:
            raise RuntimeError('cap8 evidence budget exceeded')
        result = dict(state='COMPLETE',cohort=cohort,**cap_gate(records,cohort),
            evidence_kind='REPEATED_FROZEN_MODEL_CAP8_POLICY_DEVELOPMENT_ONLY',actual_exe_cv=False,
            model_fits=0,score_recomputations=0,protected_unchanged=True,user_edits_unchanged=True,
            elapsed_seconds=time.perf_counter()-started)
        _write(output/'summary.json',result); _write(output/'state.json',dict(state='COMPLETE'))
        return result
    except BaseException as exc:
        _write(output/'state.json',dict(state='INCOMPLETE',error=repr(exc),completed_folds=completed)); raise


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cohort',choices=('t24','five'),default='t24')
    args=p.parse_args();print(execute(Path(__file__).resolve().parents[3],args.cohort))
