"""Source-only proposal diagnostic. No target fit or release entry point."""
import argparse
from dataclasses import asdict
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

from ..ppg.audit import (write_json, require_committed, verify_hashes, protected_snapshot,
                         create_output, stage, owned_rss, output_size)
from ..ppg.contract import T1, file_hash, read_json
from ..transformer.seed_ensemble import FrozenSequences, cache_snapshot
from ..transformer.checkpoint_audit import read_tokens
from ..transformer.contract import SOURCE_SHA
from ..transfer_probe import load_source_artifact
from ..validation_metrics import matched_indices
from src.pipeline.event_stack import EventRef
from src.eval.metrics import event_iou

OUTPUT = Path('outputs/challenger_70/representation_v2/amendment_a8_dense_localization')
SPEC = 'docs/challenge_070_amendment_a8_dense_localization.md'
SEED = Path('outputs/challenger_70/representation_v2/amendment_a6_transformer/seed_ensemble')
SOURCE = Path('outputs/challenger_70/representation_v2/amendment_a3_source')


def proposals(tokens, scores):
    p = np.asarray(scores, float)
    if p.shape != tokens.center_ms.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('invalid dense probabilities')
    result, hits = [], []
    def finish():
        if len(hits) >= 2:
            left = max(tokens.start_ms, int(tokens.center_ms[hits[0]])-2500)
            right = min(tokens.end_ms, int(tokens.center_ms[hits[-1]])+2500)
            if right > left:
                result.append(EventRef(tokens.sid, left, right))
        hits.clear()
    previous = None
    for i, value in enumerate(p):
        if not tokens.valid[i] or tokens.span_id[i] < 0:
            finish(); previous = None
            continue
        if previous is not None and (tokens.span_id[i] != tokens.span_id[previous]
                                     or tokens.center_ms[i]-tokens.center_ms[previous] != 5000):
            finish()
        if value >= .5:
            if hits and tokens.center_ms[i]-tokens.center_ms[hits[-1]] > 120000:
                finish()
            hits.append(i)
        previous = i
    finish()
    return result


def register(root):
    root = Path(root)
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root).strip():
        raise RuntimeError('commit code before D0 registration')
    old = read_json(root/SEED/'registry.json')
    snapshot = cache_snapshot(root/T1/'tokens')
    if snapshot != old['token_cache']:
        raise RuntimeError('sealed T1 token inventory differs')
    names = [SPEC, (SEED/'registry.json').as_posix(), (T1/'manifest.json').as_posix(),
             (T1/'raw_checksums.json').as_posix()]
    names += [(SOURCE/n).as_posix() for n in ('source_final.pt', 'plan.json', 'summary.json')]
    names += [(T1/f'wrapper{k}_{suffix}.json').as_posix() for k in range(3)
              for suffix in ('health', 'health_scores')]
    names += [p.relative_to(root).as_posix() for p in (root/'scripts/research/representation_v2/localization').glob('*.py')]
    # Lock all imported research/production code, not merely the new CLI.
    names += [n for n in subprocess.check_output(['git', 'ls-files', 'src', 'scripts'], cwd=root, text=True).splitlines()
              if n.endswith('.py')]
    files = {n: file_hash(root/n) for n in set(names)}
    files.update({(T1/'tokens'/n).as_posix(): sha for n, sha in snapshot.items()})
    out = root/OUTPUT; create_output(out)
    write_json(out/'registry.json', dict(files=files, protected_index=protected_snapshot(root),
                                        cache_files=sorted(snapshot), threshold=.5, gap_ms=120000, min_hits=2,
                                        target_fits=0, optimizer_steps=0, outer_calls=0))
    write_json(out/'state.json', dict(status='REGISTERED', outer_calls=0))


def registration(root):
    root = Path(root); r = read_json(root/OUTPUT/'registry.json')
    verify_hashes(root, r['files'])
    if protected_snapshot(root) != r['protected_index'] or (r['threshold'], r['gap_ms'], r['min_hits']) != (.5, 120000, 2):
        raise RuntimeError('D0 contract changed')
    if sorted(cache_snapshot(root/T1/'tokens')) != r['cache_files']:
        raise RuntimeError('D0 token inventory changed')
    return r


def budget(root, out, started, previous_seconds=0):
    elapsed = time.perf_counter()-started+previous_seconds
    rss = owned_rss([os.getpid()])
    if elapsed > 600 or rss > 8*2**30 or output_size(out) > 16*2**20 or shutil.disk_usage(root).free < 20*2**30:
        raise RuntimeError('D0 budget exceeded')
    return elapsed, rss


def extract(root):
    root = Path(root); out = root/OUTPUT
    require_committed(root, out/'registry.json'); require_committed(root, out/'state.json')
    registration(root); start = time.perf_counter()
    with stage(out, 'REGISTERED', 'EXTRACTED'), threadpool_limits(limits=2):
        manifest = read_json(root/T1/'manifest.json')
        sequences = FrozenSequences(root, manifest)
        source = load_source_artifact(root/SOURCE, SOURCE_SHA)
        w = source['head']['weight'].numpy().ravel(); bias = float(source['head']['bias'].item())
        allowed = set(manifest['subject_by_session'])
        rows, peak, last_check = [], 0, 0
        for sid, header in sequences.index.items():
            tokens = read_tokens(header, allowed_sids=allowed, raw_hashes=sequences.raw, source_sha=SOURCE_SHA)
            score = expit(tokens.embeddings@w+bias)
            events = proposals(tokens, score)
            rows.append(dict(sid=sid, subject=manifest['subject_by_session'][sid],
                             start_ms=tokens.start_ms, end_ms=tokens.end_ms, n_tokens=len(score),
                             valid_tokens=int(tokens.valid.sum()), positive_tokens=int(((score >= .5)&tokens.valid).sum()),
                             proposals=[asdict(e) for e in events]))
            if time.perf_counter()-last_check > 2:
                elapsed, rss = budget(root, out, start); peak = max(peak, rss); last_check = time.perf_counter()
        write_json(out/'proposals.json', dict(sessions=rows, source_sha256=SOURCE_SHA,
                                              diagnostic_only=True, partial_session_cache=True, outer_calls=0))
        elapsed, rss = budget(root, out, start)
        write_json(out/'resources.json', dict(elapsed_seconds=elapsed, peak_rss_bytes=max(peak, rss),
                                              raw_bytes_read=0, token_sessions=len(rows), target_fits=0, outer_calls=0))
        write_json(out/'seal.json', {n: file_hash(out/n) for n in ('proposals.json', 'resources.json')})
        budget(root, out, start)


def reach(events, truths):
    return {j for j, t in enumerate(truths) if any(p.sid == t.sid and event_iou(p.interval, t.interval) >= .25 for p in events)}


def report(root):
    root = Path(root); out = root/OUTPUT
    require_committed(root, out/'seal.json'); require_committed(root, out/'resources.json')
    require_committed(root, out/'proposals.json')
    verify_hashes(out, read_json(out/'seal.json')); registration(root)
    start = time.perf_counter(); previous = read_json(out/'resources.json')['elapsed_seconds']
    with stage(out, 'EXTRACTED', 'REPORTED'):
        rows = read_json(out/'proposals.json')['sessions']; index = {r['sid']: r for r in rows}
        records = []
        for k in range(3):
            health = read_json(root/T1/f'wrapper{k}_health.json')
            old = read_json(root/T1/f'wrapper{k}_health_scores.json')
            truths = [EventRef(**t) for t in health['truths']]
            baseline = [EventRef(**c['event']) for c in old['candidates']]
            subjects = set(old['score'])
            new = [EventRef(**p) for r in rows if r['subject'] in subjects for p in r['proposals']]
            b, n = reach(baseline, truths), reach(new, truths)
            _, matched = matched_indices(new, truths)
            records.append(dict(wrapper=k, n_true=len(truths), cached_gt=sum(t.sid in index for t in truths),
                                old_candidate_count=len(baseline), new_candidate_count=len(new),
                                old_reachable=len(b), new_reachable=len(n), union_reachable=len(b|n),
                                additional_gt=[dict(index=j, truth=asdict(truths[j])) for j in sorted(n-b)],
                                candidate_without_gt_overlap=sum(not reach([p], truths) for p in new),
                                one_to_one_matched=len(matched), unmatched_candidates=len(new)-len(matched)))
        keys = ('n_true', 'cached_gt', 'old_candidate_count', 'new_candidate_count', 'old_reachable',
                'new_reachable', 'union_reachable', 'candidate_without_gt_overlap', 'one_to_one_matched', 'unmatched_candidates')
        totals = {key: sum(r[key] for r in records) for key in keys}
        totals['additional_gt'] = sum(len(r['additional_gt']) for r in records)
        result = dict(totals=totals, wrappers=records, cached_sessions=len(rows),
                      full_t24_sessions=len(read_json(root/T1/'manifest.json')['subject_by_session']),
                      valid_token_seconds=sum(r['valid_tokens']*5 for r in rows), diagnostic_only=True,
                      event_f1=None, formal_f1=None, outer_calls=0)
        write_json(out/'result.json', result)
        text = '# A8 D0 source-only定位诊断\n\n仅现有227会话缓存；不训练、不调整规则、不评价最终F1。\n\n'
        text += '|wrapper|GT|缓存覆盖GT|原可达|新可达|联合可达|新增GT|新候选|无GT重叠候选|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
        for r in records:
            text += f'|{r["wrapper"]}|{r["n_true"]}|{r["cached_gt"]}|{r["old_reachable"]}|{r["new_reachable"]}|{r["union_reachable"]}|{len(r["additional_gt"])}|{r["new_candidate_count"]}|{r["candidate_without_gt_overlap"]}|\n'
        text += '\n汇总：'+str(totals)+'\n\n新候选FP成本不是最终事件FP；source概率未经目标校准。既有缓存会话由旧候选筛选，存在覆盖选择偏差，缺缓存不当作阴性。正式F1仍.6514285714，dist未改变。\n'
        (out/'report.md').write_text(text, encoding='utf-8')
        budget(root, out, start, previous)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=('register', 'extract', 'report'))
    args = parser.parse_args(); globals()[args.phase](Path.cwd())


if __name__ == '__main__':
    main()
