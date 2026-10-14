"""Describe sealed PPG evidence; no raw access, fit, threshold search or event F1."""
from collections import Counter
import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .audit import (assert_state, require_committed, verify, load_registration,
                    write_json, stage, output_size)
from .contract import OUTPUT, T1, P_NAMES, Q_NAMES, read_json, file_hash


def describe(values):
    values = np.asarray(values, float)
    x = values[np.isfinite(values)]
    return dict(n_total=int(values.size), n_valid=int(x.size), n_missing=int(values.size-x.size),
                status='sufficient' if x.size >= 5 else 'insufficient',
                median=float(np.median(x)) if len(x) else None,
                q25=float(np.quantile(x, .25)) if len(x) else None,
                q75=float(np.quantile(x, .75)) if len(x) else None)


def correlation(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    rho = (float(spearmanr(a[ok], b[ok]).statistic) if ok.sum() >= 5
           and np.ptp(a[ok]) > 0 and np.ptp(b[ok]) > 0 else None)
    return dict(n=int(ok.sum()), rho=rho)


def score_bin(value):
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('invalid frozen probability')
    return min(int(value*4), 3)


def compare_groups(values, true_mask):
    values = np.asarray(values, float); true_mask = np.asarray(true_mask, bool)
    result = []
    for column in values.T:
        a, b = describe(column[true_mask]), describe(column[~true_mask])
        delta = a['median']-b['median'] if min(a['n_valid'], b['n_valid']) >= 5 else None
        result.append(dict(true=a, false=b, median_true_minus_false=delta))
    return result


def event_key(e):
    return e['sid'], e['start_ms'], e['end_ms']


def overlap(e, p):
    if e['sid'] != p['sid']:
        return 0
    intersection = max(0, min(e['end_ms'], p['end_ms'])-max(e['start_ms'], p['start_ms']))
    union = max(e['end_ms'], p['end_ms'])-min(e['start_ms'], p['start_ms'])
    return intersection/union if union > 0 else 0


def selection_status(event, predictions):
    if predictions is None:
        return 'unknown_unavailable'
    matches = sum(event_key(event) == event_key(p) for p in predictions)
    if matches == 1:
        return 'accepted_exact'
    if matches > 1 or any(overlap(event, p) > 0 for p in predictions):
        return 'unknown_overlap'
    return 'not_exactly_selected'


def scene_for_truth(truth, subject, meals):
    matching = [m for m in meals if m['externalid'] == subject and
                float(m['beforeTime']) <= truth['start_ms'] and float(m['afterTime']) >= truth['end_ms']]
    if len(matching) != 1:
        return 'unknown'
    meal = matching[0]
    def hand(v):
        return str(v).replace('佩戴', '').strip()
    wear, habitual = hand(meal['wearHand']), hand(meal['dietaryHand'])
    if wear not in ('左手', '右手') or habitual not in ('左手', '右手'):
        return 'unknown'
    return 'dominant' if wear == habitual else 'nondominant'


def build_report(root):
    root = Path(root); out = root/OUTPUT
    assert_state(read_json(out/'state.json'), 'EXTRACTED')
    # Check commit seal before loading any diagnostic label/score field.
    for name in ('registry.json', 'inventory.json', 'state.json', 'extraction_seal.json',
                 'candidate_features.npz', 'window_stats.npz', 'file_stats.json', 'resources.json'):
        require_committed(root, out/name)
    verify(root)
    _, inv = load_registration(root)
    with np.load(out/'candidate_features.npz', allow_pickle=False) as cache:
        f = {k: cache[k] for k in cache.files}
    with np.load(out/'window_stats.npz', allow_pickle=False) as cache:
        windows = cache['values']; sids = cache['sid']; columns = cache['columns'].tolist()
    # Motion thresholds are descriptive, computed only on retained T24 windows.
    motion_medians = {}
    for subject in sorted(set(inv['subject_by_sid'].values())):
        ids = [s for s, owner in inv['subject_by_sid'].items() if owner == subject]
        x = windows[np.isin(sids, ids), columns.index('motion')]
        motion_medians[subject] = float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan
    coverage, conditional, all_rows, hand_rows = {}, {}, [], []
    from src.data.manifests import MEALS_CSV
    t24 = set(inv['subject_by_sid'].values())
    with open(MEALS_CSV, encoding='utf-8', newline='') as handle:
        meals = [m for m in csv.DictReader(handle) if m['externalid'] in t24]
    with stage(out, 'EXTRACTED', 'REPORTED'):
        partitions = sorted({o['partition'] for o in inv['occurrences']})
        for partition in partitions:
            payload = read_json(root/T1/f'{partition}_scores.json')
            occurrences = [o for o in inv['occurrences'] if o['partition'] == partition]
            indices = np.array([o['key_index'] for o in occurrences], int)
            row_ids = np.array([o['row'] for o in occurrences], int)
            subjects = np.array([inv['subject_by_sid'][inv['keys'][i]['sid']] for i in indices])
            labels = np.asarray(payload['labels'])[row_ids].astype(bool)
            scores = np.asarray(payload['scores']['b'])[row_ids]
            if len(labels) != len(indices) or not np.isfinite(scores).all():
                raise ValueError('diagnostic evidence alignment differs')
            for j, o in enumerate(occurrences):
                if event_key(payload['candidates'][o['row']]['event']) != event_key(inv['keys'][o['key_index']]):
                    raise ValueError('candidate identity differs from unlabeled seal')
            bins = np.array([score_bin(v) for v in scores])
            q, p = f['q_main'][indices], f['p_main'][indices]
            predictions = None
            health = '_health' in partition
            if health:
                evidence = read_json(root/T1/f'{partition}.json')
                predictions = evidence['predictions']['b']
            statuses = np.array([selection_status(inv['keys'][i], predictions) for i in indices])
            by_subject = {}
            for subject in sorted(set(subjects)):
                selected = subjects == subject
                by_subject[subject] = dict(n=int(selected.sum()),
                    q={n: describe(q[selected, j]) for j, n in enumerate(Q_NAMES)},
                    p={n: describe(p[selected, j]) for j, n in enumerate(P_NAMES)},
                    requested_ms=f['requested_ms_main'][indices[selected]].sum(axis=0).tolist(),
                    phase_available=np.isfinite(f['proxies_main'][indices[selected]]).sum(axis=0).tolist())
            reasons = Counter()
            pcov, pcount = f['proxy_coverage_main'][indices], f['proxy_counts_main'][indices]
            reasons['coverage_below_half'] = int((pcov < .5).sum())
            reasons['fewer_than_six_windows'] = int((pcount < 6).sum())
            coverage[partition] = dict(n=len(indices), by_subject=by_subject,
                q={n: describe(q[:, j]) for j, n in enumerate(Q_NAMES)},
                p={n: describe(p[:, j]) for j, n in enumerate(P_NAMES)},
                missing_gate_counts=dict(reasons), selection_status_counts=dict(Counter(statuses)))
            correlations = {qn: {pn: correlation(q[:, j], p[:, k]) for k, pn in enumerate(P_NAMES)}
                            for j, qn in enumerate(Q_NAMES)}
            motion_threshold = np.array([motion_medians[s] for s in subjects])
            motion = q[:, 5]  # during ACC motion
            low, high = np.isfinite(motion) & (motion <= motion_threshold), np.isfinite(motion) & (motion > motion_threshold)
            conditional[partition] = dict(
                p_true_false=compare_groups(p, labels), q_true_false=compare_groups(q, labels),
                fixed_base_bins={str(b): compare_groups(p[bins == b], labels[bins == b]) for b in range(4)},
                quality_correlations=correlations,
                motion_strata={name: dict(n=int(mask.sum()), p_true_false=compare_groups(p[mask], labels[mask]))
                               for name, mask in [('low', low), ('high', high)]},
                acceptance_groups={status: dict(n=int((statuses == status).sum()),
                    p_true_false=compare_groups(p[statuses == status], labels[statuses == status])) for status in sorted(set(statuses))},
                anchor_difference={name: describe(f['p_shift'][indices, j]-p[:, j]) for j, name in enumerate(P_NAMES)},
                anchor_coverage_difference={name: describe(f['q_shift'][indices, j]-q[:, j]) for j, name in enumerate(Q_NAMES) if 'coverage' in name},
                anchor_availability_changed=(np.isfinite(f['p_shift'][indices]) != np.isfinite(p)).sum(axis=0).tolist())
            if health:
                for j, i in enumerate(indices):
                    all_rows.append(dict(partition=partition, key_index=int(i), subject=subjects[j],
                                         label=bool(labels[j]), score=float(scores[j]), status=statuses[j],
                                         p_available=np.isfinite(p[j]).tolist()))
                for truth in evidence['truths']:
                    subject = inv['subject_by_sid'].get(truth['sid'])
                    # A truth can belong to a session with no candidate; use T24 manifest ownership.
                    if subject is None:
                        subject = read_json(root/T1/'manifest.json')['subject_by_session'].get(truth['sid'])
                    if subject not in t24:
                        raise ValueError('non-T24 diagnostic truth')
                    related = [int(i) for i in indices if overlap(truth, inv['keys'][i]) > .25]
                    hand_rows.append(dict(subject=subject, scene=scene_for_truth(truth, subject, meals),
                        truth=truth, reachable=bool(related), related_keys=related,
                        any_p_available=bool(related and np.isfinite(f['p_main'][related]).any()),
                        any_full_p_available=bool(related and np.isfinite(f['p_main'][related]).all(axis=1).any())))
        conditional['hand_diagnostic'] = {scene: dict(
            gt=sum(r['scene'] == scene for r in hand_rows),
            reachable=sum(r['scene'] == scene and r['reachable'] for r in hand_rows),
            any_p_available=sum(r['scene'] == scene and r['any_p_available'] for r in hand_rows),
            any_full_p_available=sum(r['scene'] == scene and r['any_full_p_available'] for r in hand_rows))
            for scene in ('dominant', 'nondominant', 'unknown')}
        conditional['hand_rows'] = hand_rows
        conditional['health_occurrences'] = all_rows
        write_json(out/'coverage_by_partition.json', coverage)
        write_json(out/'conditional_evidence.json', conditional)
        lines = ['# A7 P0 候选级PPG审计', '',
                 '描述性T24诊断；原候选及评分冻结。本轮无训练、无event F1、无V7/outer读取。', '',
                 f'唯一候选{len(inv["keys"])}，分区出现记录{len(inv["occurrences"])}，原始文件{len(inv["files"])}。', '',
                 '|health分区|候选|正候选|至少一维P可用|六维P均可用|',
                 '|---|---:|---:|---:|---:|']
        for partition in [p for p in partitions if '_health' in p]:
            rows = [r for r in all_rows if r['partition'] == partition]
            lines.append(f'|{partition}|{len(rows)}|{sum(r["label"] for r in rows)}|'
                         f'{sum(any(r["p_available"]) for r in rows)}|{sum(all(r["p_available"]) for r in rows)}|')
        lines += ['', '## 手侧相关GT覆盖', '', '|场景|GT|存在相关候选|至少一维P|六维P|', '|---|---:|---:|---:|---:|']
        for scene, row in conditional['hand_diagnostic'].items():
            lines.append(f'|{scene}|{row["gt"]}|{row["reachable"]}|{row["any_p_available"]}|{row["any_full_p_available"]}|')
        lines += ['', '## 解读边界', '',
            'PPG3/5绿光映射、25Hz与首样本包锚点仍为假设。谱峰一致不证明生理准确性，共同运动可能贡献一致性。',
            'P为固定阶段差值，late相对候选开始定义且可能与during重叠；45分钟等待仅适合离线推理。',
            'motion分层基于本轮保留候选附近窗口的逐人中位数，仅作描述。条件差异及相关性不能证明因果或增量F1。',
            '短候选可能因6窗口门槛缺失；缺失未填零或插值。health和policy记录不混作独立样本。',
            '接受状态使用保存预测的精确事件匹配；重叠合并情况为unknown，policy缺少保存事件选择记录时为unknown。',
            '详细分数分箱、motion分层、Q/P相关、锚点敏感性见conditional_evidence.json；不足5个有效候选的方向不作结论。',
            '本审计没有自动晋级门槛；后续融合须按训练主体内部crossfit构造数据，不能拼接这12份OOF随机划分训练。', '']
        (out/'report.md').write_text('\n'.join(lines), encoding='utf-8')
        if output_size(out) > 128*2**20:
            raise RuntimeError('report output exceeds budget')
        write_json(out/'report_seal.json', {name: file_hash(out/name) for name in
                                          ['coverage_by_partition.json', 'conditional_evidence.json', 'report.md']})
    verify(root)


if __name__ == '__main__':
    build_report(Path.cwd())
