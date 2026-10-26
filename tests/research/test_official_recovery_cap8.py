"""Behavioral checks for the fixed final-window cap8 comparison."""
from copy import deepcopy
import importlib
from importlib.util import find_spec
from pathlib import Path
import subprocess

import numpy as np
import pytest

from src.pipeline.event_stack import EventRef, MultiScaleCandidate
from scripts.research.official_recovery.relative_score_probe import decode_relative


def api():
    name = 'scripts.research.official_recovery.cap8_probe'
    assert find_spec(name) is not None, 'fixed cap8 probe is not implemented'
    return importlib.import_module(name)


def policy():
    return dict(admission=dict(nms_iou=.3, threshold=.2, max_candidates_per_subject=6), threshold=.4582664763345077)


def metric(tp, fp, truths):
    return dict(tp=tp, fp=fp, fn=truths-tp, truths=truths, predictions=tp+fp,
                f1=2*tp/(tp+fp+truths), ppv=tp/(tp+fp), recall=tp/truths)


def reports(extra_tp=1, extra_fp=0):
    return [dict(fold=k, events=dict(A=metric(25, 10, n), G=metric(25+extra_tp, 10+extra_fp, n)))
            for k, n in enumerate((54, 55, 51))]


def test_cap8_recovers_only_two_extra_candidates_without_lowering_threshold():
    original = policy(); before = deepcopy(original)
    new = api().challenger_policy(original)
    cs = [MultiScaleCandidate(EventRef('s', 1000*i, 1000*i+100), None, None) for i in range(9)]
    scores = np.array([.95,.9,.85,.8,.75,.7,.6,.5,.4])
    old_events = decode_relative(cs, scores, ['person']*9, original, use_relative=False)
    new_events = decode_relative(cs, scores, ['person']*9, new, use_relative=False)
    assert original == before
    assert [e.start_ms for e in old_events] == [0,1000,2000,3000,4000,5000]
    assert [e.start_ms for e in new_events] == [0,1000,2000,3000,4000,5000,6000,7000]
    assert new['threshold'] == .4582664763345077


@pytest.mark.parametrize('key,value', [('max_candidates_per_subject',8),('nms_iou',.5),('threshold',.35)])
def test_nonincumbent_admission_is_rejected(key, value):
    p = policy(); p['admission'][key] = value
    with pytest.raises(ValueError): api().challenger_policy(p)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -.1, 1.1])
def test_bad_event_threshold_is_rejected(value):
    p = policy(); p['threshold'] = value
    with pytest.raises(ValueError): api().challenger_policy(p)


def test_trial_gate_requires_actual_f1_gain_and_bounded_fp():
    good = api().cap_gate(reports(), 't24')
    assert good['passed'] is True
    assert good['delta_tp'] == 3 and good['delta_fp'] == 0
    assert api().cap_gate(reports(0,0), 't24')['passed'] is False
    assert api().cap_gate(reports(1,3), 't24')['passed'] is False


def test_gate_rejects_duplicate_fold_and_changed_denominator():
    rows = reports(); rows[2]['fold'] = 1
    with pytest.raises(ValueError): api().cap_gate(rows, 't24')
    rows = reports(); rows[1]['events']['G'] = metric(26,10,54)
    with pytest.raises(ValueError): api().cap_gate(rows, 't24')


def test_gate_rejects_inconsistent_counts():
    rows = reports(); rows[0]['events']['G']['predictions'] = 999
    with pytest.raises(ValueError): api().cap_gate(rows, 't24')


def test_only_user_submission_edits_may_remain_uncommitted(tmp_path):
    def git(*args):
        return subprocess.run(['git',*args],cwd=tmp_path,check=True,capture_output=True,text=True).stdout
    git('init','-q'); git('config','user.name','Test'); git('config','user.email','test@example.invalid')
    readme=tmp_path/'dist/official_trials/deployment_f1_policy_v3/README.md'
    readme.parent.mkdir(parents=True); readme.write_text('user original')
    git('add','.'); git('commit','-qm','fixture')
    readme.unlink(); archive=tmp_path/'dist/official_trials/submission.zip'; archive.write_bytes(b'user zip')
    revision, edits=api().controlled_revision(tmp_path)
    assert revision == git('rev-parse','HEAD').strip()
    assert edits['dist/official_trials/deployment_f1_policy_v3/README.md'] is None
    assert archive.read_bytes() == b'user zip' and not readme.exists()
    (tmp_path/'uncommitted.py').write_text('new code')
    with pytest.raises(RuntimeError,match='uncommitted'): api().controlled_revision(tmp_path)


def test_real_official_best_model_and_exe_are_protected():
    root = Path(__file__).resolve().parents[2]
    identity = api().protected_baseline(root)
    assert len(identity) == 75
    assert identity['dist/official_trials/deployment_f1_policy_v3/EatingDetection.exe'] == 'fa6987b6244fe73c76ee741ea352fcc74e8e4de06de332812e99b1f1d963aa44'
