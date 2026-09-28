from copy import deepcopy
import hashlib
import json
import numpy as np
import pytest

from scripts.research.representation_v2.neutral_probe import (
    neutral_scores,check_read_phase,validate_health_topology,neutral_gate,subject_event_counts,
    assert_baseline_replay,load_frozen_policies,
)


def test_invalid_rows_become_base_and_valid_rows_are_bitwise_unchanged():
    scores={'b':[.8,.2,.9],'r':[.65,.3,.7],'f':[.7,.25,.75]}
    original=deepcopy(scores)
    result=neutral_scores(scores,[False,True,False])
    assert result['b'].tolist()==scores['b']
    assert result['r'].tolist()==[.8,.3,.9]
    assert result['f'].tolist()==[.8,.25,.9]
    assert scores==original
    result['f'][0]=0
    assert scores==original


@pytest.mark.parametrize('mask',[[True],[0,1],[True,None]])
def test_misaligned_or_nonboolean_motion_masks_are_rejected(mask):
    with pytest.raises(ValueError):neutral_scores({'b':[.4,.8],'r':[.3,.7],'f':[.2,.6]},mask)


def test_no_health_read_until_all_three_policy_wrappers_frozen():
    for frozen in [set(),{0},{0,1}]:
        with pytest.raises(RuntimeError):check_read_phase('fold0_health_scores.json',frozen)
        with pytest.raises(RuntimeError):check_read_phase('fold2_report.json',frozen)
        assert check_read_phase('fold0_policy0_scores.json',frozen)=='fold0_policy0_scores.json'
    assert check_read_phase('fold1_health_scores.json',{0,1,2})=='fold1_health_scores.json'
    with pytest.raises(ValueError):check_read_phase('../fold1_report.json',{0,1,2})


def test_health_topology_must_match_policy_training_and_exclusions():
    health={'train':['a','b','c'],'score':['h'],'excluded':['v']}
    ledgers=[{'score':[s],'excluded':['h','v']} for s in ['a','b','c']]
    validate_health_topology(health,ledgers)
    health['train']=['a','b','h']
    with pytest.raises(ValueError):validate_health_topology(health,ledgers)


def test_neutral_gate_requires_baseline_gain_and_improvement_over_old_fd():
    metric=lambda tp,fp:{'n_tp':tp,'n_pred':tp+fp,'n_true':10}
    row={'events':{'b':metric(7,5),'r':metric(7,4),'f':metric(8,3)},'checks':{'integrity':True,'resource':True}}
    rows=[deepcopy(row) for _ in range(3)]
    old=[{'events':{'f':metric(7,3)}} for _ in range(3)]
    assert neutral_gate(rows,old)['passed']
    old=[{'events':{'f':metric(9,0)}} for _ in range(3)]
    assert not neutral_gate(rows,old)['passed']


def test_subject_counts_keep_truths_from_sessions_without_candidates():
    from src.pipeline.event_stack import EventRef,MultiScaleCandidate
    candidates=[MultiScaleCandidate(EventRef('seen',0,100),None,None)]
    truths=[EventRef('seen',0,100),EventRef('no-candidate',0,100)]
    old=[{'subject':'a','events':{'b':{'n_true':2}}}]
    result=subject_event_counts(candidates,['a'],truths,{'b':[EventRef('seen',0,100)]},old)
    assert result[0]['events']['b']['n_tp']==1
    assert result[0]['events']['b']['fn']==1
    assert result[0]['events']['b']['sensitivity']==.5


@pytest.mark.parametrize('defect',['candidate','subject'])
def test_changed_baseline_diagnostics_fail_closed(defect):
    metric={'n_tp':1,'n_pred':1,'n_true':2,'sensitivity':.5,'ppv':1.,'f1':2/3}
    old={'events':{'b':metric},'predictions':{'b':[{'sid':'s','start_ms':0,'end_ms':100}]},
         'candidate_metrics':{'b':{'ap':.8,'bce':.2}},'subject_events':[{'subject':'a','events':{'b':metric}}]}
    current=deepcopy(old); assert_baseline_replay(current,old)
    if defect=='candidate':current['candidate_metrics']['b']['ap']=.81
    else:current['subject_events'][0]['events']['b']['sensitivity']=.6
    with pytest.raises(RuntimeError):assert_baseline_replay(current,old)


def test_all_frozen_policy_files_are_reloaded_and_hash_checked(tmp_path):
    hashes={}
    for i in range(3):
        path=tmp_path/f'fold{i}_policy_selection.json'
        path.write_text(json.dumps({'fold':i,'policies':{a:{'threshold':.5} for a in ['b','r','f']}}))
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    marker=tmp_path/'policy_phase_frozen.json'
    marker.write_text(json.dumps({'folds':[0,1,2],'policy_file_sha256':hashes,'health_files_read':0}))
    digest=hashlib.sha256(marker.read_bytes()).hexdigest()
    loaded=load_frozen_policies(tmp_path,digest)
    assert loaded[2]['policies']['f']['threshold']==.5
    (tmp_path/'fold2_policy_selection.json').write_text('{}')
    with pytest.raises(RuntimeError):load_frozen_policies(tmp_path,digest)
