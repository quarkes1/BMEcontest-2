from copy import deepcopy
import numpy as np
import pytest
from scripts.research.representation_v2.stability_audit import profile, validate_ledger, allowed_input, sensitivity, validate_policy_graph
from src.pipeline.event_stack import EventRef,MultiScaleCandidate


def ledger():
    return {'train':['t'],'score':['v'],'excluded':['h'],
            'graph':{'score':{'fit_subjects':['t'],'parents':[]}},
            'groups':['v','v'],'labels':[0,1],'motion_valid':[False,True],
            'candidates':[{'event':{'sid':'s','start_ms':0,'end_ms':100}},
                          {'event':{'sid':'s','start_ms':200,'end_ms':300}}],
            'scores':{'b':[.4,.8],'r':[.35,.7],'f':[.35,.75]},
            'component_probabilities':{'base':[.4,.8],'r':[.2,.4],'f':[.2,.6]},
            'readouts':{'r':{'prior':.2},'f':{'prior':.2}}}


def test_reliability_endpoints_and_empty_population_are_not_dropped():
    p=profile([0,1],[0.,1.])
    assert p['n']==2 and p['brier']==0 and p['ece']==0
    assert p['bins'][0]['n']==1 and p['bins'][9]['n']==1
    assert profile([],[]) is None


@pytest.mark.parametrize('defect',['fusion','prior','ancestry','shape','nonfinite'])
def test_corrupt_evidence_is_rejected(defect):
    d=ledger(); validate_ledger(d)
    if defect=='fusion':d['scores']['f'][0]=.9
    if defect=='prior':d['readouts']['f']['prior']=.3
    if defect=='ancestry':d['graph']['score']['fit_subjects']=['v']
    if defect=='shape':d['labels'].pop()
    if defect=='nonfinite':d['component_probabilities']['f'][1]=float('nan')
    with pytest.raises(ValueError):validate_ledger(d)


@pytest.mark.parametrize('name',['fold0_health_scores.json','../fold0_policies.json','fold3_policy0_scores.json'])
def test_health_and_unsafe_paths_are_not_allowed(name):
    with pytest.raises(ValueError):allowed_input(name)


def test_sensitivity_keeps_unreachable_truth_and_original_tie_policy():
    candidates=[MultiScaleCandidate(EventRef('s',0,100),None,None),
                MultiScaleCandidate(EventRef('s',200,300),None,None)]
    truths=[EventRef('s',0,100),EventRef('unreachable',0,100)]
    policy={'admission':{'nms_iou':.5,'threshold':.2,'max_candidates_per_subject':5},'threshold':.5}
    rows=sensitivity(candidates,np.array([.5,.47]),['v','v'],truths,policy)
    assert len(rows)==5
    assert rows[0]['tp']==1 and rows[0]['fp']==1 and rows[0]['fn']==1
    assert rows[2]['tp']==1 and rows[2]['fp']==0 and rows[2]['fn']==1
    assert rows[-1]['tp']==0 and rows[-1]['fn']==2
    assert all('f1' not in r for r in rows)


def test_policy_graph_must_match_actual_score_ancestors_and_exclude_health():
    ledgers=[]
    for subject in ['a','b','c']:
        d=ledger(); d['score']=[subject]; d['train']=sorted({'a','b','c'}-{subject})
        d['groups']=[subject,subject]; d['graph']['score']['fit_subjects']=d['train']
        ledgers.append(d)
    policy={'graph':{'p0_score':{'fit_subjects':['b','c'],'parents':[]},
                     'p1_score':{'fit_subjects':['a','c'],'parents':[]},
                     'p2_score':{'fit_subjects':['a','b'],'parents':[]},
                     'policy':{'fit_subjects':['a','b','c'],'parents':['p0_score','p1_score','p2_score']}}}
    validate_policy_graph(ledgers,policy)
    leaked=deepcopy(policy); leaked['graph']['policy']['fit_subjects'].append('h')
    with pytest.raises(ValueError):validate_policy_graph(ledgers,leaked)
    inconsistent=deepcopy(policy); inconsistent['graph']['p0_score']['fit_subjects']=['a','c']
    with pytest.raises(ValueError):validate_policy_graph(ledgers,inconsistent)
