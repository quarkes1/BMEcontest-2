"""Only complete paired evidence earns a gate; subject tails remain visible."""
from copy import deepcopy

import numpy as np
import pytest


def test_incomplete_fold_set_has_no_gate():
    from scripts.research.representation_v2.transformer.reporting import summarize
    with pytest.raises(ValueError,match='three complete'):
        summarize([])


def test_no_gt_subject_fp_and_short_match_are_not_lost():
    from scripts.research.representation_v2.transformer.reporting import make_record
    from src.pipeline.event_stack import EventRef,MultiScaleCandidate
    truths=[EventRef('a',0,100)]
    predictions={a:[EventRef('a',0,100),EventRef('b',0,100)] for a in ('b','p','c','t')}
    candidates=[MultiScaleCandidate(e,None,None) for e in predictions['b']]
    record=make_record(0,predictions,truths,{'a':'x','b':'y'},candidates,np.array([1,0]),
                       {a:np.array([.9,.7]) for a in predictions})
    assert record['events']['t']['fp']==1
    assert record['short_events']['t']=={'n_true':1,'n_tp':1}
    subject=next(s for s in record['subject_events'] if s['subject']=='y')
    assert subject['events']['t']['n_true']==0 and subject['events']['t']['fp']==1


def test_one_big_fold_gain_does_not_pass_majority_gate():
    from scripts.research.representation_v2.transformer.reporting import summarize
    def metrics(tp,fp):
        return dict(n_tp=tp,n_true=10,n_pred=tp+fp,fp=fp,fn=10-tp,f1=2*tp/(10+tp+fp),ppv=tp/(tp+fp),recall=tp/10)
    records=[]
    for k in range(3):
        base=metrics(5,5); new=metrics(10,0) if k==0 else metrics(4,5)
        arms={a:deepcopy(base if a=='b' else new) for a in ('b','p','c','t')}
        empty=dict(n_tp=0,n_true=0,n_pred=0,fp=0,fn=0,f1=0.,ppv=0.,recall=0.)
        records.append(dict(wrapper=k,events=arms,outer_calls=0,
                subject_events=[{'subject':f'{k}-{i}','events':deepcopy(arms) if i==0 else {a:deepcopy(empty) for a in arms}} for i in range(8)],
                short_events={a:{'n_true':10,'n_tp':arms[a]['n_tp']} for a in arms},
                candidate_recall={'n_true':10,'n_reachable':10}))
    result=summarize(records)
    assert result['events']['t']['f1']>result['events']['b']['f1']+.01
    assert not result['gate']['t']['passed'] and result['gate']['t']['nonworse_wrappers']==1
    assert result['outer_calls']==0 and result['development_only']
    records[2]['subject_events'][0]['subject']='0-0'
    with pytest.raises(ValueError,match='subjects'):
        summarize(records)
