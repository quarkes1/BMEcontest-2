import numpy as np
import pytest


def test_fixed_max_fusion_checks_original_component_identity():
    from scripts.research.official_recovery.max_fusion_probe import fuse
    lr=np.array([.1,.9]);gb=np.array([.8,.2]);old=.25*lr+.75*gb
    np.testing.assert_array_equal(fuse(lr,gb,old),[.8,.9])
    np.testing.assert_array_equal(old,.25*lr+.75*gb)
    for a,b,c in ((lr,gb,old+.01),([np.nan], [.1],[.1]),([1.1],[.1],[.35]),([.1],[.2,.3],[.2])):
        with pytest.raises(ValueError): fuse(a,b,c)


def test_threshold_uses_fixed_admission_and_complete_subject_truths():
    from scripts.research.official_recovery.max_fusion_probe import select_threshold
    from src.pipeline.event_stack import EventRef
    from src.pipeline.event_stack import MultiScaleCandidate
    from scripts.research.official_recovery.event_probe import decode
    candidates=[MultiScaleCandidate(EventRef('a',0,100),None,None),MultiScaleCandidate(EventRef('a',200,300),None,None)]
    admission={'nms_iou':.3,'threshold':.2,'max_candidates_per_subject':6}
    selected=select_threshold(candidates,np.array([.9,.5]),['S','S'],admission,{'a':'S'},[('S',0,100),('S',1000,1100)])
    assert selected['threshold']==.9 and selected['admission']==admission
    assert selected['selection_metrics']['truths']==2
    assert decode(candidates,np.array([.9,.5]),['S','S'],selected)==[candidates[0].event]


def test_component_reader_rejects_foreign_subject_and_invalid_component(tmp_path):
    import json
    from scripts.research.official_recovery.max_fusion_probe import load_scores
    obj={'train_subjects':['A'],'score_subjects':['B'],'graph':{'score':{'fit_subjects':['A'],'parents':[]}},
         'columns':['row','sid','start_ms','end_ms','macro','micro','subject','lr','lgbm','score'],
         'rows':[[0,'a',0,100,True,True,'B',.5,.9,.8]]}
    path=tmp_path/'scores.json';path.write_text(json.dumps(obj))
    assert load_scores(path,{'a':'B'},{'A'},{'B'},())[1].tolist()==[.9]
    with pytest.raises(ValueError): load_scores(path,{'a':'C'},{'A'},{'B'},())
    obj['rows'][0][7]=-1;path.write_text(json.dumps(obj))
    with pytest.raises(ValueError): load_scores(path,{'a':'B'},{'A'},{'B'},())


def test_fusion_gate_allows_changed_prediction_count_but_not_changed_truths():
    from scripts.research.official_recovery.max_fusion_probe import fusion_gate
    rows=[dict(fold=i,events={'A':dict(tp=20,predictions=30,truths=n),
        'G':dict(tp=23,predictions=31,truths=n)}) for i,n in enumerate((54,55,51))]
    assert fusion_gate(rows,'t24')['passed']
    rows[0]['events']['G']['truths']-=1
    with pytest.raises(ValueError): fusion_gate(rows,'t24')
