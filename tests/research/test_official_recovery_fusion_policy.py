"""Finite convex fusion, training-only selection and matched deployment aggregation."""
import importlib
from importlib.util import find_spec

import numpy as np
import pytest

from src.pipeline.event_stack import EventRef, MultiScaleCandidate


def api():
    name='scripts.research.official_recovery.fusion_policy_probe'
    assert find_spec(name) is not None, 'fusion policy probe missing'
    return importlib.import_module(name)


def test_fusion_validates_components_and_preserves_incumbent_weight():
    assert api().blend_scores([.2,.8],[.6,.4],.25).tolist() == pytest.approx([.5,.5])
    for lr, lgbm, w in (([float('nan')],[.2],.25),([.2],[1.1],.25),([.2,.3],[.2],.25),([.2],[.4],.33)):
        with pytest.raises(ValueError): api().blend_scores(lr,lgbm,w)


def test_train_selection_uses_available_discriminating_component():
    cs=[MultiScaleCandidate(EventRef('session',1000*i,1000*i+500),None,None) for i in range(2)]
    result=api().choose_fusion(cs,np.array([.9,.1]),np.array([.2,.8]),['HNU00001','HNU00001'],
        {'session':'HNU00001'},[('HNU00001',0,500)])
    assert result['policy']['weight'] in (.5,.75,1.)
    assert result['policy']['selection_metrics']['tp'] == 1
    assert result['policy']['selection_metrics']['fp'] == 0
    assert len(result['by_weight']) == 5


def test_final_threshold_is_median_for_same_weight_not_mixed_winners():
    rows=[]
    for winner in (.75,.75,.75,.25,.25):
        rows.append(dict(policy=dict(weight=winner),by_weight={str(w):dict(threshold=.8 if w==.75 else .2) for w in (0.,.25,.5,.75,1.)}))
    result=api().deployment_policy(rows)
    assert result['weight'] == .75 and result['threshold'] == .8
    assert result['thresholds'] == [.8]*5


def test_final_aggregation_rejects_missing_fold_or_nan_threshold():
    with pytest.raises(ValueError): api().deployment_policy([])
    rows=[dict(policy=dict(weight=.25),by_weight={'0.25':dict(threshold=float('nan'))})]*5
    with pytest.raises(ValueError): api().deployment_policy(rows)
