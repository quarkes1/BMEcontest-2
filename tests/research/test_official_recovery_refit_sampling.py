"""A refit must use the declared training role rather than silently ignoring it."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import expit

from src.pipeline.event_stack import EventRef
from src.pipeline.runner import WindowBatch,OuterTrainOnlyData


def fixture():
    rows=tuple(EventRef('s',i*100,(i+1)*100) for i in range(10))
    macro=WindowBatch(np.ones((10,62)),np.array([1]+[0]*9),rows)
    micro=WindowBatch(np.ones((10,47)),np.array([1]+[0]*9),rows)
    return OuterTrainOnlyData(WindowBatch(macro.features[:2],macro.labels[:2],rows[:2]),macro,(),{'s':'A'},
        {'s':(0,1000)},WindowBatch(micro.features[:2],micro.labels[:2],rows[:2]),micro)


def test_sampled_and_full_refits_use_different_training_roles_without_relabeling():
    from scripts.research.official_recovery.refit_sampling_probe import fit_windows
    data=fixture();config=SimpleNamespace(seed=20260908,micro_positive_middle_fraction=None)
    full,_,full_stats=fit_windows(data,config,'full')
    sampled,_,sample_stats=fit_windows(data,config,'sampled')
    assert full_stats['macro']['rows']==10 and sample_stats['macro']['rows']==2
    assert float(expit(full.named_steps['model']._baseline_prediction[0,0]))==pytest.approx(.1)
    assert float(expit(sampled.named_steps['model']._baseline_prediction[0,0]))==pytest.approx(.5)
    assert data.candidate_train.labels.tolist()==[1]+[0]*9


def test_unknown_training_role_is_not_silently_treated_as_full():
    from scripts.research.official_recovery.refit_sampling_probe import fit_windows
    with pytest.raises(ValueError,match='role'):
        fit_windows(fixture(),SimpleNamespace(seed=1,micro_positive_middle_fraction=None),'typo')
