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


def fixed_source(tmp_path):
    import hashlib,json
    source=tmp_path/'source';source.mkdir()
    values={'summary.json':dict(state='COMPLETE',protected_unchanged=True,
        evidence_kind='T24_NESTED_EVENT_COMPLETE_MEAL_DEVELOPMENT_ONLY'),
        'population.json':{},'eligibility_inputs.json':{},
        'fold0_policies.json':{'graph':{'policy':{'fit_subjects':['A'],'parents':[]}}},
        'fold0_health_scores.json':dict(train=['A'],score=['B'],graph={
            'threshold':{'fit_subjects':['A'],'parents':[]}})}
    for name,value in values.items(): (source/name).write_text(json.dumps(value))
    hashes={str((source/name).relative_to(tmp_path)):hashlib.sha256((source/name).read_bytes()).hexdigest() for name in values}
    manifest={'identities':{'v7':['C'],'outer':['D']},'wrappers':[dict(id=0,train=['A'],health=['B'])]}
    return source,hashes,manifest


def test_fixed_source_is_anchored_and_changed_policy_is_rejected(tmp_path):
    from scripts.research.official_recovery.refit_sampling_probe import validate_fixed_source
    source,hashes,manifest=fixed_source(tmp_path)
    assert len(validate_fixed_source(tmp_path,source,hashes,manifest))==5
    (source/'fold0_policies.json').write_text('{}')
    with pytest.raises(ValueError,match='source'):
        validate_fixed_source(tmp_path,source,hashes,manifest)


def test_fixed_source_requires_policy_ancestor_isolation_even_with_matching_hash(tmp_path):
    import json,hashlib
    from scripts.research.official_recovery.refit_sampling_probe import validate_fixed_source
    source,hashes,manifest=fixed_source(tmp_path)
    path=source/'fold0_policies.json';path.write_text(json.dumps({'graph':{'policy':{'fit_subjects':['B'],'parents':[]}}}))
    hashes[str(path.relative_to(tmp_path))]=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        validate_fixed_source(tmp_path,source,hashes,manifest)
