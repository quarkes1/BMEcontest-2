"""Frozen V7 validation must not become another model-selection loop."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest


def test_evaluation_is_blocked_before_any_registration_or_io(tmp_path):
    from scripts.research.representation_v2.transformer.v7_probe import run_evaluate
    with pytest.raises(RuntimeError, match='not frozen'):
        run_evaluate(tmp_path)


def test_frozen_files_must_all_exist_and_remain_unchanged(tmp_path):
    from scripts.research.representation_v2.transformer.v7_contract import seal, verify_seal
    (tmp_path/'policies.json').write_text('{}')
    with pytest.raises(ValueError, match='missing'):
        seal(tmp_path, ['policies.json', 'models.joblib'])
    (tmp_path/'models.joblib').write_bytes(b'local-model')
    seal(tmp_path, ['policies.json', 'models.joblib'])
    verify_seal(tmp_path)
    (tmp_path/'models.joblib').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        verify_seal(tmp_path)


def test_seal_rejects_traversal_and_overwrite(tmp_path):
    from scripts.research.representation_v2.transformer.v7_contract import seal
    with pytest.raises(ValueError):
        seal(tmp_path, ['../outside'])
    (tmp_path/'model').write_text('x')
    seal(tmp_path, ['model'])
    with pytest.raises(RuntimeError, match='exists'):
        seal(tmp_path, ['model'])


def test_subset_filter_removes_heldout_before_eligibility_processing(tmp_path, monkeypatch):
    from scripts.research.representation_v2.transformer.v7_models import SubsetSource
    from src.pipeline.runner import FilesystemDataSource, WindowBatch
    from src.pipeline.event_stack import EventRef
    batch=WindowBatch(np.array([[1.], [2.]]), np.array([1,0]),
                      (EventRef('train',0,10), EventRef('heldout',0,10)))
    monkeypatch.setattr(FilesystemDataSource, '_load_batch', staticmethod(lambda path:batch))
    monkeypatch.setattr(FilesystemDataSource, '_load_micro_batch', staticmethod(lambda path,meta:(batch,0.)))
    source=SubsetSource(tmp_path, {'train'})
    macro=source._load_batch(tmp_path/'shared.npz')
    micro,_=source._load_micro_batch(tmp_path/'shared.npz', {})
    assert [w.sid for w in macro.windows]==['train']
    assert [w.sid for w in micro.windows]==['train']
    np.testing.assert_array_equal(macro.labels,[1])


def test_registration_sessions_come_from_canonical_feature_tables_not_entire_index(tmp_path):
    from scripts.research.representation_v2.transformer.v7_models import canonical_session_ids
    import json
    directory=tmp_path/'cache/slide'; directory.mkdir(parents=True)
    for role,sid in [('train','train'),('meal_train','meal'),('no_meal_train','negative')]:
        np.savez(directory/f'fold0_{role}.npz',wid=np.array([json.dumps([sid,0,10000])]))
    # A missing raw session not in any feature table must not be added from the index.
    assert canonical_session_ids(tmp_path)=={'train','meal','negative'}


def test_primary_gate_cannot_be_replaced_by_better_secondary():
    from scripts.research.representation_v2.transformer.v7_contract import primary_gate
    result=primary_gate({'b':{'f1':.70}, 'c':{'f1':.705}, 't':{'f1':.80}})
    assert result['passed'] is False
    assert result['primary']=='c'
    assert result['secondary_delta']==pytest.approx(.10)


def test_inference_proposals_do_not_depend_on_score_truth(monkeypatch):
    from scripts.research.representation_v2.transformer.v7_models import score_ledger
    from scripts.research.representation_v2.event_probe import runner
    from src.pipeline.runner import OuterTrainOnlyData, WindowBatch
    from src.pipeline.event_stack import EventRef, MultiScaleCandidate
    event=EventRef('v',0,10000)
    batch=WindowBatch(np.array([[.2,.4]]),np.array([0]),(event,))
    data=OuterTrainOnlyData(batch,batch,(),{'v':'person'}, {'v':(0,10000)},batch,batch)
    class Model:
        def predict_proba(self,x): return np.tile([.3,.7],(len(x),1))
        classes_=np.array([0,1])
    monkeypatch.setattr(runner,'density_candidates',lambda *args:[MultiScaleCandidate(event,None,None)])
    monkeypatch.setattr(runner,'micro_candidates',lambda *args:[])
    monkeypatch.setattr(runner,'union_candidates',lambda left,right:left+right)
    monkeypatch.setattr(runner,'multiscale_verifier_features',lambda *args,**kwargs:np.zeros((1,116)))
    config=SimpleNamespace(density=None,micro_candidate=None)
    models={'macro':Model(),'micro':Model(),'micro_threshold':.35,'train':['train-person']}
    first=score_ledger(data,config,models)
    second=score_ledger(replace(data,train_truths=(event,)),config,models)
    np.testing.assert_array_equal(first.labels,[0])
    np.testing.assert_array_equal(first.features,second.features)
    assert first.candidates==second.candidates


def test_report_includes_no_gt_fp_and_requires_all_seven_people():
    from scripts.research.representation_v2.transformer.v7_probe import summarize_v7
    from src.pipeline.event_stack import EventRef
    ids=[f'p{i}' for i in range(7)]; mapping={f's{i}':p for i,p in enumerate(ids)}
    truth=EventRef('s0',0,10000); false=EventRef('s6',0,10000)
    predictions={'b':[truth], 'c':[truth,false], 't':[truth]}
    summary=summarize_v7(predictions,[truth],mapping,ids,[truth])
    assert summary['events']['c']['fp']==1
    assert summary['subject_diagnostics']['c']['no_gt_subjects']==6
    assert summary['subject_diagnostics']['c']['no_gt_fp']==1
    assert summary['short_events']['c']['n_tp']==1
    assert not summary['gate']['passed']
    with pytest.raises(ValueError, match='seven'):
        summarize_v7(predictions,[truth],{'s0':'p0'},ids,[truth])


def test_reused_oof_cannot_include_its_scoring_subject_in_ancestors():
    from scripts.research.representation_v2.transformer.v7_contract import validate_oof
    row={'train':['train'], 'score':['val'], 'groups':['val'],
         'excluded':['outer'], 'candidates':[{}], 'labels':[1], 'motion_valid':[True],
         'scores':{'b':[.5],'c':[.6],'t':[.6]}, 'components':{'c':[.9],'t':[.9]},
         'graph':{'source':{'fit_subjects':['fd'], 'parents':[]},
                  'c':{'fit_subjects':['val'],'parents':['source']},
                  'score':{'fit_subjects':[],'parents':['c']}}, 'outer_calls':0}
    with pytest.raises(ValueError, match='held-out subject'):
        validate_oof([row], {'train','val'}, {'fd'}, {'outer'}, require_three=False)


def test_existing_validation_attempt_is_not_replaced(tmp_path):
    from scripts.research.representation_v2.transformer.v7_probe import run_register
    from scripts.research.representation_v2.transformer.v7_contract import OUTPUT
    (tmp_path/OUTPUT).mkdir(parents=True)
    with pytest.raises(RuntimeError, match='exists'):
        run_register(tmp_path)


def test_v7_budget_counts_active_training_before_completion(tmp_path):
    from scripts.research.representation_v2.transformer.v7_probe import V7Resources
    resources=V7Resources(tmp_path,tmp_path,new=True)
    resources.row['neural_seconds']=7199.
    with pytest.raises(RuntimeError, match='V7 resource'):
        resources.check(active_neural_seconds=2.)


@pytest.mark.parametrize('tamper',[False,True])
def test_partial_or_tampered_freeze_is_rejected_before_model_deserialization(tmp_path,monkeypatch,tamper):
    from scripts.research.representation_v2.transformer import v7_probe
    from scripts.research.representation_v2.transformer.v7_contract import OUTPUT,seal
    output=tmp_path/OUTPUT; output.mkdir(parents=True)
    names=list(v7_probe.FROZEN_FILES) if tamper else ['models.joblib']
    for name in names: (output/name).write_bytes(b'local trusted evidence')
    seal(output,names)
    if tamper: (output/'models.joblib').write_bytes(b'altered model')
    def forbidden(*args,**kwargs):
        raise AssertionError('deserialization/raw runtime reached before integrity gate')
    monkeypatch.setattr(v7_probe.joblib,'load',forbidden)
    monkeypatch.setattr(v7_probe.torch,'load',forbidden)
    monkeypatch.setattr(v7_probe,'_runtime',forbidden)
    with pytest.raises(ValueError,match='changed|incomplete'):
        v7_probe.run_evaluate(tmp_path)
