"""A5: reject contaminated/degenerate partitions, not merely different seeds."""
import pytest


def subjects():
    return tuple(f'p{i:02d}' for i in range(24))


def history():
    s = subjects()
    return tuple((tuple(x for x in s if x not in s[k*8:(k+1)*8]), s[k*8:(k+1)*8]) for k in range(3))


def test_registered_partitions_are_deterministic_balanced_and_label_free():
    from scripts.research.representation_v2 import validation_probe as vp
    repeats = vp.registered_partitions(subjects(), history())
    assert repeats == vp.registered_partitions(tuple(reversed(subjects())), history())
    assert len(repeats) == 2
    for folds in repeats:
        assert sorted(s for _, h in folds for s in h) == list(subjects())
        assert all(len(t) == 16 and len(h) == 8 and not set(t) & set(h) for t, h in folds)


@pytest.mark.parametrize('s', [(), subjects()[:-1], subjects()[:-1]+('p00',), subjects()[:-1]+('',)])
def test_invalid_population_is_not_silently_adjusted(s):
    from scripts.research.representation_v2 import validation_probe as vp
    with pytest.raises(ValueError):
        vp.registered_partitions(s, history())


def test_equivalent_historical_partition_rejected_even_if_folds_reordered():
    from scripts.research.representation_v2 import validation_probe as vp
    repeats = vp.registered_partitions(subjects(), history())
    with pytest.raises(ValueError, match='equivalent'):
        vp.registered_partitions(subjects(), tuple(reversed(repeats[0])))


def test_identity_overlap_is_rejected():
    from scripts.research.representation_v2 import validation_probe as vp
    with pytest.raises(ValueError, match='overlap'):
        vp.validate_identities({'t24': list(subjects()), 'v7': ['p00'], 'outer': ['x'],
                                'source_train': ['FD-R-1'], 'source_val': ['FD-R-2']})


def test_manifest_hash_change_blocks_loading(tmp_path):
    from scripts.research.representation_v2 import validation_probe as vp
    out = tmp_path / vp.OUTPUT
    vp.ep.write_json(out/'manifest.json', {'id': vp.IDENTITY})
    vp.ep.write_json(out/'plan.json', {'id': vp.IDENTITY, 'manifest_sha256': '0'*64})
    with pytest.raises(RuntimeError, match='manifest'):
        vp.load_manifest(tmp_path)


def frozen_fixture(tmp_path):
    from scripts.research.representation_v2 import validation_probe as vp
    manifest={'wrapper_ids':list(range(6))}
    vp.ep.write_json(tmp_path/'manifest.json',manifest)
    vp.ep.write_json(tmp_path/'plan.json',{'id':vp.IDENTITY})
    for k in range(6):
        vp.ep.write_json(tmp_path/f'wrapper{k}_policy_selection.json',
                         {'wrapper':k,'policies':{a:{'threshold':.5} for a in ('b','r','f')}})
    return manifest


def test_all_six_policies_required_before_health(tmp_path):
    from scripts.research.representation_v2 import validation_probe as vp
    with pytest.raises(RuntimeError):vp.freeze_policies(tmp_path,{'wrapper_ids':list(range(6))})
    manifest=frozen_fixture(tmp_path)
    sha=vp.freeze_policies(tmp_path,manifest)
    assert set(vp.load_all_policies(tmp_path,sha,manifest))==set(range(6))
    vp.ep.write_json(tmp_path/'wrapper5_policy_selection.json',{'wrapper':5,'policies':{}})
    with pytest.raises(RuntimeError):vp.load_all_policies(tmp_path,sha,manifest)


def test_freeze_marker_and_manifest_tampering_are_rejected(tmp_path):
    from scripts.research.representation_v2 import validation_probe as vp
    manifest=frozen_fixture(tmp_path)
    sha=vp.freeze_policies(tmp_path,manifest)
    vp.ep.write_json(tmp_path/'manifest.json',{'wrapper_ids':[]})
    with pytest.raises(RuntimeError):vp.load_all_policies(tmp_path,sha,manifest)


def test_resource_guard_rejects_time_and_output_overruns(tmp_path,monkeypatch):
    from scripts.research.representation_v2 import validation_probe as vp
    monkeypatch.setattr(vp.shutil,'disk_usage',lambda p:type('D',(),{'free':30*1024**3})())
    with pytest.raises(RuntimeError):vp.check_resources(tmp_path,tmp_path,vp.time.perf_counter()-28801,0,None)
    with pytest.raises(RuntimeError):vp.check_resources(tmp_path,tmp_path,vp.time.perf_counter(),14401,None)
    vp.check_resources(tmp_path,tmp_path,vp.time.perf_counter(),0,None)


def test_fit_accounting_rejects_missing_split_or_steps():
    from scripts.research.representation_v2 import validation_probe as vp
    with pytest.raises(RuntimeError):vp.validate_fit_counts({'splits':23,'producer_fits':240,'verifier_fits':144,
                                                           'readout_fits':24,'steps':24000})


def test_execution_freezes_every_policy_before_first_health(tmp_path):
    from scripts.research.representation_v2 import validation_probe as vp
    manifest=frozen_fixture(tmp_path)
    manifest['wrappers']=[{'id':k} for k in range(6)]
    calls=[]
    def policy(w):
        calls.append(('policy',w['id']))
        return {'wrapper':w['id'],'policies':{a:{'threshold':.5} for a in ('b','r','f')}}
    def health(w,p):
        calls.append(('health',w['id']))
        return {'wrapper':w['id']}
    records,_=vp.execute_phases(tmp_path,manifest,policy,health)
    assert calls==[('policy',k) for k in range(6)]+[('health',k) for k in range(6)]
    assert len(records)==6


def test_new_replay_rejects_invalid_score_and_heldout_ancestor():
    from scripts.research.representation_v2 import validation_probe as vp
    row={'train':['a'],'score':['b'],'excluded':['c'],'candidates':[{}], 'groups':['b'],
         'labels':[1], 'scores':{a:[float('nan')] for a in ('b','r','f')},
         'graph':{'score':{'fit_subjects':[],'parents':[]}}}
    with pytest.raises(ValueError):vp.validate_split_evidence('.', 'x',row)


@pytest.mark.parametrize('deltas,want',[([.006,.004],'SUPPORTED_NEXT_DESIGN'),([.02,-.001],'UNSTABLE_GAIN'),
                                     ([0.,0.],'NO_SUPPORTED_GAIN'),([.001,.002],'SMALL_CONSISTENT_GAIN')])
def test_gain_classification_does_not_require_beating_other_challenger(deltas,want):
    from scripts.research.representation_v2 import validation_metrics as vm
    assert vm.classify_gain(deltas)==want


def test_matching_ties_short_meals_and_empty_subject_keep_denominator():
    from scripts.research.representation_v2 import validation_metrics as vm
    from src.pipeline.event_stack import EventRef, compute_event_metrics
    truth=(EventRef('s',0,10000),EventRef('s',0,10000),EventRef('empty',0,900000))
    pred=(EventRef('s',0,10000),EventRef('s',0,10000),EventRef('other',0,10000))
    pi,ti=vm.matched_indices(pred,truth)
    assert pi=={0,1} and ti=={0,1} and len(pi)==compute_event_metrics(pred,truth).n_tp
    row=vm.make_record({'id':0,'repeat':0,'fold':0}, {a:pred for a in ('b','r','f')},truth,
                       {'s':'a','empty':'b','other':'a'}, [], {a:[] for a in ('b','r','f')})
    empty=next(r for r in row['subject_events'] if r['subject']=='b')
    assert empty['events']['b']['n_true']==1 and empty['events']['b']['n_tp']==0
    assert row['short_events']['b']=={'n_true':2,'n_tp':2}
    assert row['candidate_metrics']['b']['ap'] is None


def test_repeat_mean_is_not_pooled_duplicate_truth_f1():
    from scripts.research.representation_v2 import validation_metrics as vm
    from src.pipeline.event_stack import EventRef
    records=[]
    for k in range(6):
        truths=[EventRef(f's{k%3}_{j}',0,10000) for j in range(8)]
        pred=truths[:(8 if k<3 else 4)]
        preds={'b':pred,'r':pred,'f':pred}
        records.append(vm.make_record({'id':k,'repeat':k//3,'fold':k%3},preds,truths,
            {t.sid:f'p{k%3*8+j}' for j,t in enumerate(truths)},[],{a:[] for a in ('b','r','f')}))
    summary=vm.summarize_repeats(records)
    assert summary['mean_f1']['b']==pytest.approx((1+2/3)/2)
    assert summary['unique_subjects']==24
    result=vm.complementarity(records)
    assert len(result['repeats'])==2
    assert len(result['repeats'][0]['gt_outcomes'])==24
    with pytest.raises(ValueError):vm.summarize_repeats(records[:-1])


def test_failed_policy_never_reaches_health(tmp_path):
    from scripts.research.representation_v2 import validation_probe as vp
    manifest=frozen_fixture(tmp_path)
    manifest['wrappers']=[{'id':k} for k in range(6)]
    def policy(w):
        if w['id']==2:raise RuntimeError('policy failure')
        return {'wrapper':w['id'],'policies':{a:{'threshold':.5} for a in ('b','r','f')}}
    def health(*args):raise AssertionError('health must remain sealed')
    with pytest.raises(RuntimeError,match='policy failure'):vp.execute_phases(tmp_path,manifest,policy,health)


def test_marker_tamper_and_cuda_budget_fail_closed(tmp_path,monkeypatch):
    from scripts.research.representation_v2 import validation_probe as vp
    manifest=frozen_fixture(tmp_path)
    sha=vp.freeze_policies(tmp_path,manifest)
    vp.ep.write_json(tmp_path/'policy_phase_frozen.json',{})
    with pytest.raises(RuntimeError):vp.load_all_policies(tmp_path,sha,manifest)
    monkeypatch.setattr(vp.shutil,'disk_usage',lambda p:type('D',(),{'free':30*1024**3})())
    monkeypatch.setattr(vp.torch.cuda,'max_memory_allocated',lambda:1024**3+1)
    with pytest.raises(RuntimeError,match='CUDA'):vp.check_resources(tmp_path,tmp_path,vp.time.perf_counter(),0,1280*1024**2)


def test_candidate_score_nonfinite_or_wrong_owner_rejected():
    from scripts.research.representation_v2 import validation_probe as vp
    row={'train':['a'],'score':['b'],'excluded':['c'],'candidates':[{}],'groups':['b'],'labels':[1],
         'scores':{a:[.5] for a in ('b','r','f')},
         'graph':{'score':{'fit_subjects':['b'],'parents':[]}}}
    with pytest.raises(ValueError,match='held-out'):vp.validate_split_evidence('.', 'x',row)


def test_actual_116d_baseline_is_existing_half_blend():
    import numpy as np
    from scripts.research.representation_v2 import validation_probe as vp
    rng=np.random.default_rng(64)
    x=rng.normal(size=(80,116));y=(x[:,0]>0).astype(int);v=x[:5]
    ps=[]
    for model in (vp.ep.runner._verifier_estimator(23,.1),vp.ep.runner._verifier_lgbm_estimator(24)):
        model.fit(x,y)
        ps.append(vp.ep.runner._positive_probability(model,v))
    np.testing.assert_allclose(vp.ep.verifier_probabilities(x,y,v,seed=3),.5*ps[0]+.5*ps[1],atol=0,rtol=0)


def test_preflight_manifest_failure_leaves_incomplete_not_ready(tmp_path,monkeypatch):
    import json
    from scripts.research.representation_v2 import validation_probe as vp
    out=tmp_path/vp.OUTPUT
    vp.ep.write_json(out/'manifest.json',{'id':vp.IDENTITY})
    vp.ep.write_json(out/'plan.json',{'id':vp.IDENTITY,'manifest_sha256':'0'*64})
    vp.ep.write_json(out/'state.json',{'status':'S15_MANIFEST_READY'})
    monkeypatch.setattr(vp,'require_clean',lambda root:None)
    with pytest.raises(RuntimeError,match='manifest'):vp.run_event(tmp_path)
    assert json.loads((out/'state.json').read_text())['status']=='S15_INCOMPLETE'


@pytest.mark.parametrize('failure',['state','cuda'])
def test_startup_failure_is_incomplete_even_before_training(tmp_path,monkeypatch,failure):
    import json
    from scripts.research.representation_v2 import validation_probe as vp
    out=tmp_path/vp.OUTPUT
    vp.ep.write_json(out/'plan.json',{})
    if failure=='state':(out/'state.json').write_text('{broken',encoding='utf-8')
    else:vp.ep.write_json(out/'state.json',{'status':'S15_MANIFEST_READY'})
    monkeypatch.setattr(vp,'require_clean',lambda root:None)
    monkeypatch.setattr(vp,'load_manifest',lambda root:{})
    monkeypatch.setattr(vp.ep,'protected_snapshot',lambda root:{})
    monkeypatch.setattr(vp.torch.cuda,'is_available',lambda:True)
    def broken_cuda():raise RuntimeError('CUDA startup failure')
    monkeypatch.setattr(vp.torch.cuda,'mem_get_info',broken_cuda)
    with pytest.raises((RuntimeError,json.JSONDecodeError)):vp.run_event(tmp_path)
    assert json.loads((out/'state.json').read_text())['status']=='S15_INCOMPLETE'
