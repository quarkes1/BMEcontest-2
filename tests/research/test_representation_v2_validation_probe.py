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
