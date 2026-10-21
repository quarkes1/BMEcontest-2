import pytest


def test_window_signal_remains_failed_but_explicit_m1_may_test_same_factory(tmp_path,monkeypatch):
    from scripts.research.official_recovery import macro_lgbm_event as probe
    from scripts.research.official_recovery.scope_audit import _sha
    verify_window_parent=probe.verify_window_parent
    from scripts.research.official_recovery.negative_clean_probe import _write
    _write(tmp_path/'state.json',{'state':'COMPLETE'})
    _write(tmp_path/'summary.json',{'state':'COMPLETE','passed':False,'delta_ap':.018877746493557,'positive_folds':2,'protected_unchanged':True})
    _write(tmp_path/'plan.json',{'id':'MACRO-LGBM-V1-WINDOW','protected':{'x':'y'},'source_sha256':{},'code_sha':'reviewed'})
    monkeypatch.setattr(probe,'WINDOW_PARENT_PLAN_SHA',_sha(tmp_path/'plan.json'),raising=False)
    monkeypatch.setattr(probe,'WINDOW_PARENT_CODE_SHA','reviewed',raising=False)
    verify_window_parent(tmp_path,{'x':'y'})
    with pytest.raises(ValueError): verify_window_parent(tmp_path,{'x':'changed'})
    _write(tmp_path/'state.json',{'state':'INCOMPLETE'})
    with pytest.raises(ValueError): verify_window_parent(tmp_path,{'x':'y'})


def test_window_parent_requires_exact_reviewed_plan_and_live_source(tmp_path,monkeypatch):
    from scripts.research.official_recovery import macro_lgbm_event as probe
    from scripts.research.official_recovery.negative_clean_probe import _write
    metadata=tmp_path/'metadata.csv';metadata.write_text('current')
    _write(tmp_path/'state.json',{'state':'COMPLETE'})
    _write(tmp_path/'summary.json',{'state':'COMPLETE','passed':False,'delta_ap':.018877746493557,'positive_folds':2,'protected_unchanged':True})
    plan={'id':'MACRO-LGBM-V1-WINDOW','protected':{},'source_sha256':{str(metadata):'not-current'},'code_sha':'reviewed'}
    _write(tmp_path/'plan.json',plan)
    monkeypatch.setattr(probe,'WINDOW_PARENT_PLAN_SHA',probe._sha(tmp_path/'plan.json'),raising=False)
    monkeypatch.setattr(probe,'WINDOW_PARENT_CODE_SHA','reviewed',raising=False)
    with pytest.raises(ValueError,match='source'): probe.verify_window_parent(tmp_path,{})
    plan['source_sha256'][str(metadata)]=probe._sha(metadata);_write(tmp_path/'plan.json',plan)
    with pytest.raises(ValueError,match='plan'): probe.verify_window_parent(tmp_path,{})
    monkeypatch.setattr(probe,'WINDOW_PARENT_PLAN_SHA',probe._sha(tmp_path/'plan.json'))
    probe.verify_window_parent(tmp_path,{})


def test_failed_event_preflight_is_incomplete_and_does_not_overwrite(tmp_path):
    from scripts.research.official_recovery.macro_lgbm_event import execute
    from scripts.research.official_recovery.negative_clean_probe import _read
    out=tmp_path/'out'
    with pytest.raises(Exception): execute(tmp_path,out)
    assert _read(out/'state.json')['state']=='INCOMPLETE'
    with pytest.raises(FileExistsError): execute(tmp_path,out)


def test_event_gate_keeps_full_denominator_and_not_ap_surrogate():
    from scripts.research.official_recovery.max_fusion_probe import fusion_gate
    rows=[dict(fold=i,events={'A':dict(tp=20,predictions=30,truths=n),
        'G':dict(tp=20,predictions=30,truths=n)}) for i,n in enumerate((54,55,51))]
    assert not fusion_gate(rows,'t24')['passed']
    for r in rows: r['events']['G']['tp']+=3
    assert fusion_gate(rows,'t24')['passed']
