import pytest

from src.pipeline.event_stack import EventRef


def test_observed_bounds_preserve_early_windows_without_using_predictions():
    from src.pipeline.event_geometry import observed_session_bounds,calibrate_events
    index={'a':(4000,10000)}
    bounds=observed_session_bounds(index,[EventRef('a',731,9000),EventRef('a',5000,11000)])
    assert bounds=={'a':(731,11000)} and index=={'a':(4000,10000)}
    original=[EventRef('a',731,8000)]
    assert calibrate_events(original,bounds)==original
    with pytest.raises(ValueError): observed_session_bounds(index,[EventRef('b',0,10)])
    with pytest.raises(ValueError): observed_session_bounds(index,[EventRef('a',10,10)])


def test_parent_protection_and_scope_metadata_must_match(tmp_path,monkeypatch):
    from scripts.research.official_recovery import geometry_probe as probe
    metadata=tmp_path/'meals.csv';metadata.write_text('frozen')
    snapshot={'dist/trial.exe':'old'}
    monkeypatch.setattr(probe,'protected_sources',lambda root:dict(snapshot))
    parent={'protected':dict(snapshot)}
    scope={'source_sha256':{'meals.csv':probe._sha(metadata)}}
    probe.verify_parent_identity(tmp_path,parent,scope,[metadata])
    snapshot['dist/trial.exe']='new'
    with pytest.raises(ValueError,match='protected'): probe.verify_parent_identity(tmp_path,parent,scope,[metadata])
    snapshot['dist/trial.exe']='old';metadata.write_text('changed')
    with pytest.raises(ValueError,match='metadata'): probe.verify_parent_identity(tmp_path,parent,scope,[metadata])


def test_integer_geometry_identity_order_and_nonmutating_scaling():
    from src.pipeline.event_geometry import calibrate_events
    base=[EventRef('a',100,501),EventRef('b',0,100)]
    bounds={'a':(0,10000),'b':(0,100)}
    assert calibrate_events(base,bounds,(1,1),0)==base
    out=calibrate_events(base,bounds,(1,2),0)
    assert out==[EventRef('a',200,401),EventRef('b',25,75)]
    assert base[0]==EventRef('a',100,501) and len(out)==len(base)


def test_geometry_repositions_at_bounds_without_deleting_events():
    from src.pipeline.event_geometry import calibrate_events
    events=[EventRef('a',100,500),EventRef('a',200,1000)]
    out=calibrate_events(events,{'a':(0,1000)},(3,2),60000)
    assert out==[EventRef('a',400,1000),EventRef('a',0,1000)]
    assert calibrate_events(events,{'a':(0,1000)},(3,2),-60000)==[EventRef('a',0,600),EventRef('a',0,1000)]


def test_geometry_rejects_unregistered_or_invalid_bounds():
    from src.pipeline.event_geometry import calibrate_events
    event=[EventRef('a',100,500)]
    for scale,shift in (((2,1),0),((1,1),1),((True,1),0),((1,1),False)):
        with pytest.raises(ValueError): calibrate_events(event,{'a':(0,1000)},scale,shift)
    with pytest.raises(ValueError): calibrate_events(event,{},(1,1),0)
    with pytest.raises(ValueError): calibrate_events(event,{'a':(200,1000)},(1,1),0)


def test_geometry_selection_uses_truths_only_to_choose_one_global_policy():
    from scripts.research.official_recovery.geometry_probe import choose_geometry
    bounds={'a':(0,3000000)}
    event=[EventRef('a',600000,1200000)]
    # All transformations retain this match; prefer exact identity on a tie.
    policy=choose_geometry(event,bounds,{'a':'A'},[('A',600000,1200000)])
    assert policy=={'scale':[1,1],'shift_ms':0}


def test_geometry_gate_requires_complete_pair_majority_and_practical_gain():
    from scripts.research.official_recovery.geometry_probe import geometry_gate
    rows=[dict(fold=i,events={a:dict(tp=20+(2 if a=='G' else 0),predictions=30,truths=n)
        for a in ('A','G')}) for i,n in enumerate((50,55,55))]
    assert geometry_gate(rows,'t24')['passed']
    with pytest.raises(ValueError): geometry_gate(rows[:2],'t24')
    rows[0]['events']['G']['truths']+=1
    with pytest.raises(ValueError): geometry_gate(rows,'t24')
    rows[0]['events']['G']['truths']-=1
    for r in rows: r['events']['G']['tp']=r['events']['A']['tp']
    assert not geometry_gate(rows,'t24')['passed']


def test_geometry_failed_preflight_does_not_overwrite(tmp_path):
    import json
    from scripts.research.official_recovery.geometry_probe import execute
    out=tmp_path/'out'
    with pytest.raises(Exception): execute(tmp_path,out)
    assert json.loads((out/'state.json').read_text())['state']=='INCOMPLETE'
    old=(out/'state.json').read_bytes()
    with pytest.raises(FileExistsError): execute(tmp_path,out)
    assert (out/'state.json').read_bytes()==old


def test_fivefold_gate_keeps_count_and_requires_three_improved_folds():
    from scripts.research.official_recovery.geometry_probe import geometry_gate
    rows=[dict(fold=i,events={a:dict(tp=20+(2 if a=='G' else 0),predictions=30,truths=n)
        for a in ('A','G')}) for i,n in enumerate((47,51,52,50,54))]
    assert geometry_gate(rows,'five')['passed']
    rows[0]['events']['G']['predictions']+=1
    with pytest.raises(ValueError,match='count'): geometry_gate(rows,'five')
    rows[0]['events']['G']['predictions']-=1
    for i,r in enumerate(rows): r['events']['G']['tp']=30 if i<2 else 20
    assert not geometry_gate(rows,'five')['passed']
