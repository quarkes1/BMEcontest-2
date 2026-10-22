import numpy as np
import pytest


def test_relative_formula_is_fixed_and_subject_local_with_finite_endpoints():
    from scripts.research.official_recovery.relative_score_probe import relative_scores
    p=np.array([.2,.8,.99]);groups=['A','A','B']
    r=relative_scores(p,groups)
    scale=np.sqrt(.74/.26)
    np.testing.assert_allclose(r[:2],[.2/(.2+.8*scale),.8/(.8+.2*scale)],rtol=1e-12)
    np.testing.assert_allclose(relative_scores([.8,.2,.01],['A','A','B'])[:2],r[[1,0]])
    assert relative_scores([],[]).size==0
    bounds=relative_scores([0.,1.],['Z','Z'])
    assert np.isfinite(bounds).all() and np.all((bounds>0)&(bounds<1))
    for values,g in (([np.nan],['A']),([1.1],['A']),([.1],[]),([[.1]],['A'])):
        with pytest.raises(ValueError): relative_scores(values,g)


def test_original_raw_probability_controls_admission_not_relative_score():
    from scripts.research.official_recovery.relative_score_probe import decode_relative,choose_threshold
    from src.pipeline.event_stack import EventRef,MultiScaleCandidate
    cs=[MultiScaleCandidate(EventRef('s',0,100),None,None)]
    admission=dict(nms_iou=.3,threshold=.42,max_candidates_per_subject=6)
    policy=dict(admission=admission,threshold=.4)
    # p=.4 fails original admission; relative score .449 passes .42, but must not be used there.
    assert decode_relative(cs,np.array([.4]),['A'],policy)==[]
    selected=choose_threshold(cs,np.array([.4]),['A'],admission,{'s':'A'},[('A',0,100)])
    assert selected['selection_metrics']['tp']==0 and selected['admission']==admission


def test_complete_threshold_has_strict_iou_and_counts_cross_sid_truth_once():
    from scripts.research.official_recovery.relative_score_probe import choose_threshold
    from src.pipeline.event_stack import EventRef,MultiScaleCandidate
    cs=[MultiScaleCandidate(EventRef(s,0,100),None,None) for s in ('a','b')]
    admission=dict(nms_iou=1.,threshold=0.,max_candidates_per_subject=None)
    result=choose_threshold(cs,np.array([.8,.8]),['A','A'],admission,{'a':'A','b':'A'},[('A',0,100)])
    assert result['selection_metrics']['truths']==1 and result['selection_metrics']['tp']==1
    boundary=choose_threshold(cs,np.array([.8,.8]),['A','A'],admission,{'a':'A','b':'A'},[('A',0,400)])
    assert boundary['selection_metrics']['tp']==0


def test_relative_gate_and_parent_identity_are_explicit(tmp_path):
    import json
    from scripts.research.official_recovery.relative_score_probe import gate,verify_parent
    rows=[dict(fold=i,events={'A':dict(tp=20,predictions=30,truths=n),
        'G':dict(tp=20,predictions=28,truths=n)}) for i,n in enumerate((54,55,51))]
    assert gate(rows,'t24')['passed']
    parent=tmp_path/'parent';parent.mkdir()
    (parent/'plan.json').write_text(json.dumps({'id':'OTHER','cohort':'t24'}))
    (parent/'state.json').write_text(json.dumps({'state':'COMPLETE'}))
    (parent/'summary.json').write_text(json.dumps({'passed':True}))
    with pytest.raises(ValueError): verify_parent(parent)


def test_relative_failed_preflight_closes_without_overwriting(tmp_path):
    from scripts.research.official_recovery.relative_score_probe import execute
    from scripts.research.official_recovery.negative_clean_probe import _read
    with pytest.raises(Exception): execute(tmp_path,tmp_path/'output')
    assert _read(tmp_path/'output/state.json')['state']=='INCOMPLETE'
    with pytest.raises(FileExistsError): execute(tmp_path,tmp_path/'output')


def test_recall_objective_changes_only_final_raw_score_threshold():
    from scripts.research.official_recovery.relative_score_probe import choose_threshold
    from src.pipeline.event_stack import EventRef,MultiScaleCandidate
    cs=[MultiScaleCandidate(EventRef('s',i*200,i*200+100),None,None) for i in range(3)]
    admission=dict(nms_iou=.3,threshold=.2,max_candidates_per_subject=6)
    # High score TP; second FP; last TP. Both objectives prefer both TP here.
    kwargs=dict(candidates=cs,raw=np.array([.9,.7,.6]),groups=['A']*3,admission=admission,
        mapping={'s':'A'},truths=[('A',0,100),('A',400,500)])
    f1=choose_threshold(**kwargs,beta=1.,use_relative=False)
    recall=choose_threshold(**kwargs,beta=1.5,use_relative=False)
    assert f1['threshold']==.6 and recall['threshold']==.6
    # Add another FP: F1 ties at 2/3, preferring the precise .9 threshold; recall accepts .5.
    cs.append(MultiScaleCandidate(EventRef('s',600,700),None,None))
    kwargs.update(candidates=cs,raw=np.array([.9,.7,.6,.5]),groups=['A']*4,truths=[('A',0,100),('A',600,700)])
    f1=choose_threshold(**kwargs,beta=1.,use_relative=False)
    recall=choose_threshold(**kwargs,beta=1.5,use_relative=False)
    assert f1['threshold']==.9 and recall['threshold']==.5
    assert recall['selection_metrics']['tp']==2 and recall['admission']==admission


def test_recall_trial_gate_accepts_bounded_fp_tradeoff_but_not_formal_improvement():
    from scripts.research.official_recovery.relative_score_probe import gate,mode_contract
    rows=[dict(fold=i,events={'A':dict(tp=20,predictions=30,truths=n),
        'G':dict(tp=22,predictions=36,truths=n)}) for i,n in enumerate((54,55,51))]
    assert gate(rows,'t24','recall125')['passed']
    assert gate(rows,'t24','recall150')['passed']
    assert not gate(rows,'t24')['passed']
    assert mode_contract('recall125')['beta']==1.25 and mode_contract('recall150')['beta']==1.5
    assert not mode_contract('recall125')['use_relative']
    with pytest.raises(ValueError): mode_contract('recall200')


def test_deployment_modes_change_reference_and_keep_fixed_objectives():
    from scripts.research.official_recovery.relative_score_probe import mode_contract
    f1=mode_contract('deploy_f1');f15=mode_contract('deploy_f15')
    assert f1['deployment_reference'] and f15['deployment_reference']
    assert f1['beta']==1. and f15['beta']==1.5 and not f1['use_relative']


def test_deployment_reference_is_exact_accepted_bundle_and_rejects_modified_copy(tmp_path):
    from pathlib import Path
    import shutil
    from scripts.research.official_recovery.relative_score_probe import deployment_reference
    root=Path(__file__).resolve().parents[2]
    bundle='models/event_stack/official_consistent_refit_v1/deployment'
    report='outputs/official_recovery/consistent_refit_v1/summary.json'
    if not (root/bundle).is_dir(): pytest.skip('accepted first trial bundle unavailable')
    shutil.copytree(root/bundle,tmp_path/bundle)
    (tmp_path/report).parent.mkdir(parents=True);shutil.copyfile(root/report,tmp_path/report)
    policy,evidence=deployment_reference(tmp_path)
    assert policy['threshold']==.6595272837359293
    assert policy['admission']==dict(nms_iou=.3,threshold=.2,max_candidates_per_subject=6)
    assert len(evidence['source_sha256'])==9
    (tmp_path/bundle/'policy.json').write_text('{}')
    with pytest.raises(ValueError): deployment_reference(tmp_path)


@pytest.fixture
def sealed_probe_fixture(tmp_path,monkeypatch):
    from scripts.research.official_recovery import relative_score_probe as probe
    from scripts.research.official_recovery.negative_clean_probe import _write
    from src.pipeline.event_stack import EventRef
    root=tmp_path;subjects=[f'S{i}' for i in range(9)];mapping={f's{i}':s for i,s in enumerate(subjects)}
    truths=[];wrappers=[];manifest=dict(wrappers=[],identities=dict(v7=[],outer=[]))
    for fold,total in enumerate((54,55,51)):
        score=set(subjects[fold*3:fold*3+3]);train=set(subjects)-score
        truths.extend((sorted(score)[j%3],(j//3)*200,(j//3)*200+100) for j in range(total))
        sid=next(k for k,v in mapping.items() if v==sorted(score)[0])
        wrappers.append(dict(id=fold,train=sorted(train),score=sorted(score),health_events=[EventRef(sid,999999,1000000)]))
        partitions=[]
        for i in range(3):
            valid=set(sorted(train)[i*2:i*2+2]);partitions.append(dict(train=sorted(train-valid),score=sorted(valid)))
        manifest['wrappers'].append(dict(id=fold,policy_splits=partitions))
        admission=dict(nms_iou=.3,threshold=.2,max_candidates_per_subject=6)
        source=root/probe.T24_SOURCE;source.mkdir(parents=True,exist_ok=True)
        _write(source/f'fold{fold}_policies.json',dict(policies={'A':dict(admission=admission,threshold=.8)}))
        def score_file(name,fit,valid):
            rows=[[i,next(k for k,v in mapping.items() if v==s),0,100,True,True,s,.8,.8,.8] for i,s in enumerate(sorted(valid))]
            _write(source/name,dict(train_subjects=sorted(fit),score_subjects=sorted(valid),
                columns=['row','sid','start_ms','end_ms','macro','micro','subject','lr','lgbm','score'],rows=rows,
                graph={'score':dict(fit_subjects=sorted(fit),parents=[])}))
        for i,part in enumerate(partitions): score_file(f'fold{fold}_policy{i}_A_scores.json',set(part['train']),set(part['score']))
        score_file(f'fold{fold}_health_A_scores.json',train,score)
    for name in (probe.SPEC,probe.RECALL_SPEC):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('registered')
    path=root/probe.MANIFEST;path.parent.mkdir(parents=True,exist_ok=True);_write(path,manifest)
    external=root/'input.txt';external.write_text('frozen')
    protected=root/'protected.bin';protected.write_bytes(b'original')
    monkeypatch.setattr(probe,'inputs',lambda *a,**k:{'input.txt':probe._sha(external)})
    monkeypatch.setattr(probe,'protected_sources',lambda r:{'protected.bin':probe._sha(protected)})
    monkeypatch.setattr(probe,'load_context',lambda *a,**k:(truths,mapping,{},wrappers))
    monkeypatch.setattr(probe.subprocess,'check_output',lambda args,**kwargs:'' if args[1]=='status' else 'revision')
    path=root/probe.DEPLOY_SPEC;path.write_text('registered')
    reference=dict(admission=dict(nms_iou=.3,threshold=.2,max_candidates_per_subject=6),threshold=.99)
    monkeypatch.setattr(probe,'deployment_reference',lambda r:(reference,dict(source_sha256={'input.txt':probe._sha(external)})))
    return probe,root,external,protected


@pytest.mark.parametrize('mode',['relative','recall125','recall150','deploy_f1','deploy_f15'])
def test_real_policy_seal_score_and_mode_parent_reject_changed_report(sealed_probe_fixture,mode):
    probe,root,_,_=sealed_probe_fixture
    output=root/probe.mode_contract(mode)['t24']
    assert probe.execute(root,output,mode=mode)['state']=='POLICY_FROZEN'
    result=probe.execute(root,output,'score',mode=mode)
    assert result['state']=='COMPLETE' and result['passed']
    if mode.startswith('deploy_'):
        assert result['evidence_kind']=='FROZEN_MODEL_DEPLOYMENT_POLICY_DEVELOPMENT_TRIAL_ONLY'
        assert result['actual_exe_cv'] is False
        assert result['aggregate']['A']['fp']==0
        assert probe._read(output/'fold0_report.json')['events']['source_A']['fp']==1
    probe.verify_parent(output,mode)
    other='relative' if mode!='relative' else 'recall125'
    with pytest.raises(ValueError): probe.verify_parent(output,other)
    report=probe._read(output/'fold0_report.json');report['events']['G']['tp']=0
    probe._write(output/'fold0_report.json',report)
    with pytest.raises(ValueError,match='gate'): probe.verify_parent(output,mode)


@pytest.mark.parametrize('drift',['input','protected','seal'])
def test_score_phase_rejects_actual_input_protection_or_seal_drift(sealed_probe_fixture,drift):
    probe,root,external,protected=sealed_probe_fixture;output=root/'evidence'
    probe.execute(root,output)
    if drift=='input': external.write_text('changed')
    elif drift=='protected': protected.write_bytes(b'changed')
    else:
        policy=probe._read(output/'fold0_policy.json');policy['policy']['threshold']=.1
        probe._write(output/'fold0_policy.json',policy)
    with pytest.raises(ValueError): probe.execute(root,output,'score')
    assert probe._read(output/'state.json')['state']=='INCOMPLETE'
