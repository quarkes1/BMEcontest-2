"""Deployment refits preserve the sampled-window and subject-OOF contracts."""
import json
import numpy as np
import pytest

from src.pipeline.event_stack import EventRef
from src.pipeline.runner import WindowBatch


def fixture_batch():
    sessions=['a']*12+['b']*5
    labels=np.array([1]+[0]*10+[-1]+[0]*5,dtype=np.int8)
    return WindowBatch(np.arange(34).reshape(17,2),labels,
        tuple(EventRef(s,i*100,(i+1)*100) for i,s in enumerate(sessions)))


def test_single_sampling_preserves_positives_caps_negatives_and_drops_gray():
    from scripts.research.official_recovery.consistent_refit import sample_batch
    raw=fixture_batch();batch,stats=sample_batch(raw)
    assert len(batch.windows)==5
    assert np.count_nonzero(batch.labels==1)==1 and np.count_nonzero(batch.labels==0)==4
    assert sum(w.sid=='a' for w in batch.windows)==4
    assert sum(w.sid=='b' for w in batch.windows)==1
    assert stats['positive_ratio']==.2 and stats['rows']==5
    again,other=sample_batch(raw)
    assert again.windows==batch.windows and other['sample_identity_sha256']==stats['sample_identity_sha256']
    assert len(raw.windows)==17 and raw.labels[11]==-1


def test_sampler_rejects_duplicate_windows_or_unsupported_labels():
    from scripts.research.official_recovery.consistent_refit import sample_batch
    raw=fixture_batch()
    duplicate=WindowBatch(raw.features,raw.labels,(raw.windows[1],)+raw.windows[1:])
    with pytest.raises(ValueError,match='duplicate'): sample_batch(duplicate)
    labels=raw.labels.copy();labels[0]=2
    with pytest.raises(ValueError,match='label'): sample_batch(WindowBatch(raw.features,labels,raw.windows))


def test_failed_validation_cannot_authorize_refitting(tmp_path):
    from scripts.research.official_recovery.consistent_refit import validated_gate
    (tmp_path/'summary.json').write_text(json.dumps({'state':'COMPLETE','passed':False}))
    with pytest.raises(ValueError,match='gate'): validated_gate(tmp_path,tmp_path)


def test_policy_aggregation_uses_only_frozen_fields_not_fold_performance():
    from scripts.research.official_recovery.consistent_refit import aggregate_policy
    settings={str(i):dict(policy=dict(admission=dict(nms_iou=.3,threshold=.2,max_candidates_per_subject=6),
        threshold=t),micro_threshold=.1,blend=.25,verifier_c=.1) for i,t in enumerate([.5,.6,.7,.8,.9])}
    p=aggregate_policy(settings)
    assert p['threshold']==.7 and p['blend_weight']==.25 and p['max_candidates_per_subject']==6
    assert p['micro_threshold']==.1 and p['admission_threshold']==.2


@pytest.mark.parametrize('empty',[False,True])
def test_atomic_directory_publication_never_replaces_existing_destination(tmp_path,empty):
    from scripts.research.official_recovery.consistent_refit import install_new_directory
    source=tmp_path/'staging';source.mkdir();(source/'new.txt').write_text('new')
    target=tmp_path/'destination';target.mkdir()
    if not empty: (target/'old.txt').write_text('old')
    with pytest.raises(OSError): install_new_directory(source,target)
    assert source.is_dir() and not (target/'new.txt').exists()
    if not empty: assert (target/'old.txt').read_text()=='old'


def test_atomic_directory_publication_moves_a_new_complete_tree(tmp_path):
    from scripts.research.official_recovery.consistent_refit import install_new_directory
    source=tmp_path/'staging';source.mkdir();(source/'model.txt').write_text('model')
    target=tmp_path/'destination'
    install_new_directory(source,target)
    assert not source.exists() and (target/'model.txt').read_text()=='model'
