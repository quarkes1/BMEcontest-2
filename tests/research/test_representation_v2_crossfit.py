"""An inner score's candidate producers must exclude its subjects."""

import numpy as np
import pytest
from dataclasses import replace

from scripts.research.representation_v2.crossfit import build_candidate_ledgers, inner_subject_partitions, slice_outer_train
from src.pipeline.challenger.e03 import registered_e03_config
from src.pipeline.event_stack import EventRef
from src.pipeline.runner import OuterTrainOnlyData, WindowBatch


def _batch(sids):
    windows = tuple(EventRef(sid, i * 10_000, (i + 1) * 10_000) for i, sid in enumerate(sids))
    return WindowBatch(np.zeros((len(sids), 2)), np.asarray([i % 2 for i in range(len(sids))]), windows)


def _data():
    sids = tuple(f"sid{i}" for i in range(8))
    return OuterTrainOnlyData(
        window_train=_batch(sids), candidate_train=_batch(sids),
        train_truths=(EventRef("sid0", 0, 10_000), EventRef("sid7", 70_000, 80_000)),
        subject_by_session={sid: f"person{i}" for i, sid in enumerate(sids)},
        session_bounds_by_sid={sid: (0, 100_000) for sid in sids},
        micro_window_train=_batch(sids), micro_candidate_train=_batch(sids),
    )


def test_slice_removes_held_out_subject_from_every_train_input():
    sliced = slice_outer_train(_data(), {"person0", "person1"})
    assert {x.sid for x in sliced.window_train.windows} == {"sid0", "sid1"}
    assert {x.sid for x in sliced.candidate_train.windows} == {"sid0", "sid1"}
    assert {x.sid for x in sliced.micro_window_train.windows} == {"sid0", "sid1"}
    assert {x.sid for x in sliced.micro_candidate_train.windows} == {"sid0", "sid1"}
    assert [x.sid for x in sliced.train_truths] == ["sid0"]
    assert set(sliced.subject_by_session) == {"sid0", "sid1"}


def test_inner_partitions_are_subject_disjoint_and_cover_every_subject():
    folds = inner_subject_partitions(_data(), n_splits=4)
    assert len(folds) == 4
    assert set.union(*(set(score) for _, score in folds)) == {f"person{i}" for i in range(8)}
    assert all(not set(train) & set(score) for train, score in folds)


def test_slice_rejects_unknown_subject():
    with pytest.raises(ValueError, match="unknown"):
        slice_outer_train(_data(), {"not-present"})


def test_candidate_producer_rejects_overlapping_subject_ancestry():
    config = replace(registered_e03_config(0, workers=1), micro_positive_middle_fraction=None)
    with pytest.raises(ValueError, match="overlap"):
        build_candidate_ledgers(_data(), config, {"person0", "person1"}, {"person1"})


def test_candidate_producer_never_uses_score_truth_for_features():
    data = _data()
    config = replace(registered_e03_config(0, workers=1), micro_positive_middle_fraction=None)
    train = {f"person{i}" for i in range(6)}
    score = {"person6", "person7"}
    audit = {}
    left_t, left_v = build_candidate_ledgers(data, config, train, score, fit_audit=audit)
    assert len([k for k in audit if k.endswith('refit')]) == 2
    assert set(audit['macro_refit']['fit_subjects']) == train
    assert set(audit['threshold']['fit_subjects']) == {'person0'}
    from scripts.research.representation_v2.provenance import assert_no_heldout_ancestors
    assert_no_heldout_ancestors('candidates', score, audit)
    changed = replace(data, train_truths=tuple(truth for truth in data.train_truths if truth.sid != "sid7"))
    right_t, right_v = build_candidate_ledgers(changed, config, train, score)
    assert left_t.fit_subjects == tuple(sorted(train))
    assert left_v.score_subjects == tuple(sorted(score))
    assert not set(left_v.fit_subjects) & set(left_v.score_subjects)
    assert left_v.candidates == right_v.candidates
    np.testing.assert_array_equal(left_v.features, right_v.features)
    assert left_v.micro_threshold == right_v.micro_threshold


def test_training_only_ledger_matches_oof_without_validation_refit():
    data = _data()
    config = replace(registered_e03_config(0, workers=1), micro_positive_middle_fraction=None)
    train = {f'person{i}' for i in range(6)}
    reference, _ = build_candidate_ledgers(data, config, train, {'person6', 'person7'})
    audit = {}
    actual, validation = build_candidate_ledgers(data, config, train, (), fit_audit=audit)
    assert validation is None
    assert not any(name.endswith('refit') for name in audit)
    assert actual.candidates == reference.candidates
    np.testing.assert_array_equal(actual.features, reference.features)
    np.testing.assert_array_equal(actual.labels, reference.labels)
    assert actual.micro_threshold == reference.micro_threshold
    assert all(set(node['fit_subjects']) <= train for node in audit.values())
