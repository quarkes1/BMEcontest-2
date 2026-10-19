"""Frozen-result audits must not turn evaluation-scope changes into model gains."""
from pathlib import Path

import pytest


def test_strict_matching_excludes_exact_quarter_and_retains_one_to_one():
    from scripts.research.official_recovery.scope_audit import match_events
    predictions = [('A', 0, 25), ('A', 0, 25), ('B', 0, 100)]
    truths = [('A', 0, 100)]
    assert match_events(predictions, truths, strict=True) == []
    assert match_events(predictions, truths, strict=False) == [(0, 0)]


def test_matching_keeps_missing_truths_and_does_not_cross_subjects():
    from scripts.research.official_recovery.scope_audit import event_metrics
    assert event_metrics([('B', 0, 100)], [('A', 0, 100)]) == {
        'tp': 0, 'fp': 1, 'fn': 1, 'predictions': 1, 'truths': 1,
        'recall': 0.0, 'ppv': 0.0, 'f1': 0.0,
    }
    empty = event_metrics([], [('A', 0, 100)])
    assert (empty['tp'], empty['fn'], empty['predictions']) == (0, 1, 0)


def test_normalized_fold_collision_fails_instead_of_leaking_subjects():
    from scripts.research.official_recovery.scope_audit import fold_subjects
    with pytest.raises(ValueError, match='multiple folds'):
        fold_subjects([{ 'fold': 0, 'subjects': ['HNU 01'] },
                       { 'fold': 1, 'subjects': ['hnu01'] }])


def test_normalized_id_aliases_are_rejected_within_a_fold():
    from scripts.research.official_recovery.scope_audit import fold_subjects
    with pytest.raises(ValueError, match='aliases'):
        fold_subjects([{'fold': 0, 'subjects': ['HNU 01', 'hnu01']}])


def test_ledger_identity_checks_missing_and_duplicate_predictions():
    from scripts.research.official_recovery.scope_audit import validate_prediction_rows
    row = {'fold': '0', 'final_index': '0', 'group': 'tp_reference'}
    with pytest.raises(ValueError, match='counts'):
        validate_prediction_rows([row])
    with pytest.raises(ValueError, match='duplicate'):
        validate_prediction_rows([row, row])


def test_complete_truths_keep_uncovered_meals_and_reject_invalid_intervals():
    from scripts.research.official_recovery.scope_audit import prepare_truths
    rows = [dict(subject='a', start_ms=0, end_ms=50),
            dict(subject='a', start_ms=200, end_ms=250),
            dict(subject='z', start_ms=0, end_ms=50)]
    kept, outside = prepare_truths(rows, {'A': 0})
    assert kept == [('A', 0, 50), ('A', 200, 250)]
    assert outside == [('Z', 0, 50)]
    with pytest.raises(ValueError, match='positive'):
        prepare_truths([dict(subject='a', start_ms=4, end_ms=4)], {'A': 0})


def test_duplicate_ground_truth_is_reported_not_silently_dropped():
    from scripts.research.official_recovery.scope_audit import prepare_truths
    row = dict(subject='a', start_ms=0, end_ms=50)
    with pytest.raises(ValueError, match='duplicate truth'):
        prepare_truths([row, row], {'A': 0})


def test_output_writer_refuses_existing_directory(tmp_path):
    from scripts.research.official_recovery.scope_audit import write_audit
    occupied = tmp_path/'occupied'
    occupied.mkdir()
    (occupied/'keep.txt').write_text('keep', encoding='utf-8')
    with pytest.raises(FileExistsError):
        write_audit(occupied, {})
    assert (occupied/'keep.txt').read_text() == 'keep'


def test_matching_rejects_nonpositive_intervals():
    from scripts.research.official_recovery.scope_audit import match_events
    with pytest.raises(ValueError, match='positive'):
        match_events([('A', 50, 50)], [('A', 0, 100)])
