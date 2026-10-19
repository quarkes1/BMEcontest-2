"""Labels depend on real meal overlap, not containment in an export file."""
import numpy as np
import pytest


def test_cross_file_meal_is_positive_inside_either_file():
    from scripts.research.official_recovery.label_probe import corrected_labels
    # Meal [80,180] spans hypothetical file boundary100; both windows are eating.
    assert corrected_labels([(70,110),(130,170)],[(80,180)],micro=False).tolist()==[1,1]


def test_macro_and_micro_half_overlap_keep_their_original_tie_rules():
    from scripts.research.official_recovery.label_probe import corrected_labels
    assert corrected_labels([(0,100)],[(50,200)],micro=False).tolist()==[-1]
    assert corrected_labels([(0,100)],[(50,200)],micro=True).tolist()==[1]


def test_unlabeled_gaps_near_meals_are_not_trained_as_negatives():
    from scripts.research.official_recovery.label_probe import corrected_labels
    assert corrected_labels([(0,100000),(100000,200000)],[(500000,600000)],micro=False).tolist()==[0,0]
    assert corrected_labels([(200001,300001)],[(500000,600000)],micro=False).tolist()==[-1]


def test_session_sampling_keeps_new_positive_and_is_deterministic():
    from scripts.research.official_recovery.label_probe import sample_rows
    labels=np.array([1]+[0]*20+[-1],dtype=np.int8)
    ids=np.array(['S']*len(labels))
    a=sample_rows(labels,ids)
    assert len(a)==4 and 0 in a and 21 not in a
    assert np.array_equal(a,sample_rows(labels,ids))


def test_pilot_partitions_reject_heldout_and_missing_coverage():
    from scripts.research.official_recovery.label_probe import validate_partitions
    manifest={'identities':{'t24':['A','B'],'v7':['Z'],'outer':[]},
              'wrappers':[{'id':0,'train':['A'],'health':['B']},
                          {'id':1,'train':['B'],'health':['A']}]}
    validate_partitions(manifest,expected_subjects=2)
    manifest['wrappers'][0]['train']=['Z']
    with pytest.raises(ValueError,match='partition'):
        validate_partitions(manifest,expected_subjects=2)
