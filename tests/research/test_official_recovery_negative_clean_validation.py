from pathlib import Path

import pytest


def test_policy_subject_partitions_are_fixed_disjoint_and_complete():
    from scripts.research.official_recovery.negative_clean_validation import policy_partitions
    subjects=[f'S{i}' for i in range(31)]
    splits=policy_partitions(subjects)
    assert splits==policy_partitions(list(reversed(subjects)))
    assert sorted(s for _,v in splits for s in v)==sorted(subjects)
    assert all(not set(t)&set(v) and set(t)|set(v)==set(subjects) for t,v in splits)
    with pytest.raises(ValueError): policy_partitions(subjects+subjects[:1])


def test_gate_requires_five_complete_folds_gain_majority_and_no_tp_loss():
    from scripts.research.official_recovery.negative_clean_validation import validation_gate
    rows=[dict(fold=i,events={a:dict(tp=20+(2 if a=='W' else 0),predictions=30,truths=n)
        for a in ('A','W')}) for i,n in enumerate((47,51,52,50,54))]
    assert validation_gate(rows)['passed']
    with pytest.raises(ValueError): validation_gate(rows[:4])
    rows[0]['events']['W']['truths']+=1
    with pytest.raises(ValueError): validation_gate(rows)
    rows[0]['events']['W']['truths']-=1
    for r in rows: r['events']['W']['tp']=19;r['events']['W']['predictions']=20
    assert not validation_gate(rows)['passed']
    for r in rows: r['events']['W']['tp']=20;r['events']['W']['predictions']=30
    assert not validation_gate(rows)['passed']


def test_explicit_paired_arms_reject_invalid_before_fitting():
    from scripts.research.official_recovery.negative_clean_probe import _paired
    for arms in (('A','W','W'),('W',),('A','unknown'),()):
        with pytest.raises(ValueError,match='arms'):
            _paired(None,None,(),(),(),(),Path('.'),'unused',arms=arms)


def test_validation_failed_preflight_does_not_overwrite(tmp_path):
    import json
    from scripts.research.official_recovery.negative_clean_validation import execute
    output=tmp_path/'output'
    with pytest.raises(Exception): execute(tmp_path,output)
    assert json.loads((output/'state.json').read_text())['state']=='INCOMPLETE'
    before=(output/'state.json').read_bytes()
    with pytest.raises(FileExistsError): execute(tmp_path,output)
    assert (output/'state.json').read_bytes()==before
