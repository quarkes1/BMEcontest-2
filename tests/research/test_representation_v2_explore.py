from copy import deepcopy
import pytest

from scripts.research.representation_v2.explore import select_gate, write_json


def test_gate_refuses_ap_gain_with_worse_calibration_or_collapse():
    metric = lambda ap,bce: {'average_precision':ap,'raw_logit_bce':bce}
    row = {'collapse':{'passed':True},'metrics':{
        'm':{'all':metric(.2,.35)},'r':{'all':metric(.3,.35)},
        's':{'all':metric(.34,.36)},'prior':{'all':metric(.1,.35)}},
        'checks':dict.fromkeys(['provenance','finite','optimizer','resource'],True)}
    rows = [deepcopy(row) for _ in range(3)]
    assert select_gate(rows)['passed']
    rows[0]['metrics']['s']['all']['raw_logit_bce'] = 1.
    assert not select_gate(rows)['passed']
    rows = [deepcopy(row) for _ in range(3)]
    rows[0]['checks']['provenance'] = False
    assert not select_gate(rows)['passed']
    rows[0]['checks'].pop('provenance')
    assert not select_gate(rows)['passed']
    rows = [deepcopy(row) for _ in range(3)]
    rows[0]['collapse']['passed'] = False
    assert not select_gate(rows)['passed']


def test_evidence_json_uses_lf_bytes(tmp_path):
    path=tmp_path/'evidence.json'
    write_json(path, {'value':[1,2]})
    assert b'\r' not in path.read_bytes()
    assert not path.with_suffix('.json.tmp').exists()
