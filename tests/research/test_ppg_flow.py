"""Small synthetic full flow: real processes/git seals, no real sensor reads."""
import subprocess

import numpy as np
import pandas as pd
import pytest


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True, stderr=subprocess.DEVNULL)


def make_registered(root, monkeypatch, absent_ppg=False):
    from scripts.research.representation_v2.ppg import audit as a
    from scripts.research.representation_v2.ppg.contract import (OUTPUT, T1, SCHEMA,
                                                               file_hash, validate_inventory)
    raw = root/'raw.txt'
    x = np.arange(750)/25
    frame = pd.DataFrame({'PPG_TIME': np.repeat(2000000+np.arange(50)*600, 15),
                  'PPG3': 100+np.sin(2*np.pi*1.2*x), 'PPG5': 200+np.sin(2*np.pi*1.2*x),
                  'ACC_TIME': 2000000+np.arange(750)*40,
                  'ACC_X': np.ones(750), 'ACC_Y': np.ones(750), 'ACC_Z': np.ones(750)})
    if absent_ppg:
        frame[['PPG_TIME', 'PPG3', 'PPG5']] = 0
    frame.to_csv(raw, index=False)
    key = dict(sid='s', start_ms=2000000, end_ms=2180000)
    inv = dict(keys=[key], occurrences=[dict(partition='wrapper0_health', row=0, key_index=0)],
               subject_by_sid={'s': 'A'}, files=[dict(path='raw.txt', sid='s',
                   sha256=file_hash(raw), bytes=raw.stat().st_size)])
    out = root/OUTPUT; out.mkdir(parents=True)
    t1 = root/T1; t1.mkdir(parents=True)
    a.write_json(t1/'manifest.json', {'subject_by_session': {'s': 'A'}})
    a.write_json(t1/'wrapper0_health_scores.json',
                 dict(candidates=[dict(event=key)], labels=[1], scores={'b': [.75]}))
    a.write_json(t1/'wrapper0_health.json', dict(predictions={'b': [key]}, truths=[key]))
    meal = root/'meal.csv'
    meal.write_text('externalid,beforeTime,afterTime,wearHand,dietaryHand\nA,1900000,2200000,左手,右手\n', encoding='utf-8')
    import src.data.manifests as manifests
    monkeypatch.setattr(manifests, 'MEALS_CSV', meal)
    monkeypatch.setattr(a, 'validate_inventory', lambda v: validate_inventory(v, expected=(1, 1, 1)))
    git(root, 'init', '-q')
    git(root, 'config', 'user.email', 'test@example.invalid')
    git(root, 'config', 'user.name', 'PPG test')
    a.write_json(out/'inventory.json', inv)
    a.write_json(out/'registry.json', dict(schema=SCHEMA, files={},
                 inventory_sha256=file_hash(out/'inventory.json'), protected_index='', registration_hash_bytes=raw.stat().st_size))
    a.write_json(out/'state.json', dict(status='REGISTERED', training_fits=0, outer_calls=0))
    git(root, 'add', '--', OUTPUT.as_posix(), T1.as_posix(), 'raw.txt', 'meal.csv')
    git(root, 'commit', '-qm', 'synthetic register')
    registry = a.read_json(out/'registry.json')
    registry['protected_index'] = a.protected_snapshot(root)
    a.write_json(out/'registry.json', registry)
    git(root, 'add', '--', OUTPUT.as_posix())
    git(root, 'commit', '-qm', 'synthetic protected identity')
    return out, raw


def test_real_synthetic_flow_and_label_gate(tmp_path, monkeypatch):
    from scripts.research.representation_v2.ppg import audit as a, report as r
    from scripts.research.representation_v2.ppg.contract import OUTPUT, read_json
    out, raw = make_registered(tmp_path, monkeypatch)
    a.extract(tmp_path)
    assert read_json(out/'state.json')['status'] == 'EXTRACTED'
    with pytest.raises(RuntimeError, match='committed'):
        r.build_report(tmp_path)
    git(tmp_path, 'add', '--', OUTPUT.as_posix())
    git(tmp_path, 'commit', '-qm', 'synthetic seal')
    # Removing raw makes any accidental raw reread observable.
    raw.unlink()
    r.build_report(tmp_path)
    assert read_json(out/'state.json')['status'] == 'REPORTED'
    assert read_json(out/'coverage_by_partition.json')['wrapper0_health']['n'] == 1
    assert read_json(out/'conditional_evidence.json')['hand_diagnostic']['nondominant']['gt'] == 1
    a.verify(tmp_path)


def test_changed_raw_fails_closed(tmp_path, monkeypatch):
    from scripts.research.representation_v2.ppg import audit as a
    from scripts.research.representation_v2.ppg.contract import read_json
    out, raw = make_registered(tmp_path, monkeypatch)
    raw.write_text(raw.read_text()+'\n')
    with pytest.raises(RuntimeError, match='raw input changed'):
        a.extract(tmp_path)
    assert read_json(out/'state.json')['status'] == 'INCOMPLETE'


def test_absent_ppg_reports_missing_instead_of_crashing(tmp_path, monkeypatch):
    from scripts.research.representation_v2.ppg import audit as a, report as r
    from scripts.research.representation_v2.ppg.contract import OUTPUT, read_json
    out, _ = make_registered(tmp_path, monkeypatch, absent_ppg=True)
    a.extract(tmp_path)
    git(tmp_path, 'add', '--', OUTPUT.as_posix())
    git(tmp_path, 'commit', '-qm', 'synthetic empty window seal')
    r.build_report(tmp_path)
    row = read_json(out/'coverage_by_partition.json')['wrapper0_health']
    assert row['p']['during_minus_pre_frequency']['n_missing'] == 1
    assert row['q']['during_coverage']['median'] == 0
