import hashlib
import numpy as np
import pandas as pd
import pytest


def test_spectral_pair_and_invalid():
    from scripts.research.representation_v2.ppg.window import spectral_stats
    t = np.arange(750)/25
    x = 100 + np.sin(2*np.pi*1.2*t)
    result = spectral_stats(np.column_stack([x, x]))
    assert result['valid'] and result['pair_pass']
    assert abs(result['frequency']-1.2) < 25/512
    assert not spectral_stats(np.column_stack([x, 100+np.sin(2*np.pi*2*t)]))['pair_pass']
    for y in (np.zeros((750, 2)), np.ones((750, 2)), np.full((750, 2), np.nan)):
        assert not spectral_stats(y)['valid']


def test_zero_dc_not_replaced_by_epsilon():
    from scripts.research.representation_v2.ppg.window import spectral_stats
    x = np.tile([-1., 1.], 375)
    result = spectral_stats(np.column_stack([x, x]))
    assert np.isnan(result['rms3_dc'])


def test_file_chunk_order_hash_and_acc(tmp_path):
    from scripts.research.representation_v2.ppg.window import extract_file
    t = np.arange(750)/25
    df = pd.DataFrame({'PPG_TIME': np.repeat(1000+np.arange(50)*600, 15),
        'PPG3': 100+np.sin(2*np.pi*1.2*t), 'PPG5': 200+np.sin(2*np.pi*1.2*t),
        'ACC_TIME': 1000+np.arange(750)*40, 'ACC_X': np.arange(750)%3+1,
        'ACC_Y': np.ones(750), 'ACC_Z': np.ones(750)})
    p = tmp_path/'collect_data.txt'; df.to_csv(p, index=False)
    a = extract_file(p, 's', chunksize=17)
    b = extract_file(p, 's', chunksize=1000)
    assert a['sha256'] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert len(a['windows']) == 1
    assert a['windows'][0] == b['windows'][0]
    df[df.columns[::-1]].to_csv(p, index=False)
    assert extract_file(p, 's')['windows'] == a['windows']


def test_motion_minimum_and_exclusive_end():
    from scripts.research.representation_v2.ppg.window import motion_std
    ts = np.arange(100)
    assert np.isnan(motion_std(ts, np.ones(100), 0, 99))
    assert motion_std(ts, np.ones(100), 0, 100) == 0


def test_tab_delimited_sensor_header(tmp_path):
    from scripts.research.representation_v2.ppg.window import extract_file
    x = np.arange(750)/25
    frame = pd.DataFrame({'PPG_TIME': np.repeat(1000+np.arange(50)*600, 15),
        'PPG3': 100+np.sin(2*np.pi*1.2*x), 'PPG5': 200+np.sin(2*np.pi*1.2*x),
        'ACC_TIME': 1000+np.arange(750)*40, 'ACC_X': 1., 'ACC_Y': 1., 'ACC_Z': 1.})
    path = tmp_path/'collect_data.txt'
    frame.to_csv(path, sep='\t', index=False)
    result = extract_file(path, 's')
    assert len(result['windows']) == 1 and result['windows'][0]['pair_pass']
