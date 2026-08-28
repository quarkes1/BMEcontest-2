# -*- coding: utf-8 -*-
"""L1 守门员单测：特征形状、可分性、唤醒灵敏度、序列化体积。"""
import numpy as np
from src.data.loader import SessionData
from src.features.pose import GRAV
from src.models.l1_gate import gate_features, L1Gate, N_FEATURES

def _eating_session(n=1100, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(n) / 100.0
    a = np.vstack([0.3 * rng.randn(n) + 0.8 * np.sin(2 * np.pi * 1.5 * t),
                   0.3 * rng.randn(n),
                   GRAV + 0.3 * rng.randn(n)])
    g = 1.5 * rng.randn(3, n)
    p = 50 * rng.randn(44, n)
    return SessionData(
        acc=a.astype(np.float32), gyro=g.astype(np.float32), ppg=p.astype(np.float32),
        t_acc=(np.arange(n) * 10).astype(np.int64),
        t_ppg=np.full(n, -1, dtype=np.int64),
        imu_valid=np.ones(n, dtype=bool), ppg_valid=np.ones(n, dtype=bool),
        meta={"row_rate": 100.0})

def _still_session(n=1100, seed=1):
    rng = np.random.RandomState(seed)
    a = np.vstack([0.01 * rng.randn(n), 0.01 * rng.randn(n), GRAV + 0.01 * rng.randn(n)])
    g = 0.02 * rng.randn(3, n)
    p = 2 * rng.randn(44, n)
    return SessionData(
        acc=a.astype(np.float32), gyro=g.astype(np.float32), ppg=p.astype(np.float32),
        t_acc=(np.arange(n) * 10).astype(np.int64),
        t_ppg=np.full(n, -1, dtype=np.int64),
        imu_valid=np.ones(n, dtype=bool), ppg_valid=np.ones(n, dtype=bool),
        meta={"row_rate": 100.0})

def _make_dataset(n_eat=60, n_still=60):
    X = np.vstack([gate_features(_eating_session(seed=i), 0, 525) for i in range(n_eat)]
                  + [gate_features(_still_session(seed=100 + i), 0, 525) for i in range(n_still)])
    y = np.r_[np.ones(n_eat), np.zeros(n_still)]
    return X.astype(np.float32), y

def test_gate_features_shape():
    X = gate_features(_eating_session(), 0, 525)
    assert X.shape == (N_FEATURES,)
    assert np.isfinite(X).all()

def test_fit_wakeup_sensitivity():
    X, y = _make_dataset()
    order = np.random.RandomState(0).permutation(len(y))
    tr, te = order[:80], order[80:]
    gate = L1Gate().fit(X[tr], y[tr])
    mask = gate.wakeup(X[te])
    sens = mask[y[te] == 1].mean()          # 唤醒对进食段高召回
    spec = 1 - mask[y[te] == 0].mean()      # 对静止段高排除
    assert sens >= 0.9, f"sens={sens}"
    assert spec >= 0.7, f"spec={spec}"

def test_model_size_under_8kb():
    X, y = _make_dataset(n_eat=30, n_still=30)
    gate = L1Gate().fit(X, y)
    assert gate.size_bytes() < 8 * 1024

def test_save_load_roundtrip(tmp_path):
    X, y = _make_dataset(n_eat=20, n_still=20)
    gate = L1Gate().fit(X, y)
    p = tmp_path / "l1.pkl"
    gate.save(p)
    gate2 = L1Gate.load(p)
    assert np.allclose(gate.predict_proba(X), gate2.predict_proba(X))

def test_ppg_channel_feature():
    """PPG 通道：活跃 PPG（大 AC）与静默 PPG 在第 6/7 维特征上可区分。"""
    fe = gate_features(_eating_session(), 0, 525)
    fs = gate_features(_still_session(), 0, 525)
    assert fe[6] > fs[6] * 5
