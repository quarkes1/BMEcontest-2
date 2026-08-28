# -*- coding: utf-8 -*-
"""L2 场景门控单测：倾角特征的场景可分性、低置信回退、序列化往返。"""
import numpy as np
from src.data.loader import SessionData
from src.features.pose import GRAV, per_row_tilt
from src.models.l2_scene_gate import (scene_features, L2SceneGate, N_FEATURES,
                                      SCENE_DOMINANT, SCENE_NONDOMINANT)

def _scene_session(dominant=True, n=1100, seed=0, fs=100.0):
    """合成会话：倾角序列 = 30°±变化。dominant：大幅快变（进食动作）；
    nondominant：近恒定（腕部静止，靠 PPG）。"""
    rng = np.random.RandomState(seed)
    t = np.arange(n) / fs
    if dominant:
        tilt = 30 + 12 * np.sin(2 * np.pi * 1.2 * t) + 3 * rng.randn(n)
        g_amp = 1.0
    else:
        tilt = 30 + 0.4 * rng.randn(n)
        g_amp = 0.05
    tilt = np.clip(tilt, 5, 85)
    a = np.vstack([GRAV * np.sin(np.deg2rad(tilt)),
                   np.zeros(n),
                   GRAV * np.cos(np.deg2rad(tilt))])
    g = g_amp * rng.randn(3, n)
    return SessionData(
        acc=a.astype(np.float32), gyro=g.astype(np.float32),
        ppg=np.zeros((44, n), dtype=np.float32),
        t_acc=(np.arange(n) * 10).astype(np.int64),
        t_ppg=np.full(n, -1, dtype=np.int64),
        imu_valid=np.ones(n, dtype=bool), ppg_valid=np.zeros(n, dtype=bool),
        meta={"row_rate": fs})

def _make_dataset(n_dom=50, n_non=50):
    X = np.vstack([scene_features(_scene_session(dominant=True, seed=i), 0, 525)
                   for i in range(n_dom)]
                  + [scene_features(_scene_session(dominant=False, seed=100 + i), 0, 525)
                     for i in range(n_non)])
    y = np.r_[np.full(n_dom, SCENE_DOMINANT), np.full(n_non, SCENE_NONDOMINANT)]
    return X.astype(np.float32), y

def test_scene_features_shape():
    X = scene_features(_scene_session(), 0, 525)
    assert X.shape == (N_FEATURES,)
    assert np.isfinite(X).all()

def test_scene_separability():
    X, y = _make_dataset()
    order = np.random.RandomState(0).permutation(len(y))
    tr, te = order[:70], order[70:]
    gate = L2SceneGate().fit(X[tr], y[tr])
    pred = gate.predict_scene(X[te])
    conf_mask = pred >= 0
    acc = (pred[conf_mask] == y[te][conf_mask]).mean()
    assert conf_mask.mean() > 0.8, "大多数样本应高置信"
    assert acc > 0.9, f"acc={acc}"

def test_low_confidence_returns_negative():
    """接近边界的样本返回 -1（低置信→IMU 保守分支）。"""
    X, y = _make_dataset(n_dom=30, n_non=30)
    gate = L2SceneGate().fit(X, y)
    probe = np.linspace(0.4, 0.6, 200)          # 特征空间插值，横跨决策边界
    base_dom = X[y == 0].mean(axis=0)
    base_non = X[y == 1].mean(axis=0)
    Xp = np.array([base_dom * (1 - w) + base_non * w for w in probe])
    pred = gate.predict_scene(Xp)
    assert (-1 in pred), "边界附近应出现低置信回退样本"

def test_tilt_rows_reuse():
    s = _scene_session()
    internal = scene_features(s, 100, 625)
    tilt = per_row_tilt(s.acc, 100.0)
    reused = scene_features(s, 100, 625, tilt_rows=tilt)
    assert np.allclose(internal, reused, rtol=1e-4)

def test_save_load_roundtrip(tmp_path):
    X, y = _make_dataset(n_dom=20, n_non=20)
    gate = L2SceneGate().fit(X, y)
    p = tmp_path / "l2.pkl"
    gate.save(p)
    gate2 = L2SceneGate.load(p)
    assert np.allclose(gate.predict_proba(X), gate2.predict_proba(X))
