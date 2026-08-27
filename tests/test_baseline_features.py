# -*- coding: utf-8 -*-
import numpy as np
from src.features.baseline_features import window_features
from src.data.loader import SessionData

def test_window_features_shape():
    N = 1100
    s = SessionData(
        acc=np.random.randn(3, N).astype(np.float32),
        gyro=np.random.randn(3, N).astype(np.float32),
        ppg=np.random.randn(44, N).astype(np.float32),
        t_acc=(np.arange(N) * 10).astype(np.int64),
        t_ppg=np.full(N, -1, dtype=np.int64),
        imu_valid=np.ones(N, dtype=bool), ppg_valid=np.zeros(N, dtype=bool),
        meta={"row_rate": 100.0})
    f = window_features(s, 0, 525)
    assert f.shape == (37,)
    assert np.isfinite(f).all()
