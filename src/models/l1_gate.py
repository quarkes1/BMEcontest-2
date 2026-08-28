# -*- coding: utf-8 -*-
"""L1 事件唤醒层（守门员，spec §3）：
- IMU 通道：合加速度 2s 滑动方差 + 过零率 + 0.5-2Hz 带通能量 + 姿态变化（窗口级）
- PPG 通道：8s 短窗 AC 幅值 + 直流漂移（部署时每 30s 周期巡值，补非惯用手静态进食）
- 分类器：决策树 max_depth=6（序列化 <8KB，超低功耗常驻叙事）
- 训练监督 = 窗口进食标签；推理用低阈值高召回唤醒（宁多勿漏）
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from src.features.baseline_features import _bandpass

IMU_DIMS = 6
PPG_DIMS = 2
N_FEATURES = IMU_DIMS + PPG_DIMS
PPG_CHANNELS = slice(0, 10)          # 前 10 通道（与基线特征同口径）


def gate_features(session, start, end) -> np.ndarray:
    """窗口级 L1 特征（8 维）。start/end 为行号。"""
    fs = session.meta.get("row_rate", 105.0)
    a = session.acc[:, start:end]
    g = session.gyro[:, start:end]
    am = np.linalg.norm(a, axis=0)
    gm = np.linalg.norm(g, axis=0)

    # 2s 滑动方差的最大值（窗口内）
    win = max(4, int(fs * 2.0))
    csum = np.cumsum(np.insert(am.astype(np.float64), 0, 0.0))
    csum2 = np.cumsum(np.insert(am.astype(np.float64) ** 2, 0, 0.0))
    n = np.arange(1, len(am) + 1)
    n0 = np.clip(n - win, 1, None)
    cnt = n - n0 + 1
    var_series = np.clip((csum2[1:] - csum2[n0 - 1]) / cnt
                         - ((csum[1:] - csum[n0 - 1]) / cnt) ** 2, 0, None)
    var_2s = float(var_series.max()) if len(var_series) else 0.0

    am_c = am - am.mean()
    zcr = float(np.count_nonzero(np.diff(np.signbit(am_c))) / max(1, len(am_c) - 1))
    am_bp = _bandpass(am, 0.5, 2.0, fs)

    # 姿态变化：窗口首末 1s 倾角差（用低通重力方向近似）
    w_tilt = max(4, int(fs * 1.0))
    def _tilt_end(seg):
        if len(seg) < w_tilt:
            seg = np.pad(seg, (w_tilt - len(seg), 0), mode="edge")
        gv = seg.mean(axis=1)
        gn = np.linalg.norm(gv) + 1e-9
        return np.degrees(np.arccos(np.clip(gv[2] / gn, -1, 1)))
    tilt_change = abs(_tilt_end(a[:, -w_tilt:]) - _tilt_end(a[:, :w_tilt]))

    # PPG 8s 短窗（若窗口不足 8s 则用窗口本身）
    p = session.ppg[PPG_CHANNELS, start:end]
    if p.shape[1] > 1:
        ppg_ac = float(np.mean(np.std(p, axis=1)))
        ppg_dc = float(abs(p[:, -1].mean() - p[:, 0].mean()))
    else:
        ppg_ac = ppg_dc = 0.0

    return np.array([
        var_2s, zcr, float(np.mean(np.abs(am_bp))), float(np.std(am)),
        float(np.std(gm)), tilt_change, ppg_ac, ppg_dc,
    ], dtype=np.float32)


class L1Gate:
    """IMU 唤醒决策树。fit(X, y)：X=(n,8) 特征，y=进食标签 0/1。"""
    def __init__(self, max_depth=6, min_samples_leaf=8, class_weight="balanced"):
        self.model = DecisionTreeClassifier(
            max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            class_weight=class_weight, random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def wakeup(self, X, threshold=0.15):
        """高灵敏度唤醒掩码（默认阈值 0.15，宁多勿漏）。"""
        return self.predict_proba(X) >= threshold

    def to_bytes(self):
        import pickle
        return pickle.dumps(self.model)

    def size_bytes(self):
        return len(self.to_bytes())

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, path):
        import pickle
        with open(path, "rb") as f:
            model = pickle.load(f)
        gate = cls.__new__(cls)
        gate.model = model
        return gate
