# -*- coding: utf-8 -*-
"""L2 场景门控层（分流器，spec §3）：
- 特征：倾角（重力与腕 Z 夹角）均值/标准差、倾角变化率分位数、陀螺活动度、IMU 有效率
- 分类器：逻辑回归，scene 真值监督（dominant 惯用手 / nondominant 非惯用手）
- 训练数据 = 正标签（进食）窗口；部署时低置信（<0.6）走 IMU 保守分支（L3a）
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.features.pose import per_row_tilt

N_FEATURES = 7
SCENE_DOMINANT = 0
SCENE_NONDOMINANT = 1
CONFIDENCE_MIN = 0.6


def scene_features(session, start, end, tilt_rows=None) -> np.ndarray:
    """窗口级 L2 特征（7 维）。tilt_rows 可传入全会话 per_row_tilt 结果避免重复计算。"""
    fs = session.meta.get("row_rate", 105.0)
    if tilt_rows is None:
        tilt_rows = per_row_tilt(session.acc, fs)
    tilt = tilt_rows[start:end]
    dtilt = np.abs(np.diff(np.r_[tilt[0], tilt]))
    g = session.gyro[:, start:end]
    gm = np.linalg.norm(g, axis=0)
    return np.array([
        float(np.mean(tilt)), float(np.std(tilt)),
        float(np.percentile(tilt, 90) - np.percentile(tilt, 10)),
        float(np.percentile(dtilt, 50)), float(np.percentile(dtilt, 90)),
        float(np.std(gm)),
        float(session.imu_valid[start:end].mean()),
    ], dtype=np.float32)


class L2SceneGate:
    """场景分流逻辑回归。fit(X, y_scene)：y 用 SCENE_DOMINANT/SCENE_NONDOMINANT。"""
    def __init__(self, C=1.0, max_iter=2000):
        self.model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)

    def fit(self, X, y_scene):
        self.model.fit(X, y_scene)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, SCENE_NONDOMINANT]

    def predict_scene(self, X):
        """返回 0/1 场景编码；概率低于置信阈值的样本标记为 -1（低置信→IMU 保守分支）。"""
        p = self.predict_proba(X)
        scene = (p >= 0.5).astype(int)
        scene[np.abs(p - 0.5) < (CONFIDENCE_MIN - 0.5)] = -1
        return scene

    def to_bytes(self):
        import pickle
        return pickle.dumps(self.model)

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
