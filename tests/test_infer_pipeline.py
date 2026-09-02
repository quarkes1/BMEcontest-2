# -*- coding: utf-8 -*-
"""推理管线单测：级联流程 + 测试集容错（损坏文件跳过、清单明细）。"""
import numpy as np
from src.data.loader import SessionData
from src.infer.pipeline import InferencePipeline, predict_batch

def _session(n=3300, seed=0):
    rng = np.random.RandomState(seed)
    return SessionData(
        acc=rng.randn(3, n).astype(np.float32),
        gyro=rng.randn(3, n).astype(np.float32),
        ppg=(rng.randn(44, n) + 100).astype(np.float32),
        t_acc=(np.arange(n) * 10).astype(np.int64),
        t_ppg=np.full(n, -1, dtype=np.int64),
        imu_valid=np.ones(n, dtype=bool), ppg_valid=np.ones(n, dtype=bool),
        meta={"row_rate": 100.0})

class _FakeL1:
    def wakeup(self, X, thr):
        return np.ones(len(X), dtype=bool)          # 全唤醒

class _FakeL2:
    def predict_scene(self, X):
        return np.zeros(len(X), dtype=np.int64)     # 全 dominant → L3a

class _FakeL3a:
    def predict(self, X):
        # 每 5 个窗口一段高概率（模拟进食段）
        p = np.zeros(len(X), dtype=np.float32)
        p[10:60] = 0.9
        return p

class _FakeL3b:
    def predict(self, X, h):
        return np.zeros(len(X), dtype=np.float32)

def _pipeline():
    return InferencePipeline(
        l1=_FakeL1(), l2=_FakeL2(), l3a=_FakeL3a(), l3b=_FakeL3b(),
        l3a_stats={"mean": np.zeros((11, 1), np.float32), "std": np.ones((11, 1), np.float32)},
        l3b_stats={"mean_h": np.zeros(8, np.float32), "std_h": np.ones(8, np.float32)})

def test_predict_session_produces_events():
    s = _session(n=9000)                    # 90 个窗口，高概率段 50 个 = 50s > 45s 门槛
    evs = _pipeline().predict_session(s)
    assert len(evs) >= 1
    assert all(isinstance(s, int) and isinstance(e, int) and s < e for s, e in evs)

def test_predict_session_empty_no_crash():
    # 无有效 IMU 时间戳 → 窗口全被跳过 → 空事件
    s = _session()
    s.t_acc[:] = -1
    assert _pipeline().predict_session(s) == []

def test_predict_batch_skips_corrupt(tmp_path):
    """损坏会话跳过并记入清单；正常会话继续处理。"""
    import src.config as config
    sid_corrupt = "sensorData-corrupt-fake"
    d = config.SENSOR_DIR / sid_corrupt
    d.mkdir(parents=True, exist_ok=True)
    (d / "collect_data123_456_789.txt").write_bytes(b"\x00\x01\x02" * 100 + b"garbage")
    results, manifest = predict_batch(
        [sid_corrupt, "sensorData-nonexistent-fake"], _pipeline(),
        index=None, log_path=str(tmp_path / "predict.log"))
    assert sid_corrupt in [s["session_id"] for s in manifest["skipped"]]
    assert any("binary_corrupt" == s["reason"] for s in manifest["skipped"])
    assert any("error" in s["reason"] or "FileNotFound" in s["reason"]
               for s in manifest["skipped"])
    assert manifest["n_total"] == 2
    assert len(manifest["processed"]) + len(manifest["skipped"]) == 2
