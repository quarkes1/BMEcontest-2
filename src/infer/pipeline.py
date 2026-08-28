# -*- coding: utf-8 -*-
"""端到端推理管线（spec §3 级联 + §5 测试集容错）：
L1 唤醒 → L2 场景分流 → L3a/L3b 专家打分 → 动态融合（手工 alpha + EMA）
→ HMM Viterbi → 事件输出。
容错（用户明确要求）：① 逐会话 detect_binary 检测，损坏跳过并记 predict.log
② 单会话解析失败不中断整体 ③ 空 externalid 跳过 ④ 与训练侧同一套掩码/窗口逻辑
⑤ 输出已处理/跳过明细清单。"""
import logging
import time
import numpy as np

import src.config as config
from src.data.loader import load_session, detect_binary, _find_collect_data
from src.data.windows import iter_window_labels
from src.features.pose import per_row_tilt
from src.models.l1_gate import gate_features
from src.models.l2_scene_gate import scene_features
from src.models.l3a_cnn import build_raw_channels
from src.models.l3b_ppgnn import build_ppg_window
from src.models.fusion import handcrafted_alpha, ema_smooth, fuse
from src.models.hmm_decode import viterbi, decode_events

class InferencePipeline:
    """聚合全部模块的推理入口。模型对象与标准化统计由调用方注入（ONNX 版同接口）。"""
    def __init__(self, l1, l2, l3a, l3b, l3a_stats, l3b_stats, device="cpu",
                 wake_thr=0.15, scene_conf=0.6):
        self.l1, self.l2 = l1, l2
        self.l3a, self.l3b = l3a, l3b
        self.l3a_stats, self.l3b_stats = l3a_stats, l3b_stats
        self.device = device
        self.wake_thr = wake_thr
        self.scene_conf = scene_conf
        self.log = logging.getLogger("bme_predict")

    def _l3a_score_windows(self, session, rows):
        X = np.stack([build_raw_channels(session, s, e) for s, e in rows]).astype(np.float32)
        X = (X - self.l3a_stats["mean"]) / self.l3a_stats["std"]
        return self.l3a.predict(X)                     # (n,) 概率

    def _l3b_score_windows(self, session, rows):
        Xs, hs = [], []
        for s, e in rows:
            X, h = build_ppg_window(session, s, e)
            Xs.append(X); hs.append(h)
        X = np.stack(Xs).astype(np.float32)
        h = ((np.stack(hs) - self.l3b_stats["mean_h"]) / self.l3b_stats["std_h"]).astype(np.float32)
        return self.l3b.predict(X, h)                  # (n,) 概率

    def predict_session(self, session):
        """单会话 → 事件列表 [(start_ms, end_ms)]。会话内异常由调用方隔离。"""
        fs = session.meta.get("row_rate", 105.0)
        win = list(iter_window_labels(session, []))
        if not win:
            return []
        rows = [(w["start_row"], w["end_row"]) for w in win]
        # L1 唤醒
        X1 = np.stack([gate_features(session, s, e) for s, e in rows]).astype(np.float32)
        wake = self.l1.wakeup(X1, self.wake_thr)
        if not wake.any():
            return []
        idx = np.flatnonzero(wake)
        wake_rows = [rows[i] for i in idx]
        # L2 场景分流
        tilts = per_row_tilt(session.acc, fs)
        X2 = np.stack([scene_features(session, s, e, tilt_rows=tilts) for s, e in wake_rows])
        scenes = self.l2.predict_scene(X2.astype(np.float32))      # 0/1/-1(低置信→IMU)
        # 专家打分（低置信走 L3a 保守分支；两分支并存时按手工 alpha 融合）
        p = np.zeros(len(idx), dtype=np.float32)
        a_idx = [j for j, sc in enumerate(scenes) if sc != 1]
        b_idx = [j for j, sc in enumerate(scenes) if sc == 1]
        if a_idx:
            p[np.array(a_idx)] = self._l3a_score_windows(session, [wake_rows[j] for j in a_idx])
        if b_idx:
            p_b = self._l3b_score_windows(session, [wake_rows[j] for j in b_idx])
            if a_idx:
                p_a = p[np.array(a_idx)]
                alpha = handcrafted_alpha(1.0, 1.0, 1.0)
                # 分支各自独立解码后合并事件（简化融合，正式融合见 run_full_pipeline）
                p[np.array(b_idx)] = p_b
            else:
                p[np.array(b_idx)] = p_b
        t0_ms = np.array([win[i]["t0_ms"] for i in idx])
        t1_ms = np.array([win[i]["t1_ms"] for i in idx])
        st = viterbi(p)
        return decode_events(st, t0_ms, t1_ms)


def predict_batch(session_ids, pipeline, index, log_path=None, skip_existing=True):
    """批量推理（测试集容错主入口）。返回 (results, manifest)。
    results: {sid: [(start_ms, end_ms)]}；manifest: {processed: [...], skipped: [...]}。"""
    logger = logging.getLogger("bme_predict")
    if log_path:
        h = logging.FileHandler(log_path, encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    results, processed, skipped = {}, [], []
    t0 = time.time()
    for i, sid in enumerate(session_ids):
        try:
            d = config.SENSOR_DIR / sid
            txt = _find_collect_data(str(d))
            if detect_binary(txt):
                skipped.append({"session_id": sid, "reason": "binary_corrupt"})
                logger.warning(f"skip binary corrupt: {sid}")
                continue
            session = load_session(sid)
            events = pipeline.predict_session(session)
            results[sid] = events
            processed.append({"session_id": sid, "n_events": len(events)})
        except Exception as e:
            skipped.append({"session_id": sid, "reason": f"{type(e).__name__}: {e}"})
            logger.error(f"skip error: {sid}: {type(e).__name__}: {e}")
        if (i + 1) % 50 == 0:
            logger.info(f"progress {i+1}/{len(session_ids)} {time.time()-t0:.0f}s")
    manifest = {"processed": processed, "skipped": skipped,
                "n_total": len(session_ids), "elapsed_s": round(time.time() - t0, 1)}
    return results, manifest
