# -*- coding: utf-8 -*-
"""ONNX Runtime 后端（PyInstaller 打包用：无 torch 依赖）。
与 scripts/bme_predict.py 中 _TorchL3a/_TorchL3b 同接口。"""
import json
import numpy as np
import src.config as config


class OnnxL3a:
    def __init__(self, fold=0):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(
            str(config.MODEL_DIR / f"l3a_cnn_fold{fold}.onnx"),
            providers=["CPUExecutionProvider"])
        stats = json.loads((config.CACHE_DIR / "l3a_raw" / f"fold{fold}" / "stats.json")
                           .read_text(encoding="utf-8"))
        self.mean = np.array(stats["mean"], dtype=np.float32).reshape(1, 11, 1)
        self.std = np.array(stats["std"], dtype=np.float32).reshape(1, 11, 1) + 1e-6

    def predict(self, X):
        X = (X - self.mean) / self.std
        out = self.sess.run(None, {"x": X})[0]
        return out[:, 0].astype(np.float32)


class OnnxL3b:
    def __init__(self, fold=0):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(
            str(config.MODEL_DIR / f"l3b_ppgnn_fold{fold}.onnx"),
            providers=["CPUExecutionProvider"])
        out_dir = config.CACHE_DIR / "l3b_raw" / f"fold{fold}"
        hs = np.concatenate([np.load(f)["hrv"] for f in out_dir.glob("*.npz")]).astype(np.float32)
        self.mean_h = hs.mean(axis=0); self.std_h = hs.std(axis=0) + 1e-6

    def predict(self, X, h):
        """X: (n, 44, 720), h: (n, 8) → (n,) 概率（单窗序列）。"""
        h = ((h - self.mean_h) / self.std_h).astype(np.float32)
        out = self.sess.run(None, {"x": X[:, None], "h": h[:, None]})[0]
        return out[:, 0].astype(np.float32)
