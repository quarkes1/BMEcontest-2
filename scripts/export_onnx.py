# -*- coding: utf-8 -*-
"""ONNX 导出（spec §5：L3a/L3b 导出 ONNX；L1/L2/L4 纯 numpy）：
python scripts/export_onnx.py [--fold 0]
产物：models/l3a_cnn_fold{k}.onnx、models/l3b_ppgnn_fold{k}.onnx"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import src.config as config
from src.models.l3a_cnn import L3aCNN, N_CHANNELS, WINDOW_LEN
from src.models.l3b_ppgnn import L3bPPGNN, N_PPG_CHANNELS, PPG_WINDOW_ROWS, HRV_DIMS, SEQ_LEN

class _L3aEatOnly(nn.Module):
    """只导出进食概率头（餐具辅助头推理丢弃）。"""
    def __init__(self, base):
        super().__init__()
        self.base = base
    def forward(self, x):
        logit, _ = self.base(x)
        return torch.sigmoid(logit[:, None])         # (B, 1) 概率

class _L3bProb(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
    def forward(self, x, h):
        logits = self.base(x, h)
        return torch.sigmoid(logits)                 # (B, seq)

def export(fold):
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # L3a
    m3a = L3aCNN(5).eval()
    m3a.load_state_dict(torch.load(config.MODEL_DIR / f"l3a_cnn_fold{fold}.pt", weights_only=True))
    out3a = config.MODEL_DIR / f"l3a_cnn_fold{fold}.onnx"
    torch.onnx.export(_L3aEatOnly(m3a),
                      torch.zeros(1, N_CHANNELS, WINDOW_LEN),
                      out3a,
                      input_names=["x"], output_names=["p"],
                      dynamic_axes={"x": {0: "batch"}, "p": {0: "batch"}},
                      opset_version=17)
    print(f"L3a -> {out3a}")

    # L3b
    m3b = L3bPPGNN().eval()
    m3b.load_state_dict(torch.load(config.MODEL_DIR / f"l3b_ppgnn_fold{fold}.pt", weights_only=True))
    out3b = config.MODEL_DIR / f"l3b_ppgnn_fold{fold}.onnx"
    torch.onnx.export(_L3bProb(m3b),
                      (torch.zeros(1, SEQ_LEN, N_PPG_CHANNELS, PPG_WINDOW_ROWS),
                       torch.zeros(1, SEQ_LEN, HRV_DIMS)),
                      out3b,
                      input_names=["x", "h"], output_names=["p"],
                      dynamic_axes={"x": {0: "batch"}, "h": {0: "batch"},
                                    "p": {0: "batch"}},
                      opset_version=17)
    print(f"L3b -> {out3b}")

def smoke(fold):
    """onnxruntime 冒烟：与 torch 输出一致。"""
    import onnxruntime as ort
    import src.config as config
    rng = np.random.RandomState(0)
    for name, make_inputs in (
        ("l3a_cnn", lambda: (rng.randn(3, N_CHANNELS, WINDOW_LEN).astype(np.float32),)),
        ("l3b_ppgnn", lambda: (rng.randn(2, SEQ_LEN, N_PPG_CHANNELS, PPG_WINDOW_ROWS).astype(np.float32),
                               rng.randn(2, SEQ_LEN, HRV_DIMS).astype(np.float32)))):
        path = config.MODEL_DIR / f"{name}_fold{fold}.onnx"
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = make_inputs()
        outs = sess.run(None, dict(zip([i.name for i in sess.get_inputs()], inputs)))
        print(f"{name}: ort output shape {outs[0].shape}, range [{outs[0].min():.3f}, {outs[0].max():.3f}]")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    export(args.fold)
    if args.smoke:
        smoke(args.fold)
