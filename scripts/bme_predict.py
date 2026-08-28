# -*- coding: utf-8 -*-
"""推理 CLI（提交物①的源码形态）：
python scripts/bme_predict.py <测试数据目录> [--out predict.csv] [--config adapter.json]
[--manifest manifest.json] [--log predict.log]
流程：发现会话 → 逐会话容错推理（损坏跳过）→ predict.csv + 清单 + 日志。
使用 fold 0 模型与统计（正式版待 5 折完成后用集成/最优折，W4 定）。"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import src.config as config
from src.infer import io_adapter
from src.infer.pipeline import InferencePipeline, predict_batch
from src.models.l1_gate import L1Gate
from src.models.l2_scene_gate import L2SceneGate


class _TorchL3a:
    """torch 后端 L3a 包装（ONNX 版同接口替换）。"""
    def __init__(self, fold=0):
        import torch
        from src.models.l3a_cnn import L3aCNN
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = L3aCNN(5).to(self.device).eval()
        self.model.load_state_dict(torch.load(
            config.MODEL_DIR / f"l3a_cnn_fold{fold}.pt", weights_only=True))
        stats = json.loads((config.CACHE_DIR / "l3a_raw" / f"fold{fold}" / "stats.json")
                           .read_text(encoding="utf-8"))
        self.mean = np.array(stats["mean"], dtype=np.float32).reshape(1, 11, 1)
        self.std = np.array(stats["std"], dtype=np.float32).reshape(1, 11, 1) + 1e-6

    def predict(self, X):
        import torch
        with torch.no_grad():
            p = []
            for b in range(0, len(X), 512):
                xb = torch.from_numpy(X[b:b + 512]).to(self.device)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logit, _ = self.model(xb)
                p.append(torch.sigmoid(logit).float().cpu().numpy())
        return np.concatenate(p) if p else np.zeros(0, dtype=np.float32)


class _TorchL3b:
    def __init__(self, fold=0):
        import torch
        from src.models.l3b_ppgnn import L3bPPGNN
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = L3bPPGNN().to(self.device).eval()
        self.model.load_state_dict(torch.load(
            config.MODEL_DIR / f"l3b_ppgnn_fold{fold}.pt", weights_only=True))
        out_dir = config.CACHE_DIR / "l3b_raw" / f"fold{fold}"
        hs = np.concatenate([np.load(f)["hrv"] for f in out_dir.glob("*.npz")]).astype(np.float32)
        self.mean_h = hs.mean(axis=0); self.std_h = hs.std(axis=0) + 1e-6

    def predict(self, X, h):
        import torch
        n = len(X)
        sp = np.zeros(n, dtype=np.float32)
        with torch.no_grad():
            for b in range(0, n, 128):
                s1 = min(b + 128, n)
                xb = torch.from_numpy(X[b:s1]).to(self.device)
                hb = torch.from_numpy(
                    ((h[b:s1] - self.mean_h) / self.std_h)).to(self.device)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = self.model(xb[:, None], hb[:, None])
                sp[b:s1] = torch.sigmoid(logits[:, 0]).float().cpu().numpy()
        return sp


def build_pipeline(fold=0):
    l1 = L1Gate.load(config.MODEL_DIR / "l1_gate.pkl")
    l2 = L2SceneGate.load(config.MODEL_DIR / "l2_scene_gate.pkl")
    l3a = _TorchL3a(fold)
    l3b = _TorchL3b(fold)
    return InferencePipeline(
        l1=l1, l2=l2, l3a=l3a, l3b=l3b,
        l3a_stats={"mean": l3a.mean, "std": l3a.std},
        l3b_stats={"mean_h": l3b.mean_h, "std_h": l3b.std_h})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test_dir")
    ap.add_argument("--out", default="predict.csv")
    ap.add_argument("--config", default=None)
    ap.add_argument("--manifest", default="predict_manifest.json")
    ap.add_argument("--log", default="predict.log")
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()
    adapter = io_adapter.load_adapter(args.config)
    sessions = io_adapter.discover_sessions(Path(args.test_dir), adapter)
    print(f"发现 {len(sessions)} 个会话")
    pipeline = build_pipeline(args.fold)
    results, manifest = predict_batch(sessions, pipeline, index=None, log_path=args.log)
    io_adapter.write_predict_csv(results, Path(args.out), adapter)
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    print(f"完成: 处理 {len(manifest['processed'])} / 跳过 {len(manifest['skipped'])} "
          f"→ {args.out} + {args.manifest} + {args.log}")

if __name__ == "__main__":
    main()
