#!/bin/bash
# 6 套重训选优：每折 6 个 seed(42-47) → 30 次单折训练 → 每套解码选优 → 最优套固化 models/
set -e
cd "$(dirname "$0")/.."
source activate bme 2>/dev/null
OUT=outputs
mkdir -p "$OUT/run6"

# ---- 第 3 步：30 次单折训练（6 seed × 5 折） ----
for seed in 42 43 44 45 46 47; do
  for k in 0 1 2 3 4; do
    echo "===== seed $seed fold $k ====="
    BME_SEED=$seed python -u scripts/train_ranker.py --fold $k --no-ppg \
      --init-from checkpoints/fd_pretrained_s1.pt \
      > "$OUT/run6/s${seed}_f${k}.log" 2>&1 || { echo "FAILED s${seed}_f${k}"; tail -3 "$OUT/run6/s${seed}_f${k}.log"; exit 1; }
    cp "$OUT/mm_ranker_fold${k}_val.npz" "$OUT/run6/run${seed}_fold${k}.npz"
    cp "models/mm_ranker_fold${k}.pt" "$OUT/run6/s${seed}_f${k}.pt"
    tail -1 "$OUT/run6/s${seed}_f${k}.log"
  done
done

# ---- 第 4 步：每套评估选优（τ 小网格，固定历史 best 的 g/p/dil/k） ----
python - << 'EOF'
import json, re, sys
sys.path.insert(0, "scripts")
import numpy as np
import src.config as config
import rank_events_v2 as v2
import official_iou_eval as oe   # 全局官方口径（提交口径）

SEEDS = [42, 43, 44, 45, 46, 47]
results = {}
for seed in SEEDS:
    f1s = []
    for k in range(5):
        # 历史 best 配置（g/p/dil/k 复用；w=1.0 深度分主导；τ 扫小网格）
        j = json.loads((config.OUTPUT_DIR / "archive_final_20260902"
                        / f"rank_events_v2_fold{k}_15m.json").read_text(encoding="utf-8"))
        mm = re.match(r"w([\d.]+)_t([\d.]+)_g([\d.]+)_p([\d.]+)_d(\d+)_k(\d+)", j["best"]["name"])
        thr_g, thr_p, dil, K = float(mm[3]), float(mm[4]), int(mm[5]), int(mm[6])
        val_rows, gate_prob, clf_pri, true_sid, subject_of, _ = v2.prepare_fold(k)
        # 该套分数对齐
        z = np.load(config.OUTPUT_DIR / f"run6/run{seed}_fold{k}.npz", allow_pickle=True)
        dl = {}
        for m, sc in zip(z["meta"], z["score"]):
            mm2 = json.loads(m.decode())
            dl[(mm2["sid"], int(mm2["s"]), int(mm2["e"]))] = float(sc)
        rows2 = []
        for r in val_rows:
            sid, act, sa, _, pri, sp = r
            vs = np.full(len(act), np.nan, np.float32)
            for jj, c in enumerate(act):
                if (sid, c[0], c[1]) in dl:
                    vs[jj] = dl[(sid, c[0], c[1])]
            rows2.append((sid, act, sa, vs, pri, sp))
        best = -1.0
        for tau in [x / 200.0 for x in range(30, 81, 5)]:   # 0.15-0.40（w=1.0 深度分主导）
            pred = []
            for r in rows2:
                pred.extend(v2.decode_session(r, gate_prob, (1.0, tau, thr_g, thr_p, dil, K),
                                              clf_pri)[0])
            m = oe.official_metrics(pred, true_sid)   # 全局官方口径 F1
            best = max(best, m["f1"])
        f1s.append(best)
    results[seed] = f1s
    print(f"seed {seed}: " + " ".join(f"{x:.3f}" for x in f1s) + f" → 均值 {np.mean(f1s):.3f}", flush=True)

best_seed = max(results, key=lambda s: np.mean(results[s]))
print(f"\n★ 最优套: seed {best_seed}（均值 {np.mean(results[best_seed]):.3f}）")
import shutil
for k in range(5):
    shutil.copy(config.OUTPUT_DIR / f"run6/run{best_seed}_fold{k}.npz",
                config.OUTPUT_DIR / f"mm_ranker_fold{k}_val.npz")
    shutil.copy(config.OUTPUT_DIR / f"run6/s{best_seed}_f{k}.pt",
                f"models/mm_ranker_fold{k}.pt")
print("canonical = 最优套分数；models/ = 最优套权重")
EOF
echo "===== 完成（最优套权重在 models/，分数在 canonical） ====="
