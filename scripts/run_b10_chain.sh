#!/bin/bash
# B10 配置全 5 折：6ch 全量缓存(重建中) → FD init 训练 → 解码 → 官方评估汇总
set -e
cd "$(dirname "$0")/.."
source activate bme 2>/dev/null
OUT=outputs

for k in 0 1 2 3 4; do
  echo "===== fold $k B10 训练 ====="
  python -u scripts/train_ranker.py --fold $k --no-ppg --init-from checkpoints/fd_pretrained_s1.pt \
    > "$OUT/b10_train_f${k}.log" 2>&1 || { echo "fold$k FAILED"; tail -3 "$OUT/b10_train_f${k}.log"; exit 1; }
  cp "$OUT/mm_ranker_fold${k}_val.npz" "$OUT/b10_final_f${k}.npz"
  tail -1 "$OUT/b10_train_f${k}.log"
done

echo "===== 解码 15m ====="
for k in 0 1 2 3 4; do
  python -u scripts/rank_events_v2.py --fold $k --prior-grid 15m > "$OUT/b10_decode_f${k}.log" 2>&1 || exit 1
  tail -1 "$OUT/b10_decode_f${k}.log"
done

echo "===== 官方评估（5 折） ====="
python - << 'EOF'
import sys, re, json
sys.path.insert(0, "scripts")
import official_iou_eval as oe
import src.config as config
import numpy as np
f1s = []
print(f"{'fold':>4} | {'F1':>6} {'Sens':>6} {'PPV':>6} | cfg")
for k in range(5):
    j = json.loads((config.OUTPUT_DIR / f"rank_events_v2_fold{k}_15m.json").read_text(encoding="utf-8"))
    cfg_name = j["best"]["name"]
    val_rows, gate_prob, clf_pri, true_sid, _, _ = oe.v2.prepare_fold(k)
    mm = re.match(r"w([\d.]+)_t([\d.]+)_g([\d.]+)_p([\d.]+)_d(\d+)_k(\d+)", cfg_name)
    cfg = (float(mm[1]), float(mm[2]), float(mm[3]), float(mm[4]), int(mm[5]), int(mm[6]))
    m = oe.official_metrics(sum((oe.v2.decode_session(r, gate_prob, cfg, clf_pri)[0] for r in val_rows), []), true_sid)
    f1s.append(m["f1"])
    print(f"{k:>4} | {m['f1']:>6.3f} {m['sensitivity']:>6.3f} {m['ppv']:>6.3f} | {cfg_name}")
print(f"均值 F1 = {np.mean(f1s):.3f}")
EOF
echo "===== B10 全链完成 ====="
