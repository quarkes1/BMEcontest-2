#!/bin/bash
# 阶段三：7ch（gyro 4-16Hz Bite 触发器通道）全链——备份 6ch → 重建 7ch 缓存 → FD7ch 微调
set -e
cd "$(dirname "$0")/.."
source activate bme 2>/dev/null
OUT=outputs

# ---- 备份 6ch 缓存（mv 到 backup，可恢复） ----
for k in 0 1 2 3 4; do
  bak="cache/cand_windows/fold${k}_backup_6ch"
  if [ ! -d "$bak" ]; then
    mv "cache/cand_windows/fold${k}" "$bak"
    echo "备份 fold$k → $bak"
  fi
done

# ---- 重建 7ch 缓存（每折） ----
for k in 0 1 2 3 4; do
  echo "===== 重建 fold $k 7ch 缓存 ====="
  python -u scripts/build_candidate_windows.py --fold $k 2>&1 | tail -1
done

# ---- 7ch + FD 预训练微调 ----
mkdir -p "$OUT/exp_7ch"
for k in 0 1 2 3 4; do
  echo "===== fold $k 7ch FD 微调 ====="
  python -u scripts/train_ranker.py --fold $k --no-ppg --init-from checkpoints/fd_pretrained_s1_7ch.pt 2>&1 | tail -2
  cp "$OUT/mm_ranker_fold${k}_val.npz" "$OUT/seven7_fold${k}.npz"
done

# ---- 解码 + 官方评估 ----
for k in 0 1 2 3 4; do
  python -u scripts/rank_events_v2.py --fold $k --prior-grid 15m 2>&1 | tail -1
done
cp "$OUT/rank_events_v2_fold"*"_15m.json" "$OUT/exp_7ch/"

echo "===== 官方评估（7ch FD vs 6ch FD） ====="
python - << 'EOF'
import sys, re, json
sys.path.insert(0, "scripts")
import official_iou_eval as oe
import src.config as config
import numpy as np
print(f"{'fold':>5} | {'7ch+FD 现有':>11} {'6ch+FD 现有':>11}")
for k in range(5):
    j = json.loads((config.OUTPUT_DIR / f"rank_events_v2_fold{k}_15m.json").read_text(encoding="utf-8"))
    cfg_name = j["best"]["name"]
    val_rows, gate_prob, clf_pri, true_sid, _, _ = oe.v2.prepare_fold(k)
    mm = re.match(r"w([\d.]+)_t([\d.]+)_g([\d.]+)_p([\d.]+)_d(\d+)_k(\d+)", cfg_name)
    cfg = (float(mm[1]), float(mm[2]), float(mm[3]), float(mm[4]), int(mm[5]), int(mm[6]))
    m7 = oe.official_metrics(sum((oe.v2.decode_session(r, gate_prob, cfg, clf_pri)[0] for r in val_rows), []), true_sid)
    # 6ch FD（exp_fdpre JSON）
    j6 = json.loads((config.OUTPUT_DIR / "exp_fdpre" / f"rank_events_v2_fold{k}_15m.json").read_text(encoding="utf-8"))
    print(f"{k:>5} | {m7['f1']:>11.3f} | {j6['best']['f1']:>11.3f} | {cfg_name}")
EOF

echo "===== 完成（canonical 保持 7ch；如需恢复 6ch 见 backup 目录） ====="
