#!/bin/bash
# S1：MO bite 预训练迁移 → 5 折 MM-Ranker 微调 → 15m 解码 → 对比 ens3 baseline
set -e
cd "$(dirname "$0")/.."
source activate bme 2>/dev/null
OUT=outputs

mkdir -p "$OUT/exp_mo"
for k in 0 1 2 3 4; do
  echo "===== fold $k MO 预训练微调 ====="
  python -u scripts/train_ranker.py --fold $k --no-ppg --init-from models/mo_bite.pt 2>&1 | tail -3
  cp "$OUT/mm_ranker_fold${k}_val.npz" "$OUT/mo_pre_fold${k}.npz"
done

echo "===== 解码 15m ====="
for k in 0 1 2 3 4; do
  python -u scripts/rank_events_v2.py --fold $k --prior-grid 15m 2>&1 | tail -1
done
cp "$OUT/rank_events_v2_fold"*"_15m.json" "$OUT/exp_mo/"
echo "===== summarize ====="
python scripts/summarize_v2.py 15m 2>&1 | tail -14

echo "===== 恢复 canonical = ens3 ====="
for k in 0 1 2 3 4; do
  cp "$OUT/archive_best_20260831/exp_ens3/rank_events_v2_fold${k}_15m.json" "$OUT/rank_events_v2_fold${k}_15m.json"
  cp "$OUT/archive_best_20260831/mm_ranker_fold${k}_val.npz" "$OUT/mm_ranker_fold${k}_val.npz"
done
echo "===== 完成 ====="
