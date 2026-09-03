#!/bin/bash
# 打包准备链：重建候选窗缓存(6ch) → 5 折 FD 微调重训 → 打包 dist/
set -e
cd "$(dirname "$0")/.."
source activate bme 2>/dev/null
OUT=outputs

# ---- 1. cand_windows 5 折重建（6ch；validate_baselines/env 缓存已完成） ----
for k in 0 1 2 3 4; do
  echo "===== 重建 cand_windows fold $k ====="
  python -u scripts/build_candidate_windows.py --fold $k > "$OUT/pack_build_fold${k}.log" 2>&1 || {
    echo "fold$k FAILED"; tail -5 "$OUT/pack_build_fold${k}.log"; exit 1; }
  tail -1 "$OUT/pack_build_fold${k}.log"
done

# ---- 2. 5 折 FD 微调重训（打包权重） ----
mkdir -p models
for k in 0 1 2 3 4; do
  echo "===== fold $k 微调 ====="
  python -u scripts/train_ranker.py --fold $k --no-ppg --init-from checkpoints/fd_pretrained_s1.pt \
    > "$OUT/pack_train_fold${k}.log" 2>&1 || { echo "fold$k FAILED"; tail -5 "$OUT/pack_train_fold${k}.log"; exit 1; }
  tail -1 "$OUT/pack_train_fold${k}.log"
done

echo "===== 打包 dist/ ====="
bash scripts/run_pack_dist.sh
echo "===== 完成 ====="
