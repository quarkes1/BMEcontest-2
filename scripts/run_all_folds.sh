#!/bin/bash
# 5 折全链：重建缓存（gate 全体会话打分）→ 重训排序头（meta 含 gate_prob）→ v2 解码
# 用法：source activate bme && bash scripts/run_all_folds.sh [起始折]
set -e
START=${1:-0}
for k in $(seq $START 4); do
  echo "===== fold $k ====="
  python -u scripts/build_candidate_windows.py --fold $k || exit 1
  python -u scripts/train_ranker.py --fold $k --no-ppg || exit 1
  python -u scripts/rank_events_v2.py --fold $k || exit 1
done
echo "===== 汇总 ====="
python scripts/summarize_v2.py
