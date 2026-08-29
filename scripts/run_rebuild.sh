#!/bin/bash
set -u
PY=/d/Anaconda3/envs/bme/python.exe
cd /d/BMEtest || exit 1
echo "===== $(date +%F %T) 缓存重建启动 ====="
$PY -u scripts/build_feature_cache.py 2>&1 | tee outputs/rebuild_baseline_log.txt
$PY -u scripts/build_gate_cache.py 2>&1 | tee outputs/rebuild_gate_log.txt
$PY -u scripts/build_raw_cache.py --folds 0 2>&1 | tee outputs/rebuild_raw_log.txt
echo "===== $(date +%F %T) 缓存重建结束 ====="
