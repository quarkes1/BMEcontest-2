#!/bin/bash
# W3 阶段 1：L3b 两折训练（断点续跑：缓存按文件跳过）
set -u
PY=/d/Anaconda3/envs/bme/python.exe
cd /d/BMEtest || exit 1
echo "===== $(date '+%F %T') L3b 训练启动 ====="
$PY -u scripts/train_l3b.py --folds 0,1 --epochs 25 2>&1 | tee outputs/l3b_train_log.txt
echo "===== $(date '+%F %T') L3b 训练结束 exit=$? ====="
