#!/bin/bash
# IT1 链：L3a 满配模型 fold 0 训练（60 轮）→ 后处理网格搜索
set -u
PY=/d/Anaconda3/envs/bme/python.exe
cd /d/BMEtest || exit 1
echo "===== $(date '+%F %T') IT1 链启动 ====="
$PY -u scripts/train_l3a.py --folds 0 --epochs 60 --model large 2>&1 | tee outputs/l3a_it1_log.txt
$PY -u scripts/tune_postprocess.py --fold 0 --model large --build-val 2>&1 | tee outputs/tune_log.txt
echo "===== $(date '+%F %T') IT1 链结束 ====="
