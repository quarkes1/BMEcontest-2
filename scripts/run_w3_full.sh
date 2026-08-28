#!/bin/bash
# W3 收尾链：L3a folds 2-4 补训 → L3b folds 2-4 补训 → 全链路 5 折+消融 → ONNX 导出
set -u
PY=/d/Anaconda3/envs/bme/python.exe
cd /d/BMEtest || exit 1
echo "===== $(date '+%F %T') W3 收尾链启动 ====="

$PY -u scripts/train_l3a.py --folds 2,3,4 --epochs 25 2>&1 | tee outputs/l3a_train_log2.txt
if [ ! -f models/l3a_cnn_fold4.pt ]; then echo "[中止] L3a 补训未完成"; exit 1; fi

$PY -u scripts/train_l3b.py --folds 2,3,4 --epochs 25 2>&1 | tee outputs/l3b_train_log2.txt
if [ ! -f models/l3b_ppgnn_fold4.pt ]; then echo "[中止] L3b 补训未完成"; exit 1; fi

$PY -u scripts/run_full_pipeline.py --folds 0,1,2,3,4 2>&1 | tee outputs/full_pipeline_log.txt
if [ ! -f outputs/full_pipeline_report.json ]; then echo "[中止] 全链路报告未生成"; exit 1; fi

PYTHONIOENCODING=utf-8 $PY -u scripts/export_onnx.py --fold 0 --smoke 2>&1 | tee outputs/export_onnx_log.txt

echo "===== $(date '+%F %T') W3 收尾链结束 ====="
