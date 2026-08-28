#!/bin/bash
# W2 流水线：门控特征缓存 -> L1/L2 拟合评估 -> L3a 两折训练 -> 姿态曲线
# 阶段门禁：任一阶段失败/产物缺失即中止
set -u
PY=/d/Anaconda3/envs/bme/python.exe
cd /d/BMEtest || exit 1

echo "===== $(date '+%F %T') W2 流水线启动 ====="

# 阶段 1: 门控特征缓存
$PY -u scripts/build_gate_cache.py 2>&1 | tee outputs/gate_cache_log.txt
if [ ! -f cache/gate_features/build_stats.json ]; then
  echo "[中止] 门控缓存未完成，流水线停止"
  exit 1
fi

# 阶段 2: L1/L2 拟合评估
$PY -u scripts/fit_gates.py 2>&1 | tee outputs/fit_gates_log.txt
if [ ! -f outputs/l1_l2_report.json ]; then
  echo "[中止] L1/L2 拟合未完成，流水线停止"
  exit 1
fi

# 阶段 3: L3a 两折训练
$PY -u scripts/train_l3a.py --folds 0,1 --epochs 25 2>&1 | tee outputs/l3a_train_log.txt
if [ ! -f outputs/l3a_report.json ]; then
  echo "[中止] L3a 训练未完成，流水线停止"
  exit 1
fi

# 阶段 4: 姿态解算曲线
$PY -u scripts/pose_curves.py 2>&1 | tee outputs/pose_curves_log.txt

echo "===== $(date '+%F %T') W2 流水线结束 ====="
