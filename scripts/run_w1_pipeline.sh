#!/bin/bash
# W1 剩余流水线：数据校验 -> 特征缓存 -> LightGBM 基线 5 折（断点跳过已完成阶段）
# 用分离进程方式启动（防被杀）：详见会话记录
set -u
PY=/d/Anaconda3/envs/bme/python.exe
cd /d/BMEtest || exit 1

echo "===== $(date '+%F %T') W1 流水线启动 ====="

# 阶段 1: 数据校验（产出 data_quality.json 后跳过）
if [ -f outputs/data_quality.json ]; then
  echo "SKIP validate (outputs/data_quality.json 已存在)"
else
  $PY -u scripts/validate_data.py 2>&1 | tee outputs/validate_log.txt
  echo "validate exit=${PIPESTATUS[0]}"
fi

# 阶段 2: 特征缓存（脚本内部按会话跳过已存在 npz）
$PY -u scripts/build_feature_cache.py 2>&1 | tee outputs/feature_cache_log.txt
echo "cache exit=${PIPESTATUS[0]}"

# 阶段 3: 基线 5 折
$PY -u scripts/run_baseline.py 2>&1 | tee outputs/baseline_log.txt
echo "baseline exit=${PIPESTATUS[0]}"

echo "===== $(date '+%F %T') W1 流水线结束 ====="
