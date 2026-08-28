#!/bin/bash
# W1 流水线：数据校验 -> 特征缓存 -> LightGBM 基线 5 折（断点跳过已完成阶段）
# 阶段门禁：任一阶段失败/产物不完整即中止，绝不带病进入下一阶段
set -u
PY=/d/Anaconda3/envs/bme/python.exe
cd /d/BMEtest || exit 1

echo "===== $(date '+%F %T') W1 流水线启动 ====="
rm -f outputs/baseline_report.json   # 清掉旧报告，防止误读为最终结果

# 阶段 1: 数据校验（产出 data_quality.json 后跳过）
if [ -f outputs/data_quality.json ]; then
  echo "SKIP validate (outputs/data_quality.json 已存在)"
else
  $PY -u scripts/validate_data.py 2>&1 | tee outputs/validate_log.txt
  V_EXIT=${PIPESTATUS[0]}
  echo "validate exit=$V_EXIT"
  if [ "$V_EXIT" != "0" ] || [ ! -f outputs/data_quality.json ]; then
    echo "[中止] 数据校验失败，流水线停止"
    exit 1
  fi
fi

# 阶段 2: 特征缓存（脚本内部按会话跳过已存在 npz）
$PY -u scripts/build_feature_cache.py 2>&1 | tee outputs/feature_cache_log.txt
CACHE_EXIT=${PIPESTATUS[0]}
N_NPZ=$(ls cache/baseline_features/*.npz 2>/dev/null | wc -l)
echo "cache exit=$CACHE_EXIT npz_count=$N_NPZ"
if [ "$CACHE_EXIT" != "0" ] || [ "$N_NPZ" -lt 1100 ]; then
  echo "[中止] 特征缓存不完整（npz=$N_NPZ），不进入基线阶段"
  exit 1
fi

# 阶段 3: 基线 5 折
$PY -u scripts/run_baseline.py 2>&1 | tee outputs/baseline_log.txt
echo "baseline exit=${PIPESTATUS[0]}"

echo "===== $(date '+%F %T') W1 流水线结束 ====="
