#!/bin/bash
# S0-2 zcr 信号验证：env_z 缓存（validate_baselines_z）→ 15m 网格解码 → 对比 baseline
# 深度分（canonical ens3）不动；只动提案层（z 低值池）+ 特征层（17 维）
set -e
cd "$(dirname "$0")/.."
source activate bme 2>/dev/null
OUT=outputs
export BME_ENV_CACHE=validate_baselines_z

mkdir -p "$OUT/exp_zcr"
for k in 0 1 2 3 4; do
  echo "===== fold $k zcr 解码 ====="
  python -u scripts/rank_events_v2.py --fold $k --prior-grid 15m 2>&1 | tail -2
done
cp "$OUT/rank_events_v2_fold"*"_15m.json" "$OUT/exp_zcr/"
echo "===== summarize zcr ====="
python scripts/summarize_v2.py 15m 2>&1 | tail -14
echo "===== 完成 ====="
