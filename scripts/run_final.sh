#!/bin/bash
# 最终训练链（2026-08-29 晚）：L3a ResNet 5 折 → L3b BiGRU 5 折 → 后处理调优 → 全链路消融
# 纪律：缓存预建与训练重叠（后台 4 进程防内存尖峰）；每折训完即删原始缓存；
# 训练 GPU 驻留数据集；磁盘 <85% 警戒由看门狗负责。
set -u
PY=/d/Anaconda3/envs/bme/python.exe
cd /d/BMEtest || exit 1
echo "===== $(date '+%F %T') 最终训练链启动 ====="

# ---- L3a ResNet 5 折（模型齐备则跳过） ----
if [ -f models/l3a_cnn_resnet_fold4.pt ]; then
  echo "L3a 5 折模型齐备，跳过 L3a 阶段"
else
for k in 0 1 2 3 4; do
  BUILD_PID=""
  nxt=$((k+1))
  if [ $nxt -le 4 ]; then
    $PY -u scripts/build_raw_cache.py --folds $nxt --workers 4 2>&1 | tee outputs/build_fold${nxt}_log.txt &
    BUILD_PID=$!
  fi
  $PY -u scripts/train_l3a.py --folds $k --epochs 60 --model resnet --data gpu 2>&1 | tee outputs/l3a_resnet_fold${k}_log.txt
  if [ ! -f models/l3a_cnn_resnet_fold${k}.pt ]; then echo "[中止] L3a fold $k 未产出模型"; exit 1; fi
  rm -rf cache/l3a_raw/fold$k
  if [ -n "$BUILD_PID" ]; then wait $BUILD_PID; fi
done
fi

# ---- L3b BiGRU 5 折（模型齐备则跳过） ----
if [ -f models/l3b_v2_fold4.pt ]; then
  echo "L3b 5 折模型齐备，跳过 L3b 阶段"
else
for k in 0 1 2 3 4; do
  BUILD_PID=""
  nxt=$((k+1))
  if [ $nxt -le 4 ]; then
    $PY -u scripts/build_ppg_cache.py --folds $nxt --workers 4 2>&1 | tee outputs/build_l3b_fold${nxt}_log.txt &
    BUILD_PID=$!
  fi
  $PY -u scripts/train_l3b.py --folds $k --epochs 50 --model v2 --data gpu 2>&1 | tee outputs/l3b_v2_fold${k}_log.txt
  if [ ! -f models/l3b_v2_fold${k}.pt ]; then echo "[中止] L3b fold $k 未产出模型"; exit 1; fi
  rm -rf cache/l3b_raw/fold$k
  if [ -n "$BUILD_PID" ]; then wait $BUILD_PID; fi
done
fi

# ---- 后处理调优 + 全链路 5 折消融 ----
$PY -u scripts/tune_postprocess.py --fold 0 --model resnet --build-val 2>&1 | tee outputs/tune_log.txt
$PY -u scripts/run_full_pipeline.py --folds 0,1,2,3,4 2>&1 | tee outputs/full_pipeline_log.txt

echo "===== $(date '+%F %T') 最终训练链结束 ====="
