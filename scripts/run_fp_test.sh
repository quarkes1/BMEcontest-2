#!/bin/bash
set -u
PY=/d/Anaconda3/envs/bme/python.exe
cd /d/BMEtest || exit 1
$PY -u scripts/run_full_pipeline.py --folds 0,1 2>&1 | tee outputs/full_pipeline_test_log.txt
echo "exit=$?"
