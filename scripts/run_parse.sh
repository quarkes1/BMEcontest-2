#!/bin/bash
set -u
PY=/d/Anaconda3/envs/bme/python.exe
cd /d/BMEtest || exit 1
$PY -u scripts/parse_sessions.py 2>&1 | tee outputs/parse_log.txt
echo "parse exit=$?"
