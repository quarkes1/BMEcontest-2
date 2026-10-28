"""PyInstaller 精简入口：冻结运行时不训练，也不回退到仓库路径。"""
import multiprocessing
import sys
from src.pipeline.inference.official_runner import main

if __name__ == '__main__':
    multiprocessing.freeze_support()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    raise SystemExit(main())
