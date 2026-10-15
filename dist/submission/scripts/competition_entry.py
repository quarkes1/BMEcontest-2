"""Thin PyInstaller entry: no training or repository-path fallback in frozen mode."""
import multiprocessing
import sys
from src.pipeline.inference.official_runner import main

if __name__ == '__main__':
    multiprocessing.freeze_support()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    raise SystemExit(main())
