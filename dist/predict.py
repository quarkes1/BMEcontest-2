# -*- coding: utf-8 -*-
"""进食事件检测推理入口（滑窗管线部署版——主系统）。

由 predict_slide 实现（全覆盖滑窗 + HGB bag + 密度 + 复核），本文件仅转发参数。
旧版（检测即排序 + FD 深度模型）存档为 predict_legacy.py。
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from predict_slide import main

if __name__ == "__main__":
    main()
