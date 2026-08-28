# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec：bme_predict.exe（ONNX 后端，无 torch）。
构建：pyinstaller scripts/bme_predict.spec --noconfirm
依赖：numpy/scipy/sklearn/onnxruntime（torch 仅训练，不进包）"""
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hiddenimports = (
    collect_submodules("sklearn") +
    collect_submodules("onnxruntime") +
    collect_submodules("scipy.signal") +
    collect_submodules("scipy.ndimage")
)

datas = collect_data_files("sklearn") + collect_data_files("onnxruntime")

a = Analysis(
    ["scripts/bme_predict.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["torch", "torchvision", "lightgbm", "pandas", "matplotlib"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="bme_predict",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)
