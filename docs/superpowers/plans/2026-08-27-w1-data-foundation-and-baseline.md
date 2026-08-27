# W1 数据基础设施与基线评估 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭建项目环境与目录骨架，完成数据管线地基（清单/解析/窗口/划分/评估指标），跑出 LightGBM 基线 F1@IoU0.25 数值。

**Architecture:** 数据以"会话清单（manifests）→ 流式 TSV 解析（loader，行空间即时间网格）→ 窗口切分+IoU 标签（windows）→ 受试者 5 折划分（splits）"为地基；评估指标（eval/metrics）先行并以单测锁定；LightGBM 基线复用同窗口管线验证全链路。

**Tech Stack:** Python 3.11（conda env `bme`）、pandas/numpy/scipy、scikit-learn、lightgbm、pytest；RTX 5060 Laptop 8GB（本计划基线阶段仅 CPU）。

**Spec:** `docs/superpowers/specs/2026-08-27-eating-detection-design.md`

## Global Constraints

- 工作目录：`D:\BMEtest`；所有脚本相对项目根运行
- Python 环境：conda env `bme`（Python **3.11**，不使用 base 的 3.13）
- `Data/` 目录**只读**（除 scripts/data_acquisition 迁移动作外永不改写原始数据）
- `Archieves/` **禁止任何改动**
- 采样常量：IMU 行率 ~105Hz（行空间即时间网格）、PPG 有效 ~24 行/s、窗口 5s=525 行、步长 1s=105 行、L3b 窗 30s、事件 IoU 阈值 0.25、窗口正样本 IoU≥0.5、灰区丢弃、随机种子 42
- 中文输出全部 UTF-8；Windows 下 CSV 一律 `encoding='utf-8'` + csv 标准库解析
- git 提交信息用英文，不 push（用户未要求推送）
- 测试运行：`pytest tests/ -v`（在 `bme` env 下）

---

### Task 1: conda 环境与依赖

**Files:**
- Create: `requirements.txt`（主依赖清单）

**Interfaces:**
- Produces: 可用的 `bme` env；`requirements.txt` 供 README 复现引用

- [ ] **Step 1: 创建 conda 环境**

```bash
conda create -n bme python=3.11 -y
conda activate bme
```

- [ ] **Step 2: 写 requirements.txt**

```
numpy>=1.26
pandas>=2.1
scipy>=1.11
scikit-learn>=1.4
lightgbm>=4.3
pytest>=8.0
matplotlib>=3.8
# W2+ 追加: torch(2.7+cu128)、onnx、onnxruntime、fastapi、uvicorn、pyinstaller
```

- [ ] **Step 3: 安装并验证**

```bash
conda activate bme
pip install -r requirements.txt
python -c "import numpy, pandas, scipy, sklearn, lightgbm; print('env ok')"
```

- [ ] **Step 4: 提交**

```bash
git add requirements.txt
git commit -m "chore: add base requirements for W1"
```

---

### Task 2: 目录重组与 .gitignore 扩展

**Files:**
- Create: `src/`、`src/data/`、`src/features/`、`src/models/`、`src/train/`、`src/eval/`、`src/infer/`、`web/`、`scripts/`、`scripts/data_acquisition/`、`docs/`、`models/`、`cache/`、`outputs/`、`tests/`（全部含 `__init__.py` 或 `.gitkeep` 占位）
- Modify: `.gitignore`
- Move: `Data/download_data.py`、`Data/download_log.txt`、`Data/DataTable.txt` → `scripts/data_acquisition/`；`Resources/附件/数据下载脚本使用方法.txt` → `scripts/data_acquisition/`（复制而非移动，Resources 保持原样）

**Interfaces:**
- Produces: 目录骨架；`.gitignore` 扩展版

- [ ] **Step 1: 建目录**

```bash
cd /d/BMEtest
mkdir -p src/data src/features src/models src/train src/eval src/infer
mkdir -p web/static scripts/data_acquisition docs models cache outputs tests
touch src/__init__.py src/data/__init__.py src/features/__init__.py \
      src/models/__init__.py src/train/__init__.py src/eval/__init__.py src/infer/__init__.py
touch models/.gitkeep cache/.gitkeep outputs/.gitkeep
```

- [ ] **Step 2: 迁移下载脚本**

```bash
mv Data/download_data.py Data/download_log.txt Data/DataTable.txt scripts/data_acquisition/
cp "Resources/附件/数据下载脚本使用方法.txt" scripts/data_acquisition/
```

- [ ] **Step 3: 扩展 .gitignore**

追加（保留已有 `Archieves/`、`Data/` 两行）：
```
*.7z
*.pyc
__pycache__/
cache/
models/
outputs/
```

- [ ] **Step 4: 提交**

```bash
git add -A
git commit -m "chore: reorganize project structure, extend gitignore"
```

---

### Task 3: 配置与清单加载（config + manifests）

**Files:**
- Create: `src/config.py`、`src/data/manifests.py`
- Test: `tests/test_manifests.py`

**Interfaces:**
- Consumes: Task 2 目录结构；`Data/` 下 3 个 CSV（已规范化 utf-8）
- Produces:
  - `config.DATA_DIR`、`config.CACHE_DIR`、`config.OUTPUT_DIR`、`config.RANDOM_SEED=42`、`config.IMU_ROW_RATE=105`、`config.WINDOW_ROWS=525`、`config.STRIDE_ROWS=105`、`config.IOU_POS=0.5`、`config.IOU_EVENT=0.25`、`config.SCENES = {"dominant", "nondominant"}`
  - `manifests.load_sensor_index() -> pd.DataFrame`（列：session_id, externalid, dir, start_ms, end_ms；已剔除 test 用户与黑名单）
  - `manifests.load_meals() -> pd.DataFrame`（列：meal_id, externalid, scene, dietary_type, before_ms, after_ms, tableware, duration_min；无效标签与 test 用户已剔除）
  - `manifests.load_users() -> pd.DataFrame`（去重）
  - `manifests.scene_of(wear_hand: str, dietary_hand: str) -> str`：wear==dietary → "dominant" 否则 "nondominant"

- [ ] **Step 1: 写失败测试 `tests/test_manifests.py`**

```python
# -*- coding: utf-8 -*-
import pytest
import pandas as pd
from src.data.manifests import scene_of, normalize_hand, load_meals

def test_scene_of():
    assert scene_of("右手佩戴", "右手") == "dominant"
    assert scene_of("左手佩戴", "右手") == "nondominant"
    assert scene_of("左手佩戴", "左手") == "dominant"

def test_normalize_hand():
    assert normalize_hand("左手佩戴") == "左手"
    assert normalize_hand("右手") == "右手"

def test_load_meals_excludes_test_and_invalid(tmp_path, monkeypatch):
    csv = tmp_path / "meals.csv"
    csv.write_text(
        "externalid,dietaryType,beforeTime,afterTime,drinkingVolume,satiety,foodName,"
        "tablewareType,dietaryHand,wearHand\n"
        "HNU21001,晚餐,1784457642000,1784458542000,300mL,刚刚好,米饭,筷子,右手,左手佩戴\n"
        "test,晚餐,1784457642000,1784457642000,300mL,刚刚好,米饭,筷子,右手,左手佩戴\n"
        "HNU21001,午餐,1784459642000,1784459642000,300mL,刚刚好,米饭,筷子,右手,左手佩戴\n",
        encoding="utf-8")
    monkeypatch.setattr("src.data.manifests.MEALS_CSV", str(csv))
    df = load_meals()
    assert len(df) == 1
    assert df.iloc[0]["scene"] == "nondominant"
    assert df.iloc[0]["duration_min"] == pytest.approx(15.0)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `conda activate bme && cd /d/BMEtest && pytest tests/test_manifests.py -v`
Expected: FAIL（`ModuleNotFoundError: src.data.manifests`）

- [ ] **Step 3: 实现 `src/config.py`**

```python
# -*- coding: utf-8 -*-
"""全局配置：路径与常量。所有模块从本文件取数。"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent          # D:\BMEtest
DATA_DIR = ROOT_DIR / "Data"
SENSOR_DIR = DATA_DIR / "t_zsstnnrj_sensororiginaldata_system附件0826_1857"
CACHE_DIR = ROOT_DIR / "cache"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_DIR = ROOT_DIR / "models"

RANDOM_SEED = 42
IMU_ROW_RATE = 105        # IMU 有效行率（行空间即时间网格, 行/s）
WINDOW_ROWS = 525         # 5s 窗口
STRIDE_ROWS = 105         # 1s 步长
PPG_WINDOW_ROWS = 24 * 30 # L3b 30s 窗口
IOU_POS = 0.5             # 窗口正样本 IoU 下限
IOU_EVENT = 0.25          # 事件匹配 IoU 阈值（组委会口径）
EVENT_MERGE_GAP_SEC = 30
EVENT_MIN_DUR_SEC = 45
BOUNDARY_DILATION_SEC = 6

SCENES = ("dominant", "nondominant")
```

- [ ] **Step 4: 实现 `src/data/manifests.py`**

```python
# -*- coding: utf-8 -*-
"""会话清单 / 标签 / 用户表加载与归一化。"""
import csv
import pandas as pd
import src.config as config

MEALS_CSV = config.DATA_DIR / "t_zsstnnrj_mealinfo_puadqog70826_1857.csv"
USERS_CSV = config.DATA_DIR / "t_zsstnnrj_userinfobean_5d4l0nmp0826_1857.csv"
INDEX_CSV = config.DATA_DIR / "t_zsstnnrj_sensororiginaldata_system0826_1857.csv"
BLACKLIST_FILE = config.OUTPUT_DIR / "data_quality_blacklist.txt"   # Task 8 产出，无则空

def normalize_hand(value: str) -> str:
    """'左手佩戴' -> '左手'; '右手' -> '右手'"""
    return str(value).replace("佩戴", "").strip()

def scene_of(wear_hand: str, dietary_hand: str) -> str:
    """wear==dietary -> dominant（惯用手场景），否则 nondominant"""
    return "dominant" if normalize_hand(wear_hand) == normalize_hand(dietary_hand) else "nondominant"

def load_blacklist() -> set:
    if BLACKLIST_FILE.exists():
        return {line.strip() for line in BLACKLIST_FILE.read_text(encoding="utf-8").splitlines() if line.strip()}
    return set()

def load_meals() -> pd.DataFrame:
    """用餐标签：剔除 test 用户与无效（before>=after 或时长<=0）记录，附加 scene/duration 列。"""
    with open(MEALS_CSV, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        ext = str(r["externalid"]).strip()
        if ext.lower() == "test":
            continue
        before, after = int(float(r["beforeTime"])), int(float(r["afterTime"]))
        if after <= before:
            continue
        out.append({
            "meal_id": f"{ext}_{before}",
            "externalid": ext,
            "scene": scene_of(r["wearHand"], r["dietaryHand"]),
            "dietary_type": str(r["dietaryType"]).strip(),
            "before_ms": before, "after_ms": after,
            "tableware": str(r["tablewareType"]).strip(),
            "duration_min": round((after - before) / 60000.0, 2),
        })
    return pd.DataFrame(out)

def load_users() -> pd.DataFrame:
    df = pd.read_csv(USERS_CSV, encoding="utf-8")
    df = df[df["externalid"].str.lower() != "test"].drop_duplicates(subset="externalid")
    return df

def load_sensor_index() -> pd.DataFrame:
    """传感器索引：session_id=zip 名 stem；剔除 test 用户与黑名单会话。"""
    df = pd.read_csv(INDEX_CSV, encoding="utf-8")
    df["session_id"] = df["sensorData"].map(lambda p: p.split("/")[-1].removesuffix(".zip"))
    df = df[df["externalid"].str.lower() != "test"]
    blacklist = load_blacklist()
    df = df[~df["session_id"].isin(blacklist)]
    return df
```

- [ ] **Step 5: 运行测试通过**

Run: `pytest tests/test_manifests.py -v`
Expected: 3 passed

- [ ] **Step 6: 提交**

```bash
git add src/config.py src/data/manifests.py tests/test_manifests.py
git commit -m "feat: add config and session/meal manifests with tests"
```

---

### Task 4: TSV 流式解析器（loader）

**Files:**
- Create: `src/data/loader.py`
- Test: `tests/test_loader.py`

**Interfaces:**
- Consumes: `config`、`manifests`；传感器 TSV 文件路径（真实数据或测试合成文件）
- Produces:
  - `loader.detect_binary(path) -> bool`：首 512 字节含非文本字节即 True（损坏文件判定）
  - `loader.load_session_tsv(txt_path) -> SessionData`
  - `SessionData` dataclass：`acc: np.ndarray (3,N)`、`gyro: np.ndarray (3,N)`、`ppg: np.ndarray (44,N)`、`t_acc: np.ndarray (N,)`（毫秒，0 行处为 -1）、`t_ppg: np.ndarray (N,)`、`imu_valid: np.ndarray (N, bool)`、`ppg_valid: np.ndarray (N, bool)`、`meta: dict`（path、rows、row_rate）
  - `loader.load_session(session_id) -> SessionData`：按 `config.SENSOR_DIR / session_id / collect_data*.txt` 定位

- [ ] **Step 1: 写失败测试 `tests/test_loader.py`**

```python
# -*- coding: utf-8 -*-
import numpy as np
import pytest
from src.data.loader import load_session_tsv, detect_binary

HEADER = "ACC_TIME\tPPG_TIME\tGYRO_TIME\t" + "\t".join(f"PPG{i}" for i in range(1, 45)) + \
         "\tACC_X\tACC_Y\tACC_Z\tGYRO_X\tGYRO_Y\tGYRO_Z"

def make_tsv(tmp_path, lines):
    p = tmp_path / "t.txt"
    p.write_text(HEADER + "\n" + "\n".join("\t".join(map(str, row)) for row in lines), encoding="utf-8")
    return str(p)

def test_load_and_masks(tmp_path):
    # 行1: ACC+GYRO 有效 PPG 全零；行2: 全有效；行3: 全零（末行置零）
    p = make_tsv(tmp_path, [
        [1000, 0, 1000] + [0]*44 + [1.0, 0, 0, 0.1, 0, 0],
        [1095, 1016, 1095] + [5]*44 + [2.0, 0, 0, 0.2, 0, 0],
        [0, 0, 0] + [0]*44 + [0, 0, 0, 0, 0, 0],
    ])
    s = load_session_tsv(p)
    assert s.acc.shape == (3, 3) and s.ppg.shape == (44, 3)
    assert s.imu_valid.tolist() == [True, True, False]
    assert s.ppg_valid.tolist() == [False, True, False]
    assert s.t_acc[2] == -1   # 无效行时间戳置 -1
    assert s.meta["rows"] == 3

def test_detect_binary(tmp_path):
    p = tmp_path / "b.txt"
    p.write_bytes(b"\x00\xff\xf9\x05" + b"x" * 100)
    assert detect_binary(str(p)) is True
    good = tmp_path / "g.txt"
    good.write_text(HEADER + "\n" + "\t".join(["1000", "0", "1000"] + ["0"]*44 + ["1"]*6), encoding="utf-8")
    assert detect_binary(str(good)) is False
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_loader.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现 `src/data/loader.py`**

```python
# -*- coding: utf-8 -*-
"""TSV 流式解析：行空间即时间网格，掩码标记占空比采样。"""
import re
from dataclasses import dataclass, field
import numpy as np
import src.config as config

N_PPG = 44

@dataclass
class SessionData:
    acc: np.ndarray          # (3, N)
    gyro: np.ndarray         # (3, N)
    ppg: np.ndarray          # (44, N)
    t_acc: np.ndarray        # (N,) 毫秒；无效行 -1
    t_ppg: np.ndarray        # (N,) 毫秒；无效行 -1
    imu_valid: np.ndarray    # (N,) bool
    ppg_valid: np.ndarray    # (N,) bool
    meta: dict = field(default_factory=dict)

def detect_binary(path, head=512):
    with open(path, "rb") as f:
        chunk = f.read(head)
    if not chunk:
        return False
    text_ratio = sum(1 for b in chunk if b in b"\t\n\r" or 32 <= b < 127) / len(chunk)
    return text_ratio < 0.9

def _find_collect_data(path):
    """返回目录内 collect_data*.txt（每会话恰 1 个）。"""
    import os
    names = [n for n in os.listdir(path) if re.match(r"collect_data\d+_\d+_\d+\.txt$", n)]
    if not names:
        raise FileNotFoundError(f"no collect_data txt in {path}")
    return os.path.join(path, sorted(names)[0])

def load_session_tsv(txt_path) -> SessionData:
    acc_x, acc_y, acc_z, gx, gy, gz = [], [], [], [], [], []
    t_acc, t_ppg, imu_valid, ppg_valid = [], [], [], []
    ppg = [[] for _ in range(N_PPG)]
    with open(txt_path, encoding="utf-8", errors="replace") as f:
        header = f.readline()
        assert "ACC_TIME" in header, f"bad header: {txt_path}"
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 50:
                continue
            try:
                at, pt, gt = int(parts[0]), int(parts[1]), int(parts[2])
                vals = list(map(float, parts[3:50]))
            except ValueError:
                continue
            ppg_vals = vals[:N_PPG]
            a = vals[N_PPG:N_PPG + 3]; g = vals[N_PPG + 3:N_PPG + 6]
            imu_ok = (at > 0 or gt > 0) and not all(v == 0 for v in a)
            ppg_ok = pt > 0 and not all(v == 0 for v in ppg_vals)
            acc_x.append(a[0]); acc_y.append(a[1]); acc_z.append(a[2])
            gx.append(g[0]); gy.append(g[1]); gz.append(g[2])
            for j in range(N_PPG):
                ppg[j].append(ppg_vals[j])
            t_acc.append(at if imu_ok else -1)
            t_ppg.append(pt if ppg_ok else -1)
            imu_valid.append(imu_ok); ppg_valid.append(ppg_ok)
    N = len(t_acc)
    row_rate = config.IMU_ROW_RATE
    if N and imu_valid.any():
        span = (max(t_acc) - min(t for t in t_acc if t > 0)) / 1000.0
        if span > 60:
            row_rate = N / span
    return SessionData(
        acc=np.array([acc_x, acc_y, acc_z], dtype=np.float32),
        gyro=np.array([gx, gy, gz], dtype=np.float32),
        ppg=np.array(ppg, dtype=np.float32),
        t_acc=np.array(t_acc, dtype=np.int64),
        t_ppg=np.array(t_ppg, dtype=np.int64),
        imu_valid=np.array(imu_valid, dtype=bool),
        ppg_valid=np.array(ppg_valid, dtype=bool),
        meta={"path": txt_path, "rows": N, "row_rate": round(row_rate, 1)})

def load_session(session_id: str) -> SessionData:
    d = config.SENSOR_DIR / session_id
    return load_session_tsv(_find_collect_data(str(d)))
```

- [ ] **Step 4: 运行测试通过**

Run: `pytest tests/test_loader.py -v`
Expected: 2 passed

- [ ] **Step 5: 提交**

```bash
git add src/data/loader.py tests/test_loader.py
git commit -m "feat: streaming TSV session loader with duty-cycle masks"
```

---

### Task 5: 窗口切分与 IoU 标签（windows）

**Files:**
- Create: `src/data/windows.py`
- Test: `tests/test_windows.py`

**Interfaces:**
- Consumes: `config`；SessionData 行空间与 meals DataFrame（before_ms/after_ms）
- Produces:
  - `windows.time_iou(t0_ms, t1_ms, before_ms, after_ms) -> float`：时间区间 IoU
  - `windows.iter_window_labels(session: SessionData, meals: list[tuple[int,int]], window_rows=525, stride_rows=105) -> Iterator[dict]`：yield `{"start_row": i, "end_row": i+W, "t0_ms": ..., "t1_ms": ..., "label": 0/1}`；窗口时间用该窗口内首个/末个有效 t_acc（无有效则跳过）；灰区（0<IoU<0.5）跳过

- [ ] **Step 1: 写失败测试 `tests/test_windows.py`**

```python
# -*- coding: utf-8 -*-
import numpy as np
from src.data.windows import time_iou, iter_window_labels
from src.data.loader import SessionData

def test_time_iou():
    assert time_iou(0, 10, 5, 15) == 0.5 / 1.5          # 交 5 / 并 15
    assert time_iou(0, 10, 20, 30) == 0.0
    assert time_iou(0, 10, 0, 10) == 1.0

def test_iter_window_labels_pos_neg_gray():
    N = 525 * 3 + 1
    t_acc = np.arange(N) * 10   # 每行 10ms -> 105 行 = 1.05s
    s = SessionData(
        acc=np.zeros((3, N), dtype=np.float32), gyro=np.zeros((3, N), dtype=np.float32),
        ppg=np.zeros((44, N), dtype=np.float32),
        t_acc=t_acc.astype(np.int64), t_ppg=np.full(N, -1, dtype=np.int64),
        imu_valid=np.ones(N, dtype=bool), ppg_valid=np.zeros(N, dtype=bool),
        meta={"row_rate": 100.0})
    meals = [(0, 600)]   # 0~0.6s 用餐
    out = list(iter_window_labels(s, meals, window_rows=525, stride_rows=105))
    labels = [w["label"] for w in out]
    assert labels[0] == 1     # 0~1.05s 与 0~0.6s IoU=0.57>=0.5
    assert labels[1] == 0     # 1.05~2.1s IoU=0
    assert len(out) >= 2      # 灰区窗口被跳过（0.6~1.05s 之间的窗口不存在）
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_windows.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 `src/data/windows.py`**

```python
# -*- coding: utf-8 -*-
"""窗口切分与 IoU 标签构造（行空间，行率 ~105Hz）。"""
from typing import Iterator, List, Tuple
import src.config as config
from src.data.loader import SessionData

def time_iou(t0, t1, before, after) -> float:
    inter = max(0.0, min(t1, after) - max(t0, before))
    union = max(t1, after) - min(t0, before)
    return inter / union if union > 0 else 0.0

def _window_times(t_acc, start, end):
    seg = t_acc[start:end]
    seg = seg[seg > 0]
    if len(seg) == 0:
        return None
    return int(seg[0]), int(seg[-1])

def iter_window_labels(session: SessionData, meals: List[Tuple[int, int]],
                       window_rows: int = None, stride_rows: int = None) -> Iterator[dict]:
    W = window_rows or config.WINDOW_ROWS
    S = stride_rows or config.STRIDE_ROWS
    N = session.acc.shape[1]
    for i in range(0, N - W + 1, S):
        times = _window_times(session.t_acc, i, i + W)
        if times is None:
            continue
        t0, t1 = times
        label = None
        for before, after in meals:
            iou = time_iou(t0, t1, before, after)
            if iou >= config.IOU_POS:
                label = 1
                break
            if iou > 0:
                label = None          # 灰区：丢弃，不参与训练
                break
        if label is None and any(time_iou(t0, t1, b, a) > 0 for b, a in meals):
            continue                   # 灰区跳过
        if label is None:
            label = 0
        yield {"start_row": i, "end_row": i + W, "t0_ms": t0, "t1_ms": t1, "label": label}
```

- [ ] **Step 4: 运行测试通过**

Run: `pytest tests/test_windows.py -v`
Expected: 2 passed

- [ ] **Step 5: 提交**

```bash
git add src/data/windows.py tests/test_windows.py
git commit -m "feat: window slicing with IoU-based labels (pos/gray/neg)"
```

---

### Task 6: 受试者 5 折划分（splits）

**Files:**
- Create: `src/data/splits.py`
- Test: `tests/test_splits.py`

**Interfaces:**
- Consumes: `manifests`（sensor index + meals）、`config`
- Produces:
  - `splits.build_folds() -> list[dict]`：每折 `{"fold": k, "train_sessions": [...], "val_sessions": [...], "train_meals": n, "val_meals": n, "val_scene_balance": {"dominant": n, "nondominant": n}}`；GroupKFold 按受试者；写入 `cache/splits/fold{k}.json`；相同 seed 可复现
  - `splits.load_folds() -> list[dict]`：读缓存，不存在则构建

- [ ] **Step 1: 写失败测试 `tests/test_splits.py`**

```python
# -*- coding: utf-8 -*-
import json
from src.data import splits

def test_build_folds_covers_all_subjects(tmp_path, monkeypatch):
    monkeypatch.setattr(splits.config, "CACHE_DIR", tmp_path)
    folds = splits.build_folds()
    assert len(folds) == 5
    all_subjects = set()
    for f in folds:
        for s in f["train_sessions"]:
            all_subjects.add(s)
        for s in f["val_sessions"]:
            all_subjects.add(s)
        assert f["train_sessions"].isdisjoint(f["val_sessions"])
    assert len(all_subjects) >= 40          # 真实数据 45 名受试者
    assert (tmp_path / "splits" / "fold0.json").exists()
    # 同一 seed 复现
    f2 = splits.build_folds()
    assert f2[0]["val_sessions"] == folds[0]["val_sessions"]
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_splits.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 `src/data/splits.py`**

```python
# -*- coding: utf-8 -*-
"""按受试者 GroupKFold 5 折划分，manifest 落盘可复现。"""
import json
import numpy as np
from sklearn.model_selection import GroupKFold
import src.config as config
from src.data import manifests

def _fold_path(k):
    return config.CACHE_DIR / "splits" / f"fold{k}.json"

def build_folds():
    index = manifests.load_sensor_index()
    meals = manifests.load_meals()
    # 以有会话数据的受试者为划分单元
    subjects = sorted(index["externalid"].unique())
    groups = index["externalid"].map({s: i for i, s in enumerate(subjects)}).to_numpy()
    kf = GroupKFold(n_splits=5)
    folds = []
    for k, (tr_idx, va_idx) in enumerate(kf.split(index, groups=groups)):
        tr_sessions = set(index.iloc[tr_idx]["session_id"])
        va_sessions = set(index.iloc[va_idx]["session_id"])
        tr_ext = set(index.iloc[tr_idx]["externalid"])
        va_ext = set(index.iloc[va_idx]["externalid"])
        tr_meals = meals[meals["externalid"].isin(tr_ext)]
        va_meals = meals[meals["externalid"].isin(va_ext)]
        fold = {
            "fold": k,
            "train_sessions": sorted(tr_sessions),
            "val_sessions": sorted(va_sessions),
            "train_meals": int(len(tr_meals)),
            "val_meals": int(len(va_meals)),
            "val_scene_balance": {s: int((va_meals["scene"] == s).sum()) for s in config.SCENES},
        }
        folds.append(fold)
        _fold_path(k).parent.mkdir(parents=True, exist_ok=True)
        _fold_path(k).write_text(json.dumps(fold, ensure_ascii=False, indent=1), encoding="utf-8")
    return folds

def load_folds():
    if not _fold_path(0).exists():
        return build_folds()
    return [json.loads(_fold_path(k).read_text(encoding="utf-8")) for k in range(5)]
```

- [ ] **Step 4: 运行测试通过**

Run: `pytest tests/test_splits.py -v`
Expected: 1 passed（注意：此测试读真实 manifests，~几秒）

- [ ] **Step 5: 提交**

```bash
git add src/data/splits.py tests/test_splits.py
git commit -m "feat: subject-wise group 5-fold splits with reproducible manifests"
```

---

### Task 7: 评估指标（eval/metrics）—— 全项目裁判，先行锁定

**Files:**
- Create: `src/eval/metrics.py`、`src/eval/__init__.py`（已建）
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: 无（纯函数）；事件格式 `(start_ms, end_ms)` 元组列表
- Produces:
  - `metrics.event_iou(pred, true) -> float`
  - `metrics.match_events(pred_events, true_events, iou_thr=0.25) -> tuple[list, list, list]`：`(matched_true_idx, unmatched_pred_idx, unmatched_true_idx)` 一对一贪心匹配
  - `metrics.compute_metrics(pred_events, true_events, iou_thr=0.25) -> dict`：`{"n_true","n_pred","n_tp","n_fp","n_fn","sensitivity","ppv","f1","mae_start_s","mae_end_s"}`；MAE 仅对匹配对计算，无匹配对时为 None

- [ ] **Step 1: 写失败测试 `tests/test_metrics.py`**

```python
# -*- coding: utf-8 -*-
import pytest
from src.eval.metrics import event_iou, match_events, compute_metrics

def test_event_iou():
    assert event_iou((0, 10), (5, 15)) == pytest.approx(5 / 15)
    assert event_iou((0, 10), (100, 200)) == 0.0

def test_match_events_one_to_one():
    preds = [(0, 100), (0, 100)]     # 两个重复预测
    trues = [(0, 100), (500, 600)]
    matched, unmatched_p, unmatched_t = match_events(preds, trues, iou_thr=0.25)
    assert matched == [0]            # 贪心：只匹配第一个真值
    assert unmatched_p == [1]
    assert unmatched_t == [1]

def test_compute_metrics_perfect_and_missing():
    m = compute_metrics([(0, 100), (200, 300)], [(0, 100), (200, 300)])
    assert m["f1"] == 1.0 and m["mae_start_s"] == 0.0
    m2 = compute_metrics([], [(0, 100)])
    assert m2["sensitivity"] == 0.0 and m2["ppv"] == 0.0 and m2["f1"] == 0.0
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 `src/eval/metrics.py`**

```python
# -*- coding: utf-8 -*-
"""事件级评估：IoU>0.25 一对一匹配 -> F1 + 起止 MAE（组委会口径）。"""
import src.config as config

def event_iou(pred, true) -> float:
    p0, p1 = pred; t0, t1 = true
    inter = max(0.0, min(p1, t1) - max(p0, t0))
    union = max(p1, t1) - min(p0, t0)
    return inter / union if union > 0 else 0.0

def match_events(pred_events, true_events, iou_thr=None):
    iou_thr = iou_thr if iou_thr is not None else config.IOU_EVENT
    # 按 IoU 降序贪心一对一匹配
    pairs = []
    for pi, p in enumerate(pred_events):
        for ti, t in enumerate(true_events):
            pairs.append((event_iou(p, t), pi, ti))
    pairs.sort(key=lambda x: -x[0])
    used_p, used_t = set(), set()
    matched_true = []
    for iou, pi, ti in pairs:
        if iou < iou_thr:
            break
        if pi in used_p or ti in used_t:
            continue
        used_p.add(pi); used_t.add(ti)
        matched_true.append(ti)
    unmatched_pred = [i for i in range(len(pred_events)) if i not in used_p]
    unmatched_true = [i for i in range(len(true_events)) if i not in used_t]
    return matched_true, unmatched_pred, unmatched_true

def compute_metrics(pred_events, true_events, iou_thr=None) -> dict:
    iou_thr = iou_thr if iou_thr is not None else config.IOU_EVENT
    matched, unmatched_p, unmatched_t = match_events(pred_events, true_events, iou_thr)
    n_true, n_pred, n_tp = len(true_events), len(pred_events), len(matched)
    sens = n_tp / n_true if n_true else 0.0
    ppv = n_tp / n_pred if n_pred else 0.0
    f1 = 2 * sens * ppv / (sens + ppv) if (sens + ppv) else 0.0
    mae_s = mae_e = None
    if matched:
        mae_s = sum(abs(pred_events[i][0] - true_events[j][0]) for i, j in
                    zip(sorted(set(range(n_pred)) - set(unmatched_p)), matched)) / n_tp / 1000.0
        mae_e = sum(abs(pred_events[i][1] - true_events[j][1]) for i, j in
                    zip(sorted(set(range(n_pred)) - set(unmatched_p)), matched)) / n_tp / 1000.0
    return {"n_true": n_true, "n_pred": n_pred, "n_tp": n_tp,
            "n_fp": len(unmatched_p), "n_fn": len(unmatched_t),
            "sensitivity": sens, "ppv": ppv, "f1": f1,
            "mae_start_s": mae_s, "mae_end_s": mae_e}
```

- [ ] **Step 4: 运行测试通过**

Run: `pytest tests/test_metrics.py -v`
Expected: 3 passed

- [ ] **Step 5: 提交**

```bash
git add src/eval/metrics.py tests/test_metrics.py
git commit -m "feat: event-level IoU matching and F1/MAE metrics with tests"
```

---

### Task 8: 全量数据质量校验（scripts/validate_data.py）

**Files:**
- Create: `scripts/validate_data.py`

**Interfaces:**
- Consumes: `manifests.load_sensor_index()`、`loader.detect_binary`、`loader.load_session_tsv`
- Produces: `outputs/data_quality.json`（每会话：binary/rows/dup_ratio/tail_zero_ratio/row_rate/header_ok）+ `outputs/data_quality_blacklist.txt`（二进制损坏 session_id 一行一个）；终端打印汇总表

- [ ] **Step 1: 实现脚本**

```python
# -*- coding: utf-8 -*-
"""全量数据质量校验：并行扫描 1165 会话，产出质量报告与黑名单。
运行：conda activate bme && python scripts/validate_data.py（预计 1-2 小时）"""
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import src.config as config
from src.data import manifests
from src.data.loader import detect_binary, load_session_tsv, _find_collect_data

def check_one(session_id: str) -> dict:
    d = config.SENSOR_DIR / session_id
    txt = _find_collect_data(str(d))
    rec = {"session_id": session_id}
    if detect_binary(txt):
        rec["binary"] = True
        return rec
    s = load_session_tsv(txt)
    N = s.acc.shape[1]
    acc = s.acc
    dup = 0.0
    if N > 1:
        diff = np.abs(np.diff(acc, axis=1)).sum(axis=0)
        dup = float((diff == 0).mean())
    tail_zero = 0
    for k in range(min(50, N), 0, -1):
        if not (s.imu_valid[N - k] or s.ppg_valid[N - k]):
            tail_zero = k
        else:
            break
    rec.update({
        "binary": False, "rows": N,
        "dup_ratio": round(dup, 4),
        "tail_zero_rows": tail_zero,
        "imu_valid_ratio": round(float(s.imu_valid.mean()), 4),
        "ppg_valid_ratio": round(float(s.ppg_valid.mean()), 4),
        "row_rate": s.meta.get("row_rate"),
    })
    return rec

def main():
    t0 = time.time()
    index = manifests.load_sensor_index()
    ids = sorted(index["session_id"].tolist())
    print(f"扫描 {len(ids)} 个会话...")
    results = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for rec in ex.map(check_one, ids, chunksize=8):
            results.append(rec)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUT_DIR / "data_quality.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    blacklist = [r["session_id"] for r in results if r.get("binary")]
    (config.OUTPUT_DIR / "data_quality_blacklist.txt").write_text(
        "\n".join(blacklist) + "\n", encoding="utf-8")
    bad_tail = [r["session_id"] for r in results if r.get("tail_zero_rows", 0) >= 50]
    print(f"完成, 用时 {time.time()-t0:.0f}s")
    print(f"二进制损坏: {len(blacklist)} -> {blacklist}")
    print(f"尾随置零>=50行: {len(bad_tail)}")
    print(f"重复行率: 中位 {np.median([r['dup_ratio'] for r in results if 'dup_ratio' in r]):.3f}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 小样冒烟（100 会话）**

```bash
conda activate bme && cd /d/BMEtest
python -c "
import src.config as c, json
from scripts.validate_data import check_one
from src.data import manifests
ids = manifests.load_sensor_index()['session_id'].tolist()[:100]
res = [check_one(i) for i in ids]
json.dump(res, open(c.OUTPUT_DIR/'smoke100.json','w'))
print('binary:', sum(r['binary'] for r in res), 'of', len(res))
"
```
Expected: 无异常退出；打印 binary 计数（应为 0 或少量）

- [ ] **Step 3: 全量运行（后台）**

```bash
nohup python scripts/validate_data.py > outputs/validate_log.txt 2>&1 &
```

- [ ] **Step 4: 确认产物并提交**

```bash
head outputs/data_quality_blacklist.txt; python -c "import json; d=json.load(open('outputs/data_quality.json')); print(len(d), 'sessions')"
git add scripts/validate_data.py
git commit -m "feat: full-dataset quality scan producing report and blacklist"
```

---

### Task 9: LightGBM 基线（特征 v1 + 5 折训练评估）

**Files:**
- Create: `src/features/baseline_features.py`、`src/models/baseline_lgbm.py`、`scripts/run_baseline.py`
- Test: `tests/test_baseline_features.py`

**Interfaces:**
- Consumes: Task 4-7（loader/windows/splits/metrics）
- Produces:
  - `baseline_features.window_features(session, start_row, end_row) -> np.ndarray`（37 维，见下）
  - `baseline_lgbm.train_one_fold(train_data, val_data) -> model`（lightgbm Booster，early stopping 用 val 事件 F1 的窗口代理指标）
  - `baseline_lgbm.predict_session(model, session, meals_meta) -> list[tuple[int,int]]`：滑窗打分 → 阈值 → 合并 → 事件列表
  - `run_baseline.py`：5 折全流程 → `outputs/baseline_report.json` + 控制台表

- [ ] **Step 1: 写失败测试 `tests/test_baseline_features.py`**

```python
# -*- coding: utf-8 -*-
import numpy as np
from src.features.baseline_features import window_features
from src.data.loader import SessionData

def test_window_features_shape():
    N = 1100
    s = SessionData(
        acc=np.random.randn(3, N).astype(np.float32),
        gyro=np.random.randn(3, N).astype(np.float32),
        ppg=np.random.randn(44, N).astype(np.float32),
        t_acc=(np.arange(N) * 10).astype(np.int64),
        t_ppg=np.full(N, -1, dtype=np.int64),
        imu_valid=np.ones(N, dtype=bool), ppg_valid=np.zeros(N, dtype=bool),
        meta={})
    f = window_features(s, 0, 525)
    assert f.shape == (37,)
    assert np.isfinite(f).all()
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_baseline_features.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 `src/features/baseline_features.py`**

```python
# -*- coding: utf-8 -*-
"""LightGBM 基线特征 v1：窗口统计量（37 维）。"""
import numpy as np
from scipy.signal import butter, sosfilt

def _bandpass(signal, low, high, fs):
    sos = butter(4, [low, high], btype="band", fs=fs, output="sos")
    return sosfilt(sos, signal)

def window_features(session, start, end) -> np.ndarray:
    fs = session.meta.get("row_rate", 105.0)
    a = session.acc[:, start:end]
    g = session.gyro[:, start:end]
    am = np.linalg.norm(a, axis=0)                 # 合加速度
    gm = np.linalg.norm(g, axis=0)
    am_bp = _bandpass(am, 0.5, 2.0, fs)            # 0.5-2Hz 咀嚼带
    grav = np.median(a, axis=1, keepdims=True)     # 重力估计（窗口均值近似）
    la = a - grav                                   # 线加速度
    lam = np.linalg.norm(la, axis=0)
    tilt = np.degrees(np.arccos(np.clip(grav[:, 0] / (np.linalg.norm(grav[:, 0]) + 1e-9), -1, 1)))

    feats = [
        np.mean(am), np.std(am), np.percentile(am, 90), np.max(am),
        np.mean(gm), np.std(gm), np.percentile(gm, 90),
        np.mean(np.abs(am_bp)), np.std(am_bp),
        np.mean(lam), np.std(lam), np.percentile(lam, 90), np.max(lam),
        tilt, np.std(np.degrees(np.arccos(np.clip(grav[:, 0] / (np.linalg.norm(grav[:, 0]) + 1e-9), -1, 1)))),
        np.mean(a, axis=1)[0], np.mean(a, axis=1)[1], np.mean(a, axis=1)[2],
        np.std(a, axis=1)[0], np.std(a, axis=1)[1], np.std(a, axis=1)[2],
        np.std(g, axis=1)[0], np.std(g, axis=1)[1], np.std(g, axis=1)[2],
        np.mean(am_bp**2), np.max(np.abs(am_bp)),
        np.count_nonzero(np.diff(np.signbit(am_bp))),      # 过零率
        float(session.imu_valid[start:end].mean()),        # IMU 有效率
        float(session.ppg_valid[start:end].mean()),        # PPG 有效率
        np.mean(session.ppg[:, start:end], axis=1)[:10].mean(),   # 前10通道均值
        np.std(session.ppg[:, start:end], axis=1)[:10].mean(),
        np.max(np.abs(session.ppg[:, start:end]), axis=1)[:10].mean(),
    ]
    return np.array(feats, dtype=np.float32)
```

- [ ] **Step 4: 运行测试通过**

Run: `pytest tests/test_baseline_features.py -v`
Expected: 1 passed

- [ ] **Step 5: 实现 `src/models/baseline_lgbm.py`**

```python
# -*- coding: utf-8 -*-
"""LightGBM 窗口分类基线：训练 + 滑窗推理 -> 事件列表。"""
import numpy as np
import lightgbm as lgb
import src.config as config

def train_one_fold(X, y, Xv, yv):
    ds = lgb.Dataset(X, label=y)
    dv = lgb.Dataset(Xv, label=yv, reference=ds)
    params = {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
              "num_leaves": 63, "max_depth": 7, "feature_fraction": 0.8,
              "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
              "seed": config.RANDOM_SEED}
    model = lgb.train(params, ds, num_boost_round=300, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
    return model

def _probs_to_events(probs, t0s, t1s, threshold):
    """滑窗概率 -> 事件列表（起止 ms，膨胀 BOUNDARY_DILATION_SEC）。"""
    on = probs >= threshold
    events = []
    i = 0
    while i < len(on):
        if on[i]:
            j = i
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            events.append((int(t0s[i]), int(t1s[j])))
            i = j + 1
        else:
            i += 1
    dil = config.BOUNDARY_DILATION_SEC * 1000
    merged = []
    for e in events:
        if merged and e[0] - merged[-1][1] <= config.EVENT_MERGE_GAP_SEC * 1000:
            merged[-1] = (merged[-1][0], e[1])
        else:
            merged.append(e)
    return [(max(0, s - dil), e + dil) for s, e in merged
            if (e - s) >= config.EVENT_MIN_DUR_SEC * 1000]

def predict_session(model, windows, threshold=0.5):
    """windows: list[dict] 与 iter_window_labels 同构（含 start_row/end_row/t0_ms/t1_ms）。
    返回 (事件列表, 概率序列)"""
    X = [w["feat"] for w in windows]
    if not X:
        return [], []
    probs = model.predict(np.vstack(X))
    return (_probs_to_events(probs, [w["t0_ms"] for w in windows],
                             [w["t1_ms"] for w in windows], threshold), probs)
```

- [ ] **Step 6: 实现 `scripts/run_baseline.py`**

```python
# -*- coding: utf-8 -*-
"""LightGBM 基线 5 折主流程。运行：conda activate bme && python scripts/run_baseline.py"""
import json
import random
import numpy as np
import src.config as config
from src.data import manifests, splits, loader, windows
from src.features.baseline_features import window_features
from src.models.baseline_lgbm import train_one_fold, predict_session
from src.eval.metrics import compute_metrics

def collect_windows(session_ids, meals_df, neg_ratio=3.0, seed=config.RANDOM_SEED):
    rng = random.Random(seed)
    X, y = [], []
    for sid in session_ids:
        s = loader.load_session(sid)
        ext = {v: k for k, v in zip(manifests.load_sensor_index()["session_id"],
                                    manifests.load_sensor_index()["externalid"])}[sid]
        meal_list = meals_df[meals_df["externalid"] == ext][["before_ms", "after_ms"]].to_numpy().tolist()
        for w in windows.iter_window_labels(s, meal_list):
            w["feat"] = window_features(s, w["start_row"], w["end_row"])
            if w["label"] == 1:
                X.append(w["feat"]); y.append(1)
            elif w["label"] == 0:
                X.append(w["feat"]); y.append(0)
    return np.vstack(X), np.array(y)

def main():
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    folds = splits.load_folds()
    meals = manifests.load_meals()
    report = {"folds": []}
    for f in folds:
        print(f"=== fold {f['fold']}: {f['val_meals']} val meals ===")
        Xtr, ytr = collect_windows(f["train_sessions"], meals)
        # 负采样 3:1
        pos_idx = np.where(ytr == 1)[0]
        neg_idx = np.where(ytr == 0)[0]
        keep_neg = np.random.RandomState(config.RANDOM_SEED).choice(
            neg_idx, size=min(len(neg_idx), int(len(pos_idx) * 3.0)), replace=False)
        idx = np.concatenate([pos_idx, keep_neg])
        model = train_one_fold(Xtr[idx], ytr[idx], Xtr[idx[:1000]], ytr[idx[:1000]])
        # 验证集滑窗推理
        pred_events, true_events = [], []
        for sid in f["val_sessions"]:
            s = loader.load_session(sid)
            ext = next(row for row in manifests.load_sensor_index().itertuples() if row.session_id == sid).externalid
            ml = meals[meals["externalid"] == ext][["before_ms", "after_ms"]].to_numpy().tolist()
            ws = list(windows.iter_window_labels(s, ml))
            for w in ws:
                w["feat"] = window_features(s, w["start_row"], w["end_row"])
            evs, _ = predict_session(model, ws, threshold=0.5)
            pred_events.extend(evs)
            true_events.extend([(int(a), int(b)) for a, b in ml])
        m = compute_metrics(pred_events, true_events)
        print(f"  F1={m['f1']:.3f} sens={m['sensitivity']:.3f} ppv={m['ppv']:.3f} "
              f"MAE_start={m['mae_start_s']}s n_tp={m['n_tp']}/{m['n_true']}")
        report["folds"].append({"fold": f["fold"], **{k: v for k, v in m.items() if v is not None or True}})
    (config.OUTPUT_DIR / "baseline_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("report -> outputs/baseline_report.json")

if __name__ == "__main__":
    main()
```

- [ ] **Step 7: 冒烟运行（单折小样）**

```bash
conda activate bme && cd /d/BMEtest
python -c "
from src.data import manifests, splits, loader
f = splits.load_folds()[0]
print('val sessions:', len(f['val_sessions']), 'val meals:', f['val_meals'])
"
```
Expected: 打印 5 折划分摘要，无异常

- [ ] **Step 8: 全量 5 折运行（后台，预计 1-2 小时）**

```bash
nohup python scripts/run_baseline.py > outputs/baseline_log.txt 2>&1 &
```

- [ ] **Step 9: 查看报告并提交**

```bash
cat outputs/baseline_report.json | head -50
git add src/features/baseline_features.py src/models/baseline_lgbm.py scripts/run_baseline.py tests/test_baseline_features.py
git commit -m "feat: LightGBM baseline with 37-dim window features and 5-fold evaluation"
```

---

### Task 10: 文档与首次整体提交

**Files:**
- Create: `README.md`（骨架：环境安装/数据准备/复现入口/目录说明）、`docs/数据处理说明.md`（异常数据剔除记录——引用 validate_data 结果）

**Interfaces:**
- Consumes: 全部前序任务产物

- [ ] **Step 1: 写 `docs/数据处理说明.md`**

内容要点（据实填写数字）：
```
# 数据处理说明
- 数据来源：HUAWEI Research 平台下载（脚本见 scripts/data_acquisition/），2026-06-20~08-13，1165 会话/45 受试者
- 下载后规范化：GBK→utf-8、表头提行、附件路径与本地目录 100% 校验
- 剔除规则：test 用户（2 会话/1 无效标签）；无效标签（after<=before，N 条）；二进制损坏会话（M 个，见 outputs/data_quality_blacklist.txt）
- 占空比说明：PPG 78% 行全零为采样调度而非缺失，掩码处理
- 时间网格：行空间 ~105Hz（IMU）/有效 PPG ~24 行/s；窗口 5s=525 行，步长 1s
- 标签规则：窗口与事件 IoU>=0.5 正 / =0 负 / 灰区丢弃
```

- [ ] **Step 2: 写 `README.md` 骨架**

```markdown
# BMEcontest-2 — 基于智能手表传感器的进食检测

第十一届全国大学生生物医学工程创新设计竞赛 · 智能穿戴与运动健康赛道 · 赛题二。

## 目录
- `src/` 核心代码（数据管线/特征/模型/评估/推理）
- `scripts/` 实验脚本（数据校验/基线/5折评估）
- `web/` 网页应用（W3 交付）
- `ReferenceDocs/` 参考文献笔记
- `docs/` 设计文档与数据处理说明

## 环境
（conda env bme, Python 3.11；详见 requirements.txt）

## 快速开始
（数据准备 → 数据校验 → 基线 5 折：三行命令）

## 复现
（待 W4 补全完整一键复现说明）
```

- [ ] **Step 3: 整体提交**

```bash
git add README.md docs/ requirements.txt
git commit -m "docs: project README skeleton and data processing notes"
```

- [ ] **Step 4: 收尾核对**

```bash
git log --oneline | head -12
git status   # 应只有未跟踪的 outputs/cache（已忽略）
```
Expected: 12 个左右提交；工作区干净

---

## Self-Review 记录

- **Spec 覆盖**：W1 范围对应 spec §4（训练与评估地基）与 §7（W1 里程碑）；W2+ 任务（L1-L4 模型、web、打包）将在后续计划中展开
- **占位符检查**：Task 8 Step 4 / Task 10 中"N 条/M 个"为运行后回填的数字（数据产物而非占位逻辑），其余无 TBD
- **类型一致性**：`SessionData` 字段（acc/gyro/ppg/t_acc/t_ppg/imu_valid/ppg_valid/meta）在 Task 4 定义、Task 5/9 消费一致；事件元组 `(start_ms, end_ms)` 在 Task 5/7/9 一致；`iter_window_labels` 的 yield 字段（start_row/end_row/t0_ms/t1_ms/label）在 Task 5/9 一致
