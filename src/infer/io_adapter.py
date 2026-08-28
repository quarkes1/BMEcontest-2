# -*- coding: utf-8 -*-
"""测试集 I/O 适配层（spec §5）：外部数据格式与规范中间格式解耦。
接口文档发布后只改本文件。当前默认约定：
- 输入：目录，每会话一个子目录（会话 ID = 子目录名），内含 collect_data*.txt
- 输出：predict.csv，列名占位 externalid,startTime,endTime（毫秒），随文档调整
- 测试集空 externalid 会话跳过（训练集实测存在该现象）"""
import csv
import json
from pathlib import Path


def discover_sessions(test_dir: Path, adapter: dict):
    """扫描测试目录 → 会话 ID 列表（adapter.json 可覆盖目录结构约定）。"""
    d = Path(test_dir)
    if not d.exists():
        raise FileNotFoundError(f"test dir not found: {d}")
    pattern = adapter.get("session_dir_pattern", "*")
    return sorted(p.name for p in d.glob(pattern) if p.is_dir())


def resolve_externalid(session_id: str, adapter: dict):
    """会话 ID → externalid 占位映射。adapter.json 提供 id_map（可选）。"""
    id_map = adapter.get("id_map", {})
    return id_map.get(session_id, session_id)


def write_predict_csv(results: dict, out_path: Path, adapter: dict):
    """results: {sid: [(start_ms, end_ms)]} → predict.csv。"""
    cols = adapter.get("output_columns", ["externalid", "startTime", "endTime"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for sid, events in sorted(results.items()):
            ext = resolve_externalid(sid, adapter)
            for s, e in events:
                w.writerow([ext, s, e])


def load_adapter(path: Path = None) -> dict:
    if path and Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return {}
