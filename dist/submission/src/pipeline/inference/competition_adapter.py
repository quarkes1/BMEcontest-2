"""历史 CSV 输入输出格式的隔离适配层。

本模块仅为源码兼容保留四列 CSV 格式，不属于正式竞赛提交接口。
2026-10-15 官方三列接口由 official_runner 和 official_api 实现。
历史 csv 适配器接受单个 collect_data*.txt 会话或目录树，输出每个检出事件的
start_time、end_time、start_ms、end_ms，包含本地 ISO 时间和原始毫秒时间戳。
新增适配器可通过 register_adapter 注册，不改变模型推理流程。"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
import re
from typing import Mapping, Protocol, runtime_checkable

UNREGISTERED_MESSAGE = "official competition input/output adapter is not registered"
DEFAULT_ADAPTER_NAME = "csv"

_SESSION_NAME = re.compile(r"^collect_data\d+_\d+_\d+\.txt$")


@runtime_checkable
class CompetitionAdapter(Protocol):
    """在外部输入输出格式与统一数据结构之间转换。"""

    def load(self, path: Path) -> tuple[Path, str | None]:
        """从输入解析原始数据路径及可选受试者标识。"""
        ...

    def dump(self, prediction: Mapping[str, object], path: Path) -> None:
        """将统一预测结果写为适配器指定的输出格式。"""
        ...


class UnsupportedCompetitionAdapter:
    """指定接口尚未注册时拒绝运行。"""

    def load(self, path: Path) -> tuple[Path, str | None]:
        raise NotImplementedError(UNREGISTERED_MESSAGE)

    def dump(self, prediction: Mapping[str, object], path: Path) -> None:
        raise NotImplementedError(UNREGISTERED_MESSAGE)


class CsvEventAdapter:
    """历史 CSV 适配器：每个检出的进食事件写入一行。

load 接受单个原始 TXT 或包含 sensorData-* 会话子目录的文件夹。
dump 按开始时间排序，输出四列本地 ISO 时间和原始毫秒时间戳，并采用带 BOM 的 UTF-8 编码。"""

    header = ("start_time", "end_time", "start_ms", "end_ms")

    def load(self, path: Path) -> tuple[Path, str | None]:
        path = Path(path)
        if not path.exists():
            raise ValueError(f"official input does not exist: {path}")
        if path.is_dir():
            if not any(path.rglob("collect_data*.txt")):
                raise ValueError(f"no collect_data*.txt session files under: {path}")
            return path, None
        if not _SESSION_NAME.fullmatch(path.name):
            raise ValueError(f"official input is not a collect_data*.txt session: {path.name}")
        return path, None

    def dump(self, prediction: Mapping[str, object], path: Path) -> None:
        target = _output_target(Path(path))
        target.parent.mkdir(parents=True, exist_ok=True)
        events = sorted(prediction.get("events", []), key=lambda row: int(row["start_ms"]))  # type: ignore[union-attr]
        with target.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(self.header)
            for event in events:
                start_ms, end_ms = int(event["start_ms"]), int(event["end_ms"])  # type: ignore[index]
                writer.writerow([_iso_local(start_ms), _iso_local(end_ms), start_ms, end_ms])


def _output_target(path: Path) -> Path:
    """若目标为目录，则在其中写入 predict_<stem>.csv。"""
    name = str(path)
    if path.is_dir() or name.endswith(("/", "\\")):
        stem = path.resolve().name or "sessions"
        return path / f"predict_{stem}.csv"
    return path


def resolve_output_path(official_input: Path, output: str | Path | None) -> Path:
    """确定 CSV 输出路径。

不传 --output 时写入当前目录下 predict/predict_<input-name>.csv。
已存在的目录或原始参数以路径分隔符结尾时，均视为目录目标。
使用原始字符串判断末尾分隔符，避免 pathlib 规范化后丢失该信息。"""
    source = Path(official_input)
    stem = source.name if source.is_dir() else source.stem
    filename = f"predict_{stem}.csv"
    if output is None or str(output) == "":
        return Path("predict") / filename
    text = str(output)
    candidate = Path(text)
    if text.endswith(("/", "\\")) or candidate.is_dir():
        return candidate / filename
    return candidate


def _iso_local(epoch_ms: int) -> str:
    moment = datetime.fromtimestamp(epoch_ms / 1000.0)
    return f"{moment:%Y-%m-%d %H:%M:%S}.{epoch_ms % 1000:03d}"


_REGISTERED: dict[str, CompetitionAdapter] = {}


def register_adapter(name: str, adapter: CompetitionAdapter) -> None:
    """按名称注册或替换适配器。

默认 csv 适配器始终可用，也可注册 official 等其他接口名称。"""
    _REGISTERED[str(name)] = adapter


def registered_adapter(name: str = DEFAULT_ADAPTER_NAME) -> CompetitionAdapter:
    """返回指定名称的适配器，默认回退到内置 csv 适配器。"""
    if str(name) in _REGISTERED:
        return _REGISTERED[str(name)]
    if str(name) == DEFAULT_ADAPTER_NAME:
        return CsvEventAdapter()
    raise NotImplementedError(UNREGISTERED_MESSAGE)
