"""Isolated adapter boundary for the official competition input/output contract.

The audited competition materials do not define an official machine input/output
schema.  Until a real contract is published, the documented **default** is the
``csv`` adapter: official input is one ``collect_data*.txt`` session (or a folder
tree of them) and official output is a CSV with one row per detected eating event
(``start_time,end_time,start_ms,end_ms``; ISO local time with milliseconds plus
the raw epoch milliseconds).  When the official contract appears, implement a
concrete adapter, call :func:`register_adapter`, and rebuild ``dist/submission``;
no other component changes.
"""

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
    """Translate official competition I/O to/from canonical data."""

    def load(self, path: Path) -> tuple[Path, str | None]:
        """Return the canonical raw input path and optional subject id for an official input."""
        ...

    def dump(self, prediction: Mapping[str, object], path: Path) -> None:
        """Write an official competition output document for a canonical prediction."""
        ...


class UnsupportedCompetitionAdapter:
    """Refuses until the named official contract is registered."""

    def load(self, path: Path) -> tuple[Path, str | None]:
        raise NotImplementedError(UNREGISTERED_MESSAGE)

    def dump(self, prediction: Mapping[str, object], path: Path) -> None:
        raise NotImplementedError(UNREGISTERED_MESSAGE)


class CsvEventAdapter:
    """Default competition I/O: one CSV row per detected eating event.

    ``load`` accepts a single ``collect_data*.txt`` file or a folder containing
    session folders (``sensorData-*/collect_data*.txt``); ``dump`` writes
    ``start_time,end_time,start_ms,end_ms`` rows (ISO local time + epoch ms,
    UTF-8 with BOM so Excel opens it correctly), sorted by start time.
    """

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
    """A directory (existing, or trailing separator) gets ``predict_<stem>.csv`` inside."""
    name = str(path)
    if path.is_dir() or name.endswith(("/", "\\")):
        stem = path.resolve().name or "sessions"
        return path / f"predict_{stem}.csv"
    return path


def resolve_output_path(official_input: Path, output: str | Path | None) -> Path:
    """Where the CSV goes.

    No ``--output``: ``./predict/predict_<input-name>.csv`` (relative to the current
    directory).  A directory target receives the same ``predict_<input-name>.csv``
    file; it is recognized as a directory when it exists, or when the *raw* argument
    ends with a separator (``pathlib`` would strip a trailing slash, so the string
    form is what carries the intent).
    """
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
    """Register (or override) an adapter under ``name``.

    The default ``csv`` adapter is always available; register ``"official"`` (or any
    other name) here to take over once the real competition contract is published.
    """
    _REGISTERED[str(name)] = adapter


def registered_adapter(name: str = DEFAULT_ADAPTER_NAME) -> CompetitionAdapter:
    """Return the named adapter; the built-in csv adapter is the documented fallback."""
    if str(name) in _REGISTERED:
        return _REGISTERED[str(name)]
    if str(name) == DEFAULT_ADAPTER_NAME:
        return CsvEventAdapter()
    raise NotImplementedError(UNREGISTERED_MESSAGE)
