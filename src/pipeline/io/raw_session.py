"""Raw ``collect_data*.txt`` parsing without cache side effects."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

import numpy as np
import src.config as config

N_PPG = 44
_SESSION_NAME = re.compile(r"collect_data\d+_\d+_\d+\.txt")


@dataclass
class SessionData:
    acc: np.ndarray
    gyro: np.ndarray
    ppg: np.ndarray
    t_acc: np.ndarray
    t_ppg: np.ndarray
    imu_valid: np.ndarray
    ppg_valid: np.ndarray
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RawSessionSource:
    path: Path
    session_id: str
    subject_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))


def discover_raw_sessions(path: Path) -> tuple[RawSessionSource, ...]:
    """Discover sorted raw files; a multi-file directory shares one subject, not one session.

    A directory may also hold one level of ``sensorData-*``-style subdirectories (the
    competition dataset layout, e.g. the batch root passed as ``--official-input``);
    each subdirectory is then expanded with the same rules as a direct hit.
    """
    path = Path(path)
    if path.is_file():
        if not _SESSION_NAME.fullmatch(path.name):
            raise ValueError(f"supported raw session input is collect_data*.txt: {path}")
        return (RawSessionSource(path, path.parent.name, None),)
    if not path.is_dir():
        raise FileNotFoundError(path)
    sessions = _discover_in_directory(path)
    if not sessions:
        for child in sorted(child for child in path.iterdir() if child.is_dir()):
            sessions.extend(_discover_in_directory(child))
    if not sessions:
        raise FileNotFoundError(f"no collect_data txt in {path}")
    return tuple(sessions)


def _discover_in_directory(path: Path) -> list[RawSessionSource]:
    """One directory's direct sessions; a one-file directory names the session after it."""
    files = sorted(child for child in path.iterdir() if child.is_file() and _SESSION_NAME.fullmatch(child.name))
    if not files:
        return []
    if len(files) == 1:
        return [RawSessionSource(files[0], path.name, None)]
    return [RawSessionSource(file, f"{path.name}:{file.stem}", None) for file in files]


def _parse_collect_data_tsv(path: Path) -> SessionData:
    """The existing 53-column parser, retained without cache side effects."""
    acc_x, acc_y, acc_z, gx, gy, gz = [], [], [], [], [], []
    t_acc, t_ppg, imu_valid, ppg_valid = [], [], [], []
    ppg = [[] for _ in range(N_PPG)]
    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        header = handle.readline()
        assert "ACC_TIME" in header, f"bad header: {path}"
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 53:
                continue
            try:
                at, pt, gt = int(parts[0]), int(parts[1]), int(parts[2])
                values = list(map(float, parts[3:53]))
            except ValueError:
                continue
            ppg_values = values[:N_PPG]
            acc_values = values[N_PPG:N_PPG + 3]
            gyro_values = values[N_PPG + 3:N_PPG + 6]
            imu_ok = (at > 0 or gt > 0) and not all(value == 0 for value in acc_values)
            ppg_ok = pt > 0 and not all(value == 0 for value in ppg_values)
            acc_x.append(acc_values[0]); acc_y.append(acc_values[1]); acc_z.append(acc_values[2])
            gx.append(gyro_values[0]); gy.append(gyro_values[1]); gz.append(gyro_values[2])
            for index in range(N_PPG):
                ppg[index].append(ppg_values[index])
            t_acc.append(at if imu_ok else -1)
            t_ppg.append(pt if ppg_ok else -1)
            imu_valid.append(imu_ok); ppg_valid.append(ppg_ok)
    rows = len(t_acc)
    row_rate = config.IMU_ROW_RATE
    if rows and any(imu_valid):
        valid_timestamps = [timestamp for timestamp in t_acc if timestamp > 0]
        span_seconds = (max(valid_timestamps) - min(valid_timestamps)) / 1000.0
        if span_seconds > 60:
            row_rate = rows / span_seconds
    return SessionData(np.array([acc_x, acc_y, acc_z], dtype=np.float32), np.array([gx, gy, gz], dtype=np.float32), np.array(ppg, dtype=np.float32), np.array(t_acc, dtype=np.int64), np.array(t_ppg, dtype=np.int64), np.array(imu_valid, dtype=bool), np.array(ppg_valid, dtype=bool), {"path": str(path), "rows": rows, "row_rate": round(row_rate, 1)})


def load_raw_session(source: RawSessionSource) -> SessionData:
    """Load a raw source; reading it never creates a cache artifact."""
    if source.path.is_dir():
        sources = discover_raw_sessions(source.path)
        source = sources[0]
    if source.path.suffix.lower() != ".txt" or not _SESSION_NAME.fullmatch(source.path.name):
        raise ValueError("supported raw session input is collect_data*.txt")
    return _parse_collect_data_tsv(source.path)
