"""Read-only P0 audit and bounded fit-count projection."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from src.data.manifests import load_sensor_index
from src.pipeline.challenger.protection import protected_snapshot


def projected_fits(*, outer_wrappers: int, inner_folds: int, arms: int, meta_folds: int) -> dict[str, int]:
    if min(outer_wrappers, inner_folds, arms, meta_folds) < 1:
        raise ValueError("fit topology dimensions must be positive")
    partitions = outer_wrappers * inner_folds
    p2 = partitions * arms
    # The selected arm's all-T refit is identical to a P2 fit and can be reused.
    p3_additional = partitions * meta_folds
    confirmation = partitions * (meta_folds + 1)
    target_steps = 2_000 * (p2 + p3_additional + confirmation)
    return {
        "p2_target_fits": p2,
        "p3_primary_additional_fits": p3_additional,
        "p3_confirmation_fits": confirmation,
        "fd_source_fits": 1,
        "total_target_steps_if_all_gates_pass": target_steps,
        "total_steps_including_fd": target_steps + 10_000,
    }


def required_seconds_per_step(gpu_hours: int, steps: int) -> float:
    if gpu_hours <= 0 or steps <= 0:
        raise ValueError("GPU hours and steps must be positive")
    return gpu_hours * 3_600 / steps


def _fingerprint_small(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": path.as_posix(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _source_file(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": path.as_posix(), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def audit(root: Path) -> dict[str, object]:
    root = root.resolve()
    release = json.loads((root / "release/event_stack_incumbent.json").read_text(encoding="utf-8"))
    if release.get("run_key") != "160afaf81debf1ee" or abs(float(release.get("f1", -1)) - 0.6514285714285715) > 1e-12:
        raise RuntimeError("release anchor differs from registered incumbent")

    index = load_sensor_index()
    session_subjects = dict(zip(index["session_id"].astype(str), index["externalid"].astype(str)))
    folds = []
    all_train_subjects: set[str] = set()
    for outer in range(5):
        split = json.loads((root / f"cache/splits/fold{outer}.json").read_text(encoding="utf-8"))
        sessions = list(map(str, split["train_sessions"]))
        missing = sorted(set(sessions) - set(session_subjects))
        if missing:
            raise RuntimeError(f"outer fold {outer} lacks session-to-subject mapping: {missing[:3]}")
        subjects = sorted({session_subjects[s] for s in sessions})
        all_train_subjects.update(subjects)
        folds.append({"outer_fold": outer, "train_sessions": len(sessions), "train_subjects": len(subjects),
                      "subject_ids_sha256": hashlib.sha256("\n".join(subjects).encode()).hexdigest()})

    fd = root / "FDdatasets/doi-10.48804-CN8VBB/v1.0"
    fd_source = [fd / f"FD-I/{name}.pkl" for name in ("X_L", "X_R", "Y_L", "Y_R")]
    missing_fd = [p for p in fd_source if not p.is_file()]
    if missing_fd:
        raise RuntimeError(f"FD-I source missing: {missing_fd}")
    readme = (fd / "00_ReadMe.txt").read_text(encoding="utf-8")
    if "34 participants" not in readme or "64 Hz" not in readme or "labelled as 1" not in readme:
        raise RuntimeError("FD source description has unexpected identity")

    checkpoint = root / "checkpoints/fd_pretrained_s1.pt"
    protected = protected_snapshot(root)
    digest = hashlib.sha256(json.dumps(protected, sort_keys=True).encode()).hexdigest()
    fits = projected_fits(outer_wrappers=5, inner_folds=4, arms=4, meta_folds=3)
    free = shutil.disk_usage(root).free
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    return {
        "status": "P0_READY" if free >= 20 * 1024**3 else "BUDGET_REVIEW_REQUIRED",
        "base_git_sha": git_sha,
        "release": {"run_key": release["run_key"], "f1": release["f1"]},
        "protected_tree_sha256": digest,
        "protected_tree_entries": len(protected),
        "target": {"subject_count": len(all_train_subjects), "folds": folds, "session_cache_files": len(list((root / "cache/sessions").glob("*.npz")))},
        "fd": {"dataset": "doi:10.48804/CN8VBB v1", "source": [_source_file(p) for p in fd_source],
               "readme": _fingerprint_small(fd / "00_ReadMe.txt"), "reported_participants_fd_i": 34,
               "sample_rate_hz": 64, "label_1": "eating gesture", "label_2": "drinking gesture",
               "license": "CC BY-NC-ND 4.0 (official dataset page)",
               "license_url": "https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/CN8VBB",
               "old_checkpoint": _source_file(checkpoint) if checkpoint.exists() else None,
               "old_checkpoint_reusable": False,
               "old_checkpoint_reason": "provenance is insufficient; safe weights-only load fails"},
        "hardware": {"free_disk_bytes": free},
        "budget": {**fits, "gpu_hours_limit": 24, "wall_hours_limit": 48,
                   "seconds_per_step_ceiling_if_all_gates_pass": required_seconds_per_step(24, fits["total_steps_including_fd"]),
                   "cache_bytes_limit": 12 * 1024**3, "checkpoints_bytes_limit": 2 * 1024**3},
        "limits": ["no outer GT use", "no sealed diagnostic rows for training", "no legacy checkpoint reuse"],
    }
