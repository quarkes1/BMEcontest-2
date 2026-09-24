"""Fold-contained normalization and fixed-step branch training."""

from __future__ import annotations

import os
import json
import hashlib
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F

from scripts.research.representation_v2.dataset import TokenBag
from scripts.research.representation_v2.models import MotionBranch


def fit_robust_channels(bags: Sequence[TokenBag]) -> tuple[np.ndarray, np.ndarray]:
    """Estimate six channel scales from training bags only, with bounded RAM."""
    if not bags:
        raise ValueError("cannot fit normalization without training bags")
    per_bag = max(1, 500_000 // len(bags))
    chunks = []
    for bag in bags:
        valid = bag.sample_mask & bag.token_mask[:, None]
        vectors = bag.values.transpose(0, 2, 1)[valid]
        if len(vectors):
            chosen = np.linspace(0, len(vectors) - 1, min(len(vectors), per_bag), dtype=int)
            chunks.append(vectors[chosen])
    if not chunks:
        raise ValueError("training bags contain no valid motion samples")
    data = np.concatenate(chunks, axis=0).astype(np.float64)
    center = np.median(data, axis=0)
    scale = 1.4826 * np.median(np.abs(data - center), axis=0)
    return center.astype(np.float32), np.maximum(scale, 1e-3).astype(np.float32)


def collate_bags(
    bags: Sequence[TokenBag], center: np.ndarray, scale: np.ndarray
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    if not bags or np.asarray(center).shape != (6,) or np.asarray(scale).shape != (6,):
        raise ValueError("invalid batch or channel normalization")
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("channel normalization must be finite and positive")
    width = max(len(bag.token_mask) for bag in bags)
    batch = len(bags)
    values = np.zeros((batch, width, 6, 125), dtype=np.float32)
    sample_masks = np.zeros((batch, width, 125), dtype=bool)
    token_masks = np.zeros((batch, width), dtype=bool)
    times = np.full((batch, width), -1, dtype=np.int64)
    intervals = np.zeros((batch, 2), dtype=np.int64)
    for row, bag in enumerate(bags):
        count = len(bag.token_mask)
        if bag.values.shape != (count, 6, 125) or bag.sample_mask.shape != (count, 125):
            raise ValueError("token bag shape violates P1 contract")
        normalized = (bag.values - center[None, :, None]) / scale[None, :, None]
        normalized = np.where(bag.sample_mask[:, None, :], normalized, 0.0)
        values[row, :count] = normalized
        sample_masks[row, :count] = bag.sample_mask
        token_masks[row, :count] = bag.token_mask
        times[row, :count] = bag.token_center_ms
        intervals[row] = (bag.start_ms, bag.end_ms)
    return (torch.from_numpy(values), (torch.from_numpy(sample_masks), torch.from_numpy(token_masks)),
            torch.from_numpy(times), torch.from_numpy(intervals))


def _balanced_indices(labels: np.ndarray, subjects: Sequence[str], rng: np.random.Generator, size: int) -> np.ndarray:
    eligible: dict[int, dict[str, list[int]]] = {0: {}, 1: {}}
    for index, (label, subject) in enumerate(zip(labels, subjects)):
        eligible[int(label)].setdefault(str(subject), []).append(index)
    if not eligible[0] or not eligible[1]:
        raise ValueError("training partition needs both classes")
    chosen = []
    for position in range(size):
        by_subject = eligible[position % 2]
        subject = rng.choice(tuple(by_subject))
        chosen.append(int(rng.choice(by_subject[subject])))
    return np.asarray(chosen, dtype=int)


def fit_motion_branch(
    bags: Sequence[TokenBag], labels: np.ndarray, subjects: Sequence[str], *,
    steps: int, seed: int, device: str = "cpu", batch_size: int = 8,
    initial_model: MotionBranch | None = None,
    fixed_normalization: tuple[np.ndarray, np.ndarray] | None = None,
    freeze_token_encoder: bool = False,
    max_wall_seconds: float | None = None,
) -> dict[str, object]:
    if steps < 1 or batch_size != 8 or len(bags) != len(labels) or len(subjects) != len(bags):
        raise ValueError("invalid frozen training contract")
    classes = np.asarray(labels)
    if classes.ndim != 1 or not set(classes.tolist()) <= {0, 1}:
        raise ValueError("candidate labels must be binary")
    if len(set(subjects)) < 2:
        raise ValueError("training needs at least two subjects")
    if classes.min() == classes.max():
        raise ValueError("training partition needs both classes")
    if torch.device(device).type == "cpu":
        torch.set_num_threads(max(1, min(8, (os.cpu_count() or 2) - 2)))
    torch.manual_seed(seed)
    model = initial_model if initial_model is not None else MotionBranch()
    model.to(device).train()
    if freeze_token_encoder:
        for parameter in model.token_encoder.parameters():
            parameter.requires_grad_(False)
    center, scale = fixed_normalization if fixed_normalization is not None else fit_robust_channels(bags)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    losses: list[float] = []
    gradient_norms: list[float] = []
    for step in range(steps):
        if max_wall_seconds is not None and time.perf_counter() - start > max_wall_seconds:
            raise TimeoutError(f"training budget ended after {step} of {steps} steps")
        indices = _balanced_indices(classes, subjects, rng, batch_size)
        inputs = collate_bags([bags[int(i)] for i in indices], center, scale)
        values, masks, times, intervals = inputs
        values = values.to(device)
        masks = (masks[0].to(device), masks[1].to(device))
        times = times.to(device)
        intervals = intervals.to(device)
        targets = torch.as_tensor(classes[indices], dtype=torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(values, masks, times, intervals)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise RuntimeError("non-finite gradient")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        gradient_norms.append(float(norm.detach().cpu()))
    return {"model": model.eval(), "center": center, "scale": scale, "steps": steps,
            "losses": losses, "gradient_norms": gradient_norms, "wall_seconds": time.perf_counter() - start,
            "seed": seed, "train_subjects": tuple(sorted(set(subjects)))}


def predict_motion_branch(result: dict[str, object], bags: Sequence[TokenBag], *,
                          batch_size: int = 8, device: str = "cpu") -> np.ndarray:
    """Return ordered candidate probabilities without fitting on score bags."""
    if batch_size < 1:
        raise ValueError("prediction batch size must be positive")
    model = result["model"]
    if not isinstance(model, MotionBranch):
        raise ValueError("result lacks a MotionBranch")
    model.to(device).eval()
    center, scale = np.asarray(result["center"]), np.asarray(result["scale"])
    scores = []
    with torch.inference_mode():
        for start in range(0, len(bags), batch_size):
            values, masks, times, intervals = collate_bags(bags[start:start + batch_size], center, scale)
            logits = model(values.to(device), (masks[0].to(device), masks[1].to(device)),
                           times.to(device), intervals.to(device))
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores).astype(np.float64) if scores else np.empty(0, dtype=np.float64)


def save_checkpoint(path: Path, result: dict[str, object], *, provenance: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = result["model"]
    if not isinstance(model, MotionBranch):
        raise ValueError("result lacks a MotionBranch")
    def json_scalar(value: object) -> object:
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"checkpoint provenance is not JSON scalar: {type(value).__name__}")

    safe_provenance = json.loads(json.dumps(provenance, default=json_scalar))
    payload = {
        "schema": "representation-v2-motion-branch-v1",
        "weights": {key: value.cpu() for key, value in model.state_dict().items()},
        "center": np.asarray(result["center"]).tolist(),
        "scale": np.asarray(result["scale"]).tolist(),
        "steps": int(result["steps"]),
        "seed": int(result["seed"]),
        "provenance": safe_provenance,
    }
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def repair_numpy_checkpoint_provenance(path: Path, *, expected_identity: str) -> dict[str, object]:
    """Repair only this run's old NumPy subject scalars, with restricted loading.

    The known artifact was written by this project; this is never a general
    external-checkpoint loader. Model tensors are checked unchanged before the
    atomic replacement, and the replacement must pass plain weights-only load.
    """
    if not expected_identity:
        raise ValueError("repair requires a registered partition identity")
    path = Path(path)
    before_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with torch.serialization.safe_globals([np._core.multiarray.scalar, np.dtype, np.dtypes.StrDType]):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != "representation-v2-motion-branch-v1":
        raise ValueError("repair refuses an unknown checkpoint schema")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("identity") != expected_identity:
        raise ValueError("repair refuses a mismatched partition identity")
    if not isinstance(payload.get("weights"), dict) or not all(
        isinstance(value, torch.Tensor) for value in payload["weights"].values()
    ):
        raise ValueError("repair requires tensor-only model weights")

    def json_scalar(value: object) -> object:
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"unsupported provenance value: {type(value).__name__}")

    payload["provenance"] = json.loads(json.dumps(provenance, default=json_scalar))
    temporary = path.with_name(path.name + ".repair.tmp")
    try:
        torch.save(payload, temporary)
        verified = torch.load(temporary, map_location="cpu", weights_only=True)
        unchanged = payload["weights"].keys() == verified["weights"].keys() and all(
            torch.equal(payload["weights"][key], verified["weights"][key])
            for key in payload["weights"]
        )
        if not unchanged:
            raise RuntimeError("checkpoint repair altered model weights")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(path), "sha256_before": before_sha,
            "sha256_after": hashlib.sha256(path.read_bytes()).hexdigest(),
            "weights_unchanged": True, "identity": expected_identity}


def load_checkpoint(path: Path, *, device: str = "cpu") -> tuple[MotionBranch, tuple[np.ndarray, np.ndarray]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != "representation-v2-motion-branch-v1":
        raise ValueError("unknown branch checkpoint schema")
    model = MotionBranch()
    model.load_state_dict(payload["weights"], strict=True)
    center = np.asarray(payload["center"], dtype=np.float32)
    scale = np.asarray(payload["scale"], dtype=np.float32)
    if center.shape != (6,) or scale.shape != (6,) or (scale <= 0).any():
        raise ValueError("checkpoint normalization is invalid")
    return model.to(device).eval(), (center, scale)
