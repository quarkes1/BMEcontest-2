"""Fixed Q0/Q1 target-only P2 probe; FD arms are separately gated."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from scripts.research.representation_v2.crossfit import build_candidate_ledgers
from scripts.research.representation_v2.dataset import load_candidate_bags
from scripts.research.representation_v2.evaluate import probe_metrics
from scripts.research.representation_v2.models import MotionBranch
from scripts.research.representation_v2.train import fit_motion_branch, predict_motion_branch, save_checkpoint


def _identity(fold: int, inner: int, subjects: tuple[str, ...], *, plan_sha: str, code_sha: str) -> str:
    payload = {"fold": fold, "inner": inner, "train_subjects": subjects,
               "plan_sha256": plan_sha, "code_sha": code_sha, "seed": 20260924,
               "steps": 2000, "representation": "motion-branch-v1"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def run_target_probe_partition(root: Path, output: Path, data: object, config: object, *,
                               inner: int, train: tuple[str, ...], score: tuple[str, ...],
                               plan_sha: str, code_sha: str, device: str) -> dict[str, object]:
    """Complete one T/V Q0/Q1 comparison, recording per-arm held-out scores."""
    identity = _identity(config.outer_fold, inner, train, plan_sha=plan_sha, code_sha=code_sha)
    destination = output / "p2_probe" / f"fold{config.outer_fold}_inner{inner}.json"
    if destination.exists():
        prior = json.loads(destination.read_text(encoding="utf-8"))
        if prior.get("identity") != identity:
            raise RuntimeError("P2 partition evidence identity mismatch")
        for arm in ("q0", "q1"):
            if not (output / "p2_probe" / f"fold{config.outer_fold}_inner{inner}_{arm}.pt").is_file():
                raise RuntimeError("P2 partition evidence has a missing checkpoint")
        return prior
    t, v = build_candidate_ledgers(data, config, train, score)
    if not len(t.candidates) or not len(v.candidates):
        raise RuntimeError("P2 candidate ledger is empty")
    t_bags = load_candidate_bags(root, t.candidates)
    v_bags = load_candidate_bags(root, v.candidates)
    t_valid = np.asarray([bool(bag.token_mask.any()) for bag in t_bags])
    v_valid = np.asarray([bool(bag.token_mask.any()) for bag in v_bags])
    prevalence = float(np.mean(t.labels))
    torch.manual_seed(20260924)
    initial = MotionBranch()
    reports: dict[str, object] = {}
    for arm, frozen in (("q0", True), ("q1", False)):
        result = fit_motion_branch(
            [bag for bag, keep in zip(t_bags, t_valid) if keep], t.labels[t_valid], t.groups[t_valid],
            steps=2000, seed=20260924, device=device, initial_model=deepcopy(initial),
            freeze_token_encoder=frozen, max_wall_seconds=8 * 3600)
        scores = np.full(len(v_bags), prevalence, dtype=np.float64)
        valid_bags = [bag for bag, keep in zip(v_bags, v_valid) if keep]
        scores[v_valid] = predict_motion_branch(result, valid_bags, device=device)
        checkpoint = output / "p2_probe" / f"fold{config.outer_fold}_inner{inner}_{arm}.pt"
        save_checkpoint(checkpoint, result, provenance={
            "identity": identity, "arm": arm, "fit_subjects": list(train),
            "score_subjects": list(score), "plan_sha256": plan_sha, "code_sha": code_sha,
            "source": "target_only", "outer_calls": 0})
        reports[arm] = {
            "metrics": probe_metrics(v.labels, scores),
            "scores": scores.tolist(), "optimizer_steps": int(result["steps"]),
            "training_wall_seconds": result["wall_seconds"],
            "last_100_loss_mean": float(np.mean(result["losses"][-100:])),
            "max_gradient_norm": float(np.max(result["gradient_norms"])),
            "checkpoint": checkpoint.relative_to(root).as_posix(),
        }
    rows = [{"sid": candidate.event.sid, "start_ms": candidate.event.start_ms,
             "end_ms": candidate.event.end_ms, "subject": str(group), "label": int(label),
             "valid": bool(valid)}
            for candidate, group, label, valid in zip(v.candidates, v.groups, v.labels, v_valid)]
    payload = {
        "identity": identity, "fold": config.outer_fold, "inner": inner,
        "train_subjects": list(train), "score_subjects": list(score),
        "candidate_count": len(v.candidates), "train_candidate_count": len(t.candidates),
        "micro_threshold": t.micro_threshold, "train_prevalence": prevalence,
        "train_valid_count": int(t_valid.sum()), "score_valid_count": int(v_valid.sum()),
        "rows": rows, "arms": reports, "outer_calls": 0,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return payload
