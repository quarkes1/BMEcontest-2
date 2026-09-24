"""Training budget and serialization contracts."""

import numpy as np
import pytest
import torch

from scripts.research.representation_v2.dataset import TokenBag
from scripts.research.representation_v2.models import MotionBranch
from scripts.research.representation_v2.train import (
    collate_bags, fit_motion_branch, fit_robust_channels, load_checkpoint, save_checkpoint,
)


def _bag(index: int, value: float) -> TokenBag:
    values = np.full((2, 6, 125), value, dtype=np.float32)
    sample = np.ones((2, 125), dtype=bool)
    return TokenBag((f"s{index}", 0, 10_000), values, sample,
                    np.ones(2, bool), np.array([2_500, 7_500], np.int64),
                    0, 10_000, np.ones(2, np.float32))


def test_collate_and_normalization_ignore_padded_tokens():
    a, b = _bag(0, 1), _bag(1, 3)
    center, scale = fit_robust_channels([a, b])
    assert center.shape == scale.shape == (6,)
    assert np.all(scale > 0)
    batch = collate_bags([a, b], center, scale)
    assert batch[0].shape == (2, 2, 6, 125)
    assert batch[1][0].shape == (2, 2, 125)


def test_fixed_budget_and_checkpoint_round_trip(tmp_path):
    bags = [_bag(i, float(i % 2)) for i in range(8)]
    labels = np.array([i % 2 for i in range(8)], dtype=np.int8)
    subjects = [f"subject{i // 2}" for i in range(8)]
    result = fit_motion_branch(bags, labels, subjects, steps=3, seed=20260924, device="cpu")
    assert result["steps"] == 3
    assert len(result["losses"]) == 3
    assert all(np.isfinite(result["losses"]))
    path = tmp_path / "branch.pt"
    save_checkpoint(path, result, provenance={"train_subjects": subjects})
    restored, normalization = load_checkpoint(path, device="cpu")
    batch = collate_bags(bags[:2], *normalization)
    with torch.no_grad():
        assert torch.allclose(result["model"](*batch), restored(*batch))


def test_training_refuses_one_class_partition():
    bags = [_bag(i, 1) for i in range(4)]
    with pytest.raises(ValueError, match="both classes"):
        fit_motion_branch(bags, np.ones(4), [f"s{i}" for i in range(4)], steps=1, seed=1, device="cpu")


def test_small_training_set_can_learn_token_order():
    bags = []
    labels = []
    subjects = []
    for subject in range(8):
        for label in (0, 1):
            bag = _bag(len(bags), 0.0)
            bag.values[0, 0] = 1.0 if label else -1.0
            bag.values[1, 0] = -1.0 if label else 1.0
            bags.append(bag)
            labels.append(label)
            subjects.append(f"s{subject}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = fit_motion_branch(bags, np.asarray(labels), subjects, steps=80, seed=20260924, device=device)
    assert np.mean(result["losses"][-10:]) < np.mean(result["losses"][:10]) * 0.75


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_cuda_branch_score_tolerance():
    model = MotionBranch().eval()
    bags = [_bag(0, 1.0), _bag(1, 2.0)]
    center, scale = fit_robust_channels(bags)
    batch = collate_bags(bags, center, scale)
    with torch.no_grad():
        cpu = model(*batch)
        gpu = model.cuda()(batch[0].cuda(), (batch[1][0].cuda(), batch[1][1].cuda()),
                           batch[2].cuda(), batch[3].cuda()).cpu()
    assert torch.allclose(cpu, gpu, atol=1e-4, rtol=1e-4)
