"""Training budget and serialization contracts."""

import numpy as np
import pytest
import torch

from scripts.research.representation_v2.dataset import TokenBag
from scripts.research.representation_v2.models import MotionBranch
from scripts.research.representation_v2.train import (
    _balanced_indices, collate_bags, fit_motion_branch, fit_robust_channels, load_checkpoint,
    predict_motion_branch, predict_motion_logits,
    repair_numpy_checkpoint_provenance, save_checkpoint,
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
    assert torch.load(path, weights_only=True)["model_config"] == {"time_coverage_mode": "sample_time"}
    restored, normalization = load_checkpoint(path, device="cpu")
    assert restored.time_coverage_mode == "sample_time"
    batch = collate_bags(bags[:2], *normalization)
    with torch.no_grad():
        assert torch.allclose(result["model"](*batch), restored(*batch))


def test_prediction_is_chunk_invariant():
    bags = [_bag(i, float(i % 2)) for i in range(5)]
    result = fit_motion_branch(bags, np.array([0, 1, 0, 1, 0]),
                               [f"subject{i}" for i in range(5)], steps=1, seed=4)
    one = predict_motion_branch(result, bags, batch_size=1)
    many = predict_motion_branch(result, bags, batch_size=5)
    assert one.shape == (5,)
    assert np.all((one >= 0) & (one <= 1))
    np.testing.assert_allclose(one, many, atol=1e-6)


def test_balanced_indices_selects_subject_before_available_class():
    labels = np.asarray([0, 0, 0, 1])
    subjects = ["only_zero_a", "only_zero_b", "mixed", "mixed"]
    indices = _balanced_indices(labels, subjects, np.random.default_rng(7), 12_000)
    selected_subjects = np.asarray(subjects)[indices]
    selected_labels = labels[indices]
    subject_counts = {subject: int((selected_subjects == subject).sum())
                      for subject in set(subjects)}
    assert max(subject_counts.values()) - min(subject_counts.values()) < 400
    mixed = selected_labels[selected_subjects == "mixed"]
    assert abs(int((mixed == 0).sum()) - int((mixed == 1).sum())) < 200
    # Subjects with one available class are valid fallbacks, so the total batch
    # need not contain equal class counts.
    assert int((selected_labels == 0).sum()) != int((selected_labels == 1).sum())


def test_training_reports_sampling_history_and_short_run_convergence_status():
    bags = [_bag(i, float(i % 2)) for i in range(4)]
    observations = []
    result = fit_motion_branch(bags, np.asarray([0, 0, 1, 1]),
                               ["only_zero", "mixed", "mixed", "only_one"],
                               steps=100, seed=7,
                               observation_callback=lambda model, center, scale, step:
                               observations.append((model.training, center.copy(), scale.copy(), step)))
    assert result["convergence"] == "NOT_ASSESSED"
    assert result["convergence_summary"] == {
        "prior_400_loss_mean": None,
        "recent_400_loss_mean": None,
        "relative_improvement": None,
    }
    assert all(isinstance(count, int) for count in result["sample_counts"])
    assert sum(result["sample_counts"]) == 800
    history = result["training_history"]
    assert len(history) == 1
    assert history[0]["step"] == 100
    assert np.isfinite(history[0]["loss_mean"])
    assert sum(history[0]["cumulative_sample_counts"]["classes"].values()) == 800
    assert sum(history[0]["cumulative_sample_counts"]["subjects"].values()) == 800
    assert observations[0][0] is False
    assert observations[0][3] == 100


def test_checkpoint_optionally_serializes_training_observations(tmp_path):
    model = MotionBranch()
    path = tmp_path / "observations.pt"
    result = {
        "model": model, "center": np.zeros(6), "scale": np.ones(6), "steps": 100,
        "seed": 7, "training_history": [{"step": 100, "loss_mean": 0.5}],
        "convergence": "NOT_ASSESSED", "sample_counts": [3, 4],
    }
    save_checkpoint(path, result, provenance={"identity": "test"})
    payload = torch.load(path, weights_only=True)
    assert payload["training_history"] == [{"step": 100, "loss_mean": 0.5}]
    assert payload["convergence"] == "NOT_ASSESSED"
    assert payload["sample_counts"] == [3, 4]
    assert isinstance(load_checkpoint(path)[0], MotionBranch)


def test_predict_motion_logits_preserves_large_finite_values():
    bags = [_bag(0, 1.0), _bag(1, 2.0)]
    model = MotionBranch().eval()
    model.forward = lambda values, masks, times, intervals: torch.full(
        (values.shape[0],), 100.0, device=values.device
    )
    result = {"model": model, "center": np.zeros(6, np.float32), "scale": np.ones(6, np.float32)}
    logits = predict_motion_logits(result, bags)
    np.testing.assert_array_equal(logits, np.asarray([100.0, 100.0]))
    assert np.all(predict_motion_branch(result, bags) == 1.0)


def test_checkpoint_safe_load_accepts_numpy_subject_identifiers(tmp_path):
    bags = [_bag(i, float(i % 2)) for i in range(4)]
    result = fit_motion_branch(bags, np.asarray([0, 1, 0, 1]),
                               ["s0", "s1", "s2", "s3"], steps=1, seed=1)
    path = tmp_path / "numpy_subject.pt"
    save_checkpoint(path, result, provenance={"fit_subjects": [np.str_("s0"), np.str_("s1")]})
    restored, _ = load_checkpoint(path)
    assert isinstance(restored, MotionBranch)


def test_legacy_numpy_provenance_repair_preserves_weights(tmp_path):
    model = MotionBranch()
    path = tmp_path / "old.pt"
    old = {"schema": "representation-v2-motion-branch-v1", "weights": model.state_dict(),
           "center": [0.0] * 6, "scale": [1.0] * 6, "steps": 2000, "seed": 20260924,
           "provenance": {"fit_subjects": [np.str_("s0")], "identity": "known"}}
    torch.save(old, path)
    with pytest.raises(Exception):
        load_checkpoint(path)
    result = repair_numpy_checkpoint_provenance(path, expected_identity="known")
    assert result["weights_unchanged"]
    restored, _ = load_checkpoint(path)
    assert restored.time_coverage_mode == "token_average"
    assert all(torch.equal(left, right) for left, right in zip(model.state_dict().values(), restored.state_dict().values()))


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
    # Subject-first sampling intentionally does not impose a 4:4 batch split;
    # this smoke check therefore asserts learning without assuming that former
    # class-first batch composition.
    assert np.mean(result["losses"][-10:]) < np.mean(result["losses"][:10])


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
