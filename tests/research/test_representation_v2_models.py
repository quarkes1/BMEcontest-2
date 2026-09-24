"""The event branch must use time order while ignoring padded tokens."""

import torch

from scripts.research.representation_v2.models import MotionBranch, aggregate_time_bins


def _inputs():
    values = torch.randn(2, 3, 6, 125)
    sample = torch.ones(2, 3, 125, dtype=torch.bool)
    token = torch.ones(2, 3, dtype=torch.bool)
    times = torch.tensor([[2_500, 7_500, 12_500], [2_500, 7_500, 12_500]], dtype=torch.int64)
    intervals = torch.tensor([[0, 15_000], [0, 15_000]], dtype=torch.int64)
    return values, (sample, token), times, intervals


def test_output_shape_parameter_budget_and_finite_empty_bag():
    model = MotionBranch()
    values, masks, times, intervals = _inputs()
    output = model(values, masks, times, intervals)
    assert output.shape == (2,)
    assert torch.isfinite(output).all()
    assert sum(p.numel() for p in model.parameters()) < 100_000
    output_empty = model(values, (torch.zeros_like(masks[0]), torch.zeros_like(masks[1])), times, intervals)
    assert torch.isfinite(output_empty).all()


def test_padded_token_does_not_change_prediction():
    torch.manual_seed(3)
    model = MotionBranch().eval()
    values, masks, times, intervals = _inputs()
    with torch.no_grad():
        first = model(values, masks, times, intervals)
        padded_values = torch.cat((values, torch.randn(2, 1, 6, 125) * 100), dim=1)
        padded_sample = torch.cat((masks[0], torch.zeros(2, 1, 125, dtype=torch.bool)), dim=1)
        padded_token = torch.cat((masks[1], torch.zeros(2, 1, dtype=torch.bool)), dim=1)
        padded_times = torch.cat((times, torch.full((2, 1), -1, dtype=torch.int64)), dim=1)
        second = model(padded_values, (padded_sample, padded_token), padded_times, intervals)
    assert torch.allclose(first, second, atol=1e-6)


def test_ordered_bins_preserve_distinct_motion_positions():
    embedding = torch.stack((torch.ones(16), torch.ones(16) * 2))[None]
    coverage = torch.ones(1, 2)
    mask = torch.ones(1, 2, dtype=torch.bool)
    times = torch.tensor([[2_500, 7_500]], dtype=torch.int64)
    interval = torch.tensor([[0, 10_000]], dtype=torch.int64)
    bins_a, valid_a = aggregate_time_bins(embedding, coverage, mask, times, interval)
    bins_b, valid_b = aggregate_time_bins(embedding.flip(1), coverage, mask, times, interval)
    assert valid_a.equal(valid_b)
    assert not torch.allclose(bins_a, bins_b)
    assert torch.allclose(embedding.mean(1), embedding.flip(1).mean(1))


def test_chunked_and_unchunked_local_encoder_agree():
    torch.manual_seed(5)
    model = MotionBranch().eval()
    values, masks, times, intervals = _inputs()
    with torch.no_grad():
        small = model(values, masks, times, intervals, chunk_size=2)
        large = model(values, masks, times, intervals, chunk_size=100)
    assert torch.allclose(small, large, atol=1e-6)
