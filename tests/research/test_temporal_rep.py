import numpy as np
import torch

from scripts.research.temporal_rep.core import SmallTokenTCN, robust_channel_stats, topk_mean


def test_robust_channel_stats_are_channelwise_and_finite():
    x = np.arange(2 * 3 * 6 * 10, dtype=float).reshape(2, 3, 6, 10)
    center, scale = robust_channel_stats(x, np.ones((2, 3), bool))
    assert center.shape == scale.shape == (6,)
    assert np.all(scale > 0)


def test_small_tcn_returns_token_embedding_and_score():
    model = SmallTokenTCN(channels=24, embedding_dim=16)
    score, embedding = model(torch.zeros(4, 6, 50))
    assert score.shape == (4,)
    assert embedding.shape == (4, 16)
    assert sum(p.numel() for p in model.parameters()) < 100_000


def test_topk_mean_ignores_padding():
    scores = torch.tensor([[1.0, 3.0, 2.0, 99.0]])
    mask = torch.tensor([[True, True, True, False]])
    assert torch.allclose(topk_mean(scores, mask, fraction=0.5), torch.tensor([2.5]))
