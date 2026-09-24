"""Small order-aware motion branch with masked token and event pooling."""

from __future__ import annotations

import torch
from torch import nn

from scripts.research.temporal_rep.core import ResidualBlock


class _TokenEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv1d(6, 24, 5, padding=2)
        self.blocks = nn.ModuleList(ResidualBlock(24, dilation) for dilation in (1, 2, 4))
        self.project = nn.Linear(24, 16)

    def forward(self, values: torch.Tensor, sample_mask: torch.Tensor) -> torch.Tensor:
        sample_weight = sample_mask[:, None, :].to(values.dtype)
        x = torch.relu(self.stem(values * sample_weight)) * sample_weight
        for block in self.blocks:
            x = block(x) * sample_weight
        pooled = x.sum(dim=-1) / sample_weight.sum(dim=-1).clamp_min(1.0)
        return self.project(pooled)


def sample_time_coverage(sample_mask: torch.Tensor, intervals: torch.Tensor, *, bins: int = 64) -> torch.Tensor:
    """Integrate observed 40ms cells on the fixed 5s/25Hz candidate grid."""
    if sample_mask.ndim != 3 or sample_mask.shape[-1] != 125 or bins < 1:
        raise ValueError("coverage requires the registered uniform 25Hz token grid")
    if intervals.shape != (len(sample_mask), 2) or torch.any(intervals[:, 1] <= intervals[:, 0]):
        raise ValueError("invalid coverage intervals")
    cells = sample_mask.reshape(len(sample_mask), -1).to(torch.float64)
    prefix = torch.cat((torch.zeros(len(cells), 1, device=cells.device, dtype=cells.dtype), cells.cumsum(1)), dim=1)
    duration = (intervals[:, 1] - intervals[:, 0]).to(torch.float64)
    edges = duration[:, None] / 40 * torch.linspace(0, 1, bins + 1, device=cells.device, dtype=torch.float64)[None]
    indices = edges.floor().long().clamp(0, cells.shape[1])
    fractions = (edges - indices).clamp(0, 1)
    at_edge = torch.cat((cells, torch.zeros_like(cells[:, :1])), dim=1).gather(1, indices)
    integral = prefix.gather(1, indices) + fractions * at_edge
    return ((integral[:, 1:] - integral[:, :-1]) * 40 / (duration[:, None] / bins)).clamp(0, 1).float()


def aggregate_time_bins(
    embeddings: torch.Tensor,
    coverage: torch.Tensor,
    token_mask: torch.Tensor,
    times: torch.Tensor,
    intervals: torch.Tensor,
    *,
    bins: int = 64,
    time_coverage: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project every token into its real-time bin, preserving ordering and gaps."""
    if embeddings.ndim != 3 or embeddings.shape[-1] != 16 or coverage.shape != embeddings.shape[:2]:
        raise ValueError("token embeddings and coverage must align")
    if token_mask.shape != coverage.shape or times.shape != coverage.shape or intervals.shape != (len(embeddings), 2):
        raise ValueError("token masks, times and intervals must align")
    if bins != 64 or torch.any(intervals[:, 1] <= intervals[:, 0]):
        raise ValueError("invalid frozen event bin geometry")
    batch, token_count, dimension = embeddings.shape
    real = times >= 0  # collate pads timestamps with -1; invalid real tokens retain their time.
    valid = token_mask.bool() & real
    duration = intervals[:, 1] - intervals[:, 0]
    placement = ((times - intervals[:, :1]).clamp_min(0) * bins // duration[:, None]).clamp(0, bins - 1).long()
    one = torch.ones_like(coverage)
    valid_count = torch.zeros(batch, bins, dtype=coverage.dtype, device=coverage.device)
    real_count = torch.zeros_like(valid_count)
    valid_count.scatter_add_(1, placement, valid.to(coverage.dtype))
    real_count.scatter_add_(1, placement, real.to(coverage.dtype))
    coverage_sum = torch.zeros_like(valid_count)
    coverage_sum.scatter_add_(1, placement, torch.where(real, coverage, torch.zeros_like(coverage)))
    indices = placement.unsqueeze(-1).expand(-1, -1, dimension)
    vector_sum = torch.zeros(batch, bins, dimension, dtype=embeddings.dtype, device=embeddings.device)
    vector_sum.scatter_add_(1, indices, torch.where(valid.unsqueeze(-1), embeddings, torch.zeros_like(embeddings)))
    vector_max = torch.full_like(vector_sum, float("-inf"))
    vector_max.scatter_reduce_(1, indices,
                               torch.where(valid.unsqueeze(-1), embeddings, torch.full_like(embeddings, float("-inf"))),
                               reduce="amax", include_self=True)
    has_valid = valid_count > 0
    vector_max = torch.where(has_valid.unsqueeze(-1), vector_max, torch.zeros_like(vector_max))
    centers = ((torch.arange(bins, device=embeddings.device, dtype=embeddings.dtype) + 0.5) / bins)
    if time_coverage is not None and time_coverage.shape != valid_count.shape:
        raise ValueError("true time coverage must match event bins")
    metadata = torch.stack((centers.expand(batch, -1),
                            torch.full_like(valid_count, 1.0 / bins),
                            coverage_sum / real_count.clamp_min(1) if time_coverage is None else time_coverage,
                            valid_count / real_count.clamp_min(1)), dim=-1)
    result = torch.cat((vector_sum / valid_count.clamp_min(1).unsqueeze(-1), vector_max, metadata), dim=-1)
    result = result * has_valid.unsqueeze(-1).to(result.dtype)
    return result, has_valid


class MotionBranch(nn.Module):
    def __init__(self, *, time_coverage_mode: str = "sample_time") -> None:
        super().__init__()
        if time_coverage_mode not in {"sample_time", "token_average"}:
            raise ValueError("unknown time coverage mode")
        self.time_coverage_mode = time_coverage_mode
        self.token_encoder = _TokenEncoder()
        self.event_stem = nn.Conv1d(36, 32, 1)
        self.event_blocks = nn.ModuleList(ResidualBlock(32, dilation) for dilation in (1, 2, 4))
        self.head = nn.Sequential(nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(
        self,
        values: torch.Tensor,
        masks: tuple[torch.Tensor, torch.Tensor],
        times: torch.Tensor,
        intervals: torch.Tensor,
        *,
        chunk_size: int = 512,
    ) -> torch.Tensor:
        if values.ndim != 4 or values.shape[2:] != (6, 125) or chunk_size < 1:
            raise ValueError("expected B x N x 6 x 125 motion values")
        sample_mask, token_mask = masks
        if sample_mask.shape != values.shape[:2] + (125,) or token_mask.shape != values.shape[:2]:
            raise ValueError("sample and token masks do not align")
        trainable = token_mask.bool() & (times >= 0)
        flat_values = values[trainable]
        flat_samples = sample_mask[trainable]
        chunks = [self.token_encoder(flat_values[left:left + chunk_size], flat_samples[left:left + chunk_size])
                  for left in range(0, len(flat_values), chunk_size)]
        token_vectors = torch.zeros(*values.shape[:2], 16, device=values.device, dtype=values.dtype)
        if chunks:
            token_vectors[trainable] = torch.cat(chunks, dim=0)
        coverage = sample_mask.to(values.dtype).mean(dim=-1)
        binned, bin_mask = aggregate_time_bins(token_vectors, coverage, token_mask, times, intervals,
                                              time_coverage=(sample_time_coverage(sample_mask, intervals)
                                                             if self.time_coverage_mode == "sample_time" else None))
        weight = bin_mask[:, None, :].to(values.dtype)
        sequence = torch.relu(self.event_stem(binned.transpose(1, 2))) * weight
        for block in self.event_blocks:
            sequence = block(sequence) * weight
        mean = sequence.sum(dim=-1) / weight.sum(dim=-1).clamp_min(1)
        maximum = sequence.masked_fill(~bin_mask[:, None, :], float("-inf")).max(dim=-1).values
        maximum = torch.where(bin_mask.any(dim=-1, keepdim=True), maximum, torch.zeros_like(maximum))
        return self.head(torch.cat((mean, maximum), dim=-1)).squeeze(-1)
