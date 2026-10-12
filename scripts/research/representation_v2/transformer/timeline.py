"""Physical-time ownership, span-isolated chunks and full-length candidates."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SessionTokens:
    sid: str
    center_ms: np.ndarray
    span_id: np.ndarray
    coverage: np.ndarray
    valid: np.ndarray
    embeddings: np.ndarray
    start_ms: int | None = None
    end_ms: int | None = None

    def __post_init__(self):
        n = len(self.center_ms)
        if (not self.sid or np.asarray(self.center_ms).ndim != 1
                or any(np.asarray(v).shape != (n,) for v in (self.span_id, self.coverage, self.valid))
                or np.asarray(self.embeddings).shape != (n,16)
                or not np.isfinite(self.embeddings).all()
                or not np.isfinite(self.coverage).all()
                or np.any((self.coverage < 0) | (self.coverage > 1))
                or (n > 1 and np.any(np.diff(self.center_ms) <= 0))):
            raise ValueError('invalid session token timeline')
        if n:
            if self.start_ms is None: object.__setattr__(self, 'start_ms', int(self.center_ms[0]-2500))
            if self.end_ms is None: object.__setattr__(self, 'end_ms', int(self.center_ms[-1]+2500))
        if self.start_ms is None or self.end_ms is None or self.end_ms <= self.start_ms:
            raise ValueError('invalid session bounds')


@dataclass(frozen=True)
class RegionChunk:
    region: int
    token_indices: np.ndarray
    span_id: int


def owned_cells(start_ms, end_ms, centers):
    centers = np.asarray(centers, dtype=np.int64)
    if end_ms <= start_ms or np.any((centers < start_ms) | (centers >= end_ms)):
        raise ValueError('centers outside owned span')
    return (centers-start_ms)//900000


def candidate_chunks(tokens, start_ms, end_ms):
    if end_ms <= start_ms or start_ms < tokens.start_ms or end_ms > tokens.end_ms+200:
        raise ValueError('candidate outside session or invalid duration')
    times = tokens.center_ms
    region = np.full(len(times), -1, dtype=np.int8)
    for k, (left,right) in enumerate(((start_ms-600000,start_ms),
                                     (start_ms,end_ms),(end_ms,end_ms+600000))):
        region[(times >= max(left,tokens.start_ms)) & (times < min(right,tokens.end_ms))] = k
    if not np.any(region == 1):
        overlap = np.maximum(0, np.minimum(times+2500,end_ms)-np.maximum(times-2500,start_ms))
        possible = np.flatnonzero((overlap > 0) & tokens.valid)
        if len(possible): region[possible[np.argmax(overlap[possible])]] = 1
    chunks = []
    for k in range(3):
        idx = np.flatnonzero(region == k)
        if not len(idx): continue
        # Never turn a real gap, skipped token, or span transition into adjacency.
        cuts = np.flatnonzero((np.diff(idx) != 1)
                             | (np.diff(times[idx]) != 5000)
                             | (np.diff(tokens.span_id[idx]) != 0))+1
        for block in np.split(idx,cuts):
            for left in range(0,len(block),360):
                keep = block[left:left+360]
                chunks.append(RegionChunk(k,keep,int(tokens.span_id[keep[0]])))
    return chunks
