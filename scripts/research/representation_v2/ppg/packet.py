"""Contiguous timestamp packets; preserve quantized and malformed observations."""
import numpy as np


def iter_packets(chunks):
    pending = None
    seen = set()
    previous = None
    previous_valid = False

    def emit(block):
        nonlocal previous, previous_valid
        timestamp = float(block[0, 0])
        x = block[:, 1:3].copy()
        time_valid = np.isfinite(timestamp) and timestamp > 0
        reasons = []
        if not time_valid:
            reasons.append('invalid_time')
        if len(block) != 15:
            reasons.append('packet_size')
        if not np.isfinite(x).all():
            reasons.append('nonfinite')
        if np.any(np.all(x == 0, axis=0)):
            reasons.append('zero_channel')
        repeated = timestamp in seen if time_valid else False
        if repeated:
            reasons.append('repeated_timestamp')
        valid = not reasons
        broken = (not previous_valid or not valid or previous is None
                  or not 450 <= timestamp-previous <= 750)
        if time_valid:
            seen.add(timestamp)
        previous, previous_valid = timestamp, valid
        return dict(time_ms=timestamp, values=x, valid=valid, reason=','.join(reasons),
                    break_before=broken, zero_values=int((x == 0).sum()))

    for chunk in chunks:
        a = np.asarray(chunk, dtype=float)
        if a.ndim != 2 or a.shape[1] != 3:
            raise ValueError('packet rows require time/ch3/ch5')
        # All-zero triples denote absent PPG in asynchronous sensor rows.
        a = a[~np.all(a == 0, axis=1)]
        if pending is not None:
            a = np.vstack([pending, a])
        if not len(a):
            pending = None
            continue
        boundaries = np.r_[0, np.flatnonzero(a[1:, 0] != a[:-1, 0])+1, len(a)]
        for start, end in zip(boundaries[:-2], boundaries[1:-1]):
            yield emit(a[start:end])
        pending = a[boundaries[-2]:].copy()
    if pending is not None:
        yield emit(pending)
