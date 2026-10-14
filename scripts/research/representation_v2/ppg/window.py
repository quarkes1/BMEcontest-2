"""One streaming raw read; conservative spectral proxies, never clinical HR."""
from collections import Counter
import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch

from .packet import iter_packets

COLUMNS = ['PPG_TIME', 'PPG3', 'PPG5', 'ACC_TIME', 'ACC_X', 'ACC_Y', 'ACC_Z']
SOS = butter(3, [.7, 3.3], btype='bandpass', fs=25, output='sos')


class HashReader(io.RawIOBase):
    def __init__(self, handle):
        self.handle, self.digest, self.bytes = handle, hashlib.sha256(), 0

    def readable(self):
        return True

    def readinto(self, buffer):
        n = self.handle.readinto(buffer)
        if n:
            self.digest.update(memoryview(buffer)[:n]); self.bytes += n
        return n


def spectral_stats(values):
    x = np.asarray(values, dtype=float)
    out = dict(valid=False, pair_pass=False, frequency=float('nan'),
               rms3_dc=float('nan'), rms5_dc=float('nan'),
               peak3=float('nan'), peak5=float('nan'),
               concentration3=float('nan'), concentration5=float('nan'))
    if (x.shape != (750, 2) or not np.isfinite(x).all()
            or np.any(np.ptp(x, axis=0) == 0)):
        return out
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        filtered = sosfiltfilt(SOS, x-x.mean(axis=0), axis=0)
        freq, power = welch(filtered, fs=25, nperseg=512, axis=0)
        band = (freq >= .7) & (freq <= 3.)
        if not np.isfinite(power).all() or np.any(power[band].sum(axis=0) <= 0):
            return out
        peak = freq[band][np.argmax(power[band], axis=0)]
        concentration = np.array([power[band & (np.abs(freq-peak[j]) <= .12), j].sum()
                                  / power[band, j].sum() for j in range(2)])
        dc = np.abs(np.median(x, axis=0))
        ratios = np.sqrt(np.mean(filtered**2, axis=0))/dc
        ratios[~np.isfinite(ratios)] = np.nan
    out.update(valid=True, pair_pass=bool(np.all(concentration >= .15) and abs(peak[0]-peak[1]) <= 1/6),
               frequency=float(peak.mean()), rms3_dc=float(ratios[0]), rms5_dc=float(ratios[1]),
               peak3=float(peak[0]), peak5=float(peak[1]),
               concentration3=float(concentration[0]), concentration5=float(concentration[1]))
    return out


def motion_std(times, norms, start, end):
    mask = (times >= start) & (times < end) & np.isfinite(norms)
    return float(np.std(norms[mask])) if mask.sum() >= 100 else float('nan')


def extract_file(path, sid, chunksize=100000):
    path = Path(path)
    counts, packetsizes = Counter(), Counter()
    acc_t, acc_n, windows, pending = [], [], [], []
    with path.open('rb') as handle:
        hashing = HashReader(handle)
        with io.BufferedReader(hashing) as stream:
            header = stream.peek(8192).split(b'\n', 1)[0]
            delimiter = '\t' if b'\t' in header else ','
            reader = pd.read_csv(stream, sep=delimiter, usecols=COLUMNS, low_memory=False, chunksize=chunksize)

            def chunks():
                for frame in reader:
                    counts['raw_rows'] += len(frame)
                    if any(frame[c].dtype.kind not in 'fiub' for c in COLUMNS):
                        repeated_header = np.logical_and.reduce([frame[c].eq(c).to_numpy() for c in COLUMNS])
                        counts['repeated_headers'] += int(repeated_header.sum())
                        for c in COLUMNS:
                            original = frame[c]
                            numeric = pd.to_numeric(original, errors='coerce')
                            counts['nonnumeric_cells'] += int((numeric.isna() & original.notna()).sum())
                            frame[c] = numeric
                    a = frame[COLUMNS].to_numpy()
                    good = ((a[:, 3] > 0) & np.isfinite(a[:, 3:]).all(axis=1)
                            & np.any(a[:, 4:] != 0, axis=1))
                    with np.errstate(over='ignore', invalid='ignore'):
                        norms = np.linalg.norm(a[good, 4:], axis=1)
                    acc_t.append(a[good, 3]); acc_n.append(norms)
                    counts['absent_ppg_rows'] += int(np.all(a[:, :3] == 0, axis=1).sum())
                    yield a[:, :3]

            for packet in iter_packets(chunks()):
                counts['packets'] += 1
                counts['ppg_rows'] += len(packet['values'])
                counts['zero_values'] += packet['zero_values']
                packetsizes[str(len(packet['values']))] += 1
                if packet['break_before']:
                    counts['tail_packets'] += len(pending)
                    pending = []
                if not packet['valid']:
                    counts['invalid_packets'] += 1
                    for reason in packet['reason'].split(','):
                        counts[reason] += 1
                    continue
                counts['complete_valid_packets'] += 1
                pending.append(packet)
                if len(pending) == 50:
                    ts = np.array([p['time_ms'] for p in pending])
                    stats = spectral_stats(np.concatenate([p['values'] for p in pending]))
                    stats.update(sid=sid, start_ms=int(ts[0]), end_ms=int(ts[-1]+600),
                                 implied_hz=float(15000/np.median(np.diff(ts))),
                                 packet_span_ms=float(ts[-1]-ts[0]))
                    windows.append(stats); pending = []
            counts['tail_packets'] += len(pending)
            # pandas consumed EOF; digest covers every byte of the single read.
            sha, nbytes = hashing.digest.hexdigest(), hashing.bytes
    if nbytes != path.stat().st_size:
        raise ValueError('raw file changed or not fully consumed')
    times = np.concatenate(acc_t) if acc_t else np.empty(0)
    norms = np.concatenate(acc_n) if acc_n else np.empty(0)
    order = np.argsort(times, kind='stable')
    times, norms = times[order], norms[order]
    for w in windows:
        lo, hi = np.searchsorted(times, [w['start_ms'], w['end_ms']])
        w['motion'] = motion_std(times[lo:hi], norms[lo:hi], w['start_ms'], w['end_ms'])
    return dict(sid=sid, sha256=sha, bytes=nbytes, counts=dict(counts),
                packet_sizes=dict(packetsizes), windows=windows)
