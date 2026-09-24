"""Memory-bounded, participant-disjoint access to the local official FD-I pickles.

FD-I documents its 64 Hz layout and pointwise labels, but not physical sensor
units.  This adapter therefore preserves the stored values and never guesses a
source-to-target ACC or GYRO conversion.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
from scipy.signal import resample_poly


_HANDS = ("L", "R")
_CHANNELS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
_SOURCE_HZ = 64
_TARGET_HZ = 25
_TOKEN_SECONDS = 5
_SOURCE_TOKEN_SAMPLES = _SOURCE_HZ * _TOKEN_SECONDS
_TARGET_TOKEN_SAMPLES = _TARGET_HZ * _TOKEN_SECONDS
_CENTER_HALF_WIDTH_SAMPLES = _SOURCE_HZ // 2
_NEGATIVE_EXCLUSION_SAMPLES = (_SOURCE_HZ * 5) // 2


@dataclass(frozen=True)
class FDToken:
    participant_id: int
    hand: str
    start_sample: int
    label: int
    values: np.ndarray


class FDIAdapter:
    """Yield FD-I source tokens without loading both wrists at the same time.

    A pickle necessarily materializes its list while it is being decoded.  The
    adapter limits that unavoidable peak to one wrist's X/Y pair, validates the
    pair before yielding, and discards it before opening the other wrist.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        expected_participants: int = 34,
        max_feature_file_bytes: int = 4 * 1024**3,
        max_label_file_bytes: int = 512 * 1024**2,
    ) -> None:
        self.source = Path(root) / "FD-I"
        if expected_participants < 1 or max_feature_file_bytes < 1 or max_label_file_bytes < 1:
            raise ValueError("FD-I limits and participant count must be positive")
        self.expected_participants = expected_participants
        self.max_feature_file_bytes = max_feature_file_bytes
        self.max_label_file_bytes = max_label_file_bytes

    def metadata(self) -> dict[str, object]:
        """Return only documented source facts and explicit unit uncertainty."""
        return {
            "participant_count": self.expected_participants,
            "sampling_hz": _SOURCE_HZ,
            "token_seconds": _TOKEN_SECONDS,
            "target_sampling_hz": _TARGET_HZ,
            "source_channels": _CHANNELS,
            "acc_units": "undocumented_raw",
            "gyro_units": "undocumented_raw",
            "target_acc_conversion": "not_performed",
            "target_gyro_conversion": "not_performed",
            "label_0": "others",
            "label_1": "eating_gesture",
            "label_2": "drinking_gesture_excluded_from_negative_labels",
        }

    def participant_split(self, *, seed: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Make the registered 80/20 participant split before either hand is read."""
        if not isinstance(seed, (int, np.integer)):
            raise ValueError("split seed must be an integer")
        shuffled = np.random.default_rng(int(seed)).permutation(self.expected_participants)
        cut = int(self.expected_participants * 0.8)
        train = tuple(sorted(map(int, shuffled[:cut])))
        validation = tuple(sorted(map(int, shuffled[cut:])))
        self.validate_partitions(train, validation)
        return train, validation

    def validate_partitions(self, train_participants: Sequence[int], validation_participants: Sequence[int]) -> None:
        train = self._validate_participants(train_participants)
        validation = self._validate_participants(validation_participants)
        if not train or not validation:
            raise ValueError("FD-I train and validation participant groups must both be nonempty")
        if set(train) & set(validation):
            raise ValueError("FD-I participant partitions must be disjoint")

    def iter_labeled_tokens(self, participants: Sequence[int]) -> Iterator[FDToken]:
        """Yield 5 s labelled tokens for one participant group, one wrist at a time.

        Eating label ``1`` in the central ±0.5 s gives a positive token.  A
        negative token is separated from every non-other label (including
        drinking) by the registered ±2.5 s exclusion margin.  Remaining tokens
        are ambiguous and intentionally omitted.
        """
        selected = self._validate_participants(participants)
        for hand in _HANDS:
            features, labels = self._load_hand(hand)
            try:
                for participant_id in selected:
                    values = features[participant_id]
                    target = labels[participant_id]
                    self._validate_participant_arrays(values, target, participant_id, hand)
                    yield from self._tokens_for_participant(values, target, participant_id, hand)
            finally:
                # Keep the live decode bounded to one X/Y hand pair.
                del features
                del labels

    def _validate_participants(self, participants: Sequence[int]) -> tuple[int, ...]:
        values = tuple(int(participant) for participant in participants)
        if len(values) != len(set(values)):
            raise ValueError("FD-I participant ids must be unique")
        if any(participant < 0 or participant >= self.expected_participants for participant in values):
            raise ValueError("FD-I participant id is outside the configured source")
        return values

    def _load_hand(self, hand: str) -> tuple[list[object], list[object]]:
        if hand not in _HANDS:
            raise ValueError("unknown FD-I hand")
        feature_path = self.source / f"X_{hand}.pkl"
        label_path = self.source / f"Y_{hand}.pkl"
        self._check_file(feature_path, self.max_feature_file_bytes, "feature")
        self._check_file(label_path, self.max_label_file_bytes, "label")
        with feature_path.open("rb") as file:
            features = pickle.load(file)
        with label_path.open("rb") as file:
            labels = pickle.load(file)
        if not isinstance(features, list) or not isinstance(labels, list):
            raise ValueError("official FD-I pickle must contain participant lists")
        if len(features) != self.expected_participants or len(labels) != self.expected_participants:
            raise ValueError("FD-I participant count differs from the registered source")
        return features, labels

    @staticmethod
    def _check_file(path: Path, maximum: int, kind: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"FD-I {kind} pickle is missing: {path}")
        if path.stat().st_size > maximum:
            raise ValueError(f"FD-I {kind} pickle exceeds configured memory limit")

    @staticmethod
    def _validate_participant_arrays(values: object, labels: object, participant_id: int, hand: str) -> None:
        x = np.asarray(values)
        y = np.asarray(labels)
        if x.ndim != 2 or x.shape[1] != 6 or y.ndim != 1 or len(x) != len(y):
            raise ValueError(f"FD-I participant {participant_id} hand {hand} has invalid X/Y shapes")
        if not np.isfinite(x).all() or not np.isin(y, (0, 1, 2)).all():
            raise ValueError(f"FD-I participant {participant_id} hand {hand} has invalid values or labels")

    @staticmethod
    def _tokens_for_participant(values: object, labels: object, participant_id: int, hand: str) -> Iterator[FDToken]:
        x = np.asarray(values)
        y = np.asarray(labels)
        for start in range(0, len(y) - _SOURCE_TOKEN_SAMPLES + 1, _SOURCE_TOKEN_SAMPLES):
            end = start + _SOURCE_TOKEN_SAMPLES
            center = start + _SOURCE_TOKEN_SAMPLES // 2
            positive = bool(np.any(y[center - _CENTER_HALF_WIDTH_SAMPLES:center + _CENTER_HALF_WIDTH_SAMPLES + 1] == 1))
            exclusion_start = max(0, start - _NEGATIVE_EXCLUSION_SAMPLES)
            exclusion_end = min(len(y), end + _NEGATIVE_EXCLUSION_SAMPLES)
            isolated_negative = not np.any(y[exclusion_start:exclusion_end] != 0)
            if not positive and not isolated_negative:
                continue
            downsampled = resample_poly(x[start:end], _TARGET_HZ, _SOURCE_HZ, axis=0)
            if downsampled.shape != (_TARGET_TOKEN_SAMPLES, 6):
                raise RuntimeError("FD-I resampling violated the fixed 5 s input contract")
            yield FDToken(participant_id, hand, start, int(positive), downsampled.T.astype(np.float32, copy=False))
