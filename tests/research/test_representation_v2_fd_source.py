"""Contracts for the local FD-I source adapter."""

import pickle

import numpy as np
import pytest

from scripts.research.representation_v2.fd_source import FDIAdapter


def _write_fd_i(root, *, participants: int = 2, samples: int = 960) -> None:
    source = root / "FD-I"
    source.mkdir()
    features = []
    labels = []
    for participant in range(participants):
        values = np.arange(samples * 6, dtype=np.float64).reshape(samples, 6) + participant * 100_000
        target = np.zeros(samples, dtype=np.int32)
        target[128:193] = 1  # Token 0 positive; token 1 is ambiguous; token 2 is negative.
        features.append(values)
        labels.append(target)
    for hand in ("L", "R"):
        with (source / f"X_{hand}.pkl").open("wb") as file:
            pickle.dump(features, file, protocol=pickle.HIGHEST_PROTOCOL)
        with (source / f"Y_{hand}.pkl").open("wb") as file:
            pickle.dump(labels, file, protocol=pickle.HIGHEST_PROTOCOL)


def test_adapter_preserves_undocumented_units_and_fd_i_contract(tmp_path):
    _write_fd_i(tmp_path)
    adapter = FDIAdapter(tmp_path, expected_participants=2)

    metadata = adapter.metadata()

    assert metadata["sampling_hz"] == 64
    assert metadata["source_channels"] == ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
    assert metadata["gyro_units"] == "undocumented_raw"
    assert metadata["target_gyro_conversion"] == "not_performed"
    assert metadata["participant_count"] == 2


def test_adapter_yields_only_unambiguous_5_second_tokens_for_requested_participants(tmp_path):
    _write_fd_i(tmp_path)
    adapter = FDIAdapter(tmp_path, expected_participants=2)

    tokens = list(adapter.iter_labeled_tokens((1,)))

    assert len(tokens) == 4  # two hands, one positive and one negative token each
    assert {token.participant_id for token in tokens} == {1}
    assert {token.hand for token in tokens} == {"L", "R"}
    assert {token.label for token in tokens} == {0, 1}
    assert all(token.values.shape == (6, 125) for token in tokens)
    assert all(token.values.dtype == np.float32 for token in tokens)
    assert all(token.start_sample in {0, 640} for token in tokens)


def test_participant_split_is_disjoint_and_keeps_both_hands_together(tmp_path):
    _write_fd_i(tmp_path, participants=10)
    adapter = FDIAdapter(tmp_path, expected_participants=10)

    train, validation = adapter.participant_split(seed=20260924)

    assert len(train) == 8
    assert len(validation) == 2
    assert not set(train) & set(validation)
    assert tuple(sorted(train + validation)) == tuple(range(10))
    with pytest.raises(ValueError, match="disjoint"):
        adapter.validate_partitions((0, 1), (1, 2))


def test_feature_file_size_is_checked_before_unpickling(tmp_path):
    _write_fd_i(tmp_path)
    adapter = FDIAdapter(tmp_path, expected_participants=2, max_feature_file_bytes=1)

    with pytest.raises(ValueError, match="memory limit"):
        list(adapter.iter_labeled_tokens((0,)))
