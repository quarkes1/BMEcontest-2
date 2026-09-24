"""Research partition guards for the 0.70 challenger."""

import pytest

from scripts.research.representation_v2.contracts import Partition, validate_partition
from scripts.research.representation_v2.provenance import (
    assert_no_heldout_ancestors,
    cache_identity,
    validate_source,
)


def test_partition_rejects_subject_overlap():
    with pytest.raises(ValueError, match="overlap"):
        validate_partition(Partition(0, 1, None, ("s1",), ("s1",), ()))


def test_partition_accepts_disjoint_subjects():
    partition = Partition(0, 1, None, ("s1", "s2"), ("s3",), ())
    assert validate_partition(partition) == partition


def test_transitive_subject_leak_is_rejected():
    graph = {
        "base": {"fit_subjects": ["s1"], "parents": []},
        "meta": {"fit_subjects": ["s2"], "parents": ["base"]},
    }
    with pytest.raises(ValueError, match="s1"):
        assert_no_heldout_ancestors("meta", {"s1"}, graph)


def test_unknown_parent_fails_closed():
    graph = {"meta": {"fit_subjects": ["s2"], "parents": ["missing"]}}
    with pytest.raises(ValueError, match="missing"):
        assert_no_heldout_ancestors("meta", {"s1"}, graph)


def test_cycle_fails_closed():
    graph = {"a": {"fit_subjects": [], "parents": ["b"]}, "b": {"fit_subjects": [], "parents": ["a"]}}
    with pytest.raises(ValueError, match="cycle"):
        assert_no_heldout_ancestors("a", {"s1"}, graph)


def test_outer_diagnostic_source_is_forbidden():
    with pytest.raises(ValueError, match="forbidden"):
        validate_source("outputs/challenger_70/e04_verifier_audit/tp_reference.csv")
    with pytest.raises(ValueError, match="forbidden"):
        validate_source("outputs/challenger_70/temporal_rep/cache/candidate_tokens.npz")


def test_legal_source_is_accepted():
    assert validate_source("cache/sessions/example.npz") == "cache/sessions/example.npz"


def test_cache_identity_depends_on_train_subjects_and_source():
    common = {"schema": "rv2-token-v1", "source": "raw1", "fit_subjects": ["s1"]}
    assert cache_identity(common) != cache_identity({**common, "fit_subjects": ["s2"]})
    assert cache_identity(common) != cache_identity({**common, "source": "raw2"})
