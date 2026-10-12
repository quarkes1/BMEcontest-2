"""Temporal ownership must not truncate meals, duplicate tokens or bridge gaps."""
import numpy as np
import pytest


def session(count=800):
    from scripts.research.representation_v2.transformer.timeline import SessionTokens
    return SessionTokens('s', 2500+np.arange(count)*5000, np.zeros(count, np.int64),
                         np.ones(count, np.float32), np.ones(count, bool),
                         np.zeros((count, 16), np.float32), 0, count*5000)


def test_responsibility_is_unique_at_boundaries():
    from scripts.research.representation_v2.transformer.timeline import owned_cells
    np.testing.assert_array_equal(owned_cells(0,1800001,np.array([0,899999,900000,1800000])),
                                  [0,0,1,2])
    with pytest.raises(ValueError): owned_cells(0,100,np.array([100]))


def test_long_candidate_keeps_all_tokens_once():
    from scripts.research.representation_v2.transformer.timeline import candidate_chunks
    chunks = candidate_chunks(session(), 600000, 3300000)
    inside = np.concatenate([c.token_indices for c in chunks if c.region == 1])
    np.testing.assert_array_equal(inside, np.arange(120,660))
    assert max(len(c.token_indices) for c in chunks) <= 360
    all_indices = np.concatenate([c.token_indices for c in chunks])
    assert len(all_indices) == len(set(all_indices)) == 780


def test_span_gap_splits_chunk_and_does_not_reorder():
    from scripts.research.representation_v2.transformer.timeline import candidate_chunks
    tokens = session(10); tokens.span_id[5:] = 1
    chunks = candidate_chunks(tokens, 0, 50000)
    assert [c.token_indices.tolist() for c in chunks] == [list(range(5)),list(range(5,10))]
    assert [c.span_id for c in chunks] == [0,1]


def test_extremely_short_candidate_gets_overlapping_token_not_deleted():
    from scripts.research.representation_v2.transformer.timeline import candidate_chunks
    chunks = candidate_chunks(session(3), 100, 200)
    assert [c.token_indices.tolist() for c in chunks if c.region == 1] == [[0]]
    assert 0 not in np.concatenate([c.token_indices for c in chunks if c.region != 1])


def test_session_edges_clip_context_and_preserve_invalid_tokens():
    from scripts.research.representation_v2.transformer.timeline import candidate_chunks
    tokens = session(3); tokens.valid[:] = False
    chunks = candidate_chunks(tokens, 0, 5000)
    assert [c.region for c in chunks] == [1,2]
    np.testing.assert_array_equal(np.concatenate([c.token_indices for c in chunks]), [0,1,2])
    with pytest.raises(ValueError): candidate_chunks(tokens,10,10)


def test_malformed_embeddings_and_backwards_time_are_rejected():
    from scripts.research.representation_v2.transformer.timeline import SessionTokens
    with pytest.raises(ValueError):
        SessionTokens('s', np.array([4,3]),np.zeros(2),np.ones(2),np.ones(2,bool),np.zeros((2,16)))
    with pytest.raises(ValueError):
        SessionTokens('s', np.array([3]),np.zeros(1),np.ones(1),np.ones(1,bool),np.zeros((1,17)))
