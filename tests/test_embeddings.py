"""
Unit Tests for Deterministic Embedding Provider.
"""

import math
from verirag.core.embeddings import DeterministicEmbeddingProvider


def test_deterministic_embedding_dimension():
    embedder = DeterministicEmbeddingProvider(dimension=384)
    vec = embedder.embed_text("Distributed consensus and state replication.")
    assert len(vec) == 384
    norm = math.sqrt(sum(x * x for x in vec))
    assert math.isclose(norm, 1.0, rel_tol=1e-4)


def test_embedding_reproducibility():
    embedder = DeterministicEmbeddingProvider(dimension=384)
    v1 = embedder.embed_text("Zero trust security and mutual TLS.")
    v2 = embedder.embed_text("Zero trust security and mutual TLS.")
    assert v1 == v2


def test_semantic_affinity():
    embedder = DeterministicEmbeddingProvider(dimension=384)
    v_base = embedder.embed_text("database transactions and ACID properties")
    v_related = embedder.embed_text("database transactions with rollback guarantees")
    v_unrelated = embedder.embed_text("cooking authentic Italian tomato pasta sauce")

    sim_related = sum(a * b for a, b in zip(v_base, v_related))
    sim_unrelated = sum(a * b for a, b in zip(v_base, v_unrelated))

    assert sim_related > sim_unrelated
