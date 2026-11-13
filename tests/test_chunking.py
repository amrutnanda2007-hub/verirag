"""
Unit Tests for Recursive Token Chunker.
"""

from verirag.core.chunking import RecursiveTokenChunker


def test_chunking_basic():
    chunker = RecursiveTokenChunker(chunk_size=50, chunk_overlap=10)
    text = (
        "Distributed systems require consensus algorithms. "
        "Raft is one of the most popular consensus algorithms because of its understandable design. "
        "It achieves consensus through an elected leader that manages a replicated log. "
        "If a node crashes, the system can continue to operate normally as long as a quorum exists."
    )
    chunks = chunker.chunk_document("doc_01", text)
    assert len(chunks) >= 2
    assert all(c.document_id == "doc_01" for c in chunks)
    assert all(c.chunk_id.startswith("chk_") for c in chunks)
    assert all(len(c.content) > 0 for c in chunks)


def test_chunking_overlap_validation():
    try:
        RecursiveTokenChunker(chunk_size=50, chunk_overlap=60)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_chunk_span_integrity():
    chunker = RecursiveTokenChunker(chunk_size=40, chunk_overlap=5)
    text = "Section 1: Authentication.\n\nSection 2: Authorization.\n\nSection 3: Audit Logging."
    chunks = chunker.chunk_document("doc_span", text)
    for c in chunks:
        assert c.char_start >= 0
        assert c.char_end > c.char_start
        assert c.token_count > 0

