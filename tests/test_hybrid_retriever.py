"""
Unit Tests for Hybrid Retriever & RRF Fusion.
"""

from verirag.core.bm25 import BM25Index
from verirag.core.chunking import DocumentChunk
from verirag.core.embeddings import DeterministicEmbeddingProvider
from verirag.core.hybrid_retriever import HybridRetriever
from verirag.core.reranker import CrossEncoderReranker
from verirag.core.vector_store import VectorStore


def test_hybrid_retrieval_flow():
    vector_store = VectorStore()
    bm25_index = BM25Index()
    embedder = DeterministicEmbeddingProvider(dimension=128)
    reranker = CrossEncoderReranker()

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        embedding_provider=embedder,
        reranker=reranker,
        rrf_k=60,
        alpha=0.5,
    )

    chunks = [
        DocumentChunk(
            chunk_id="c1",
            document_id="doc_raft",
            content="Raft uses randomized election timers between 150ms and 300ms to elect a cluster leader.",
            char_start=0,
            char_end=85,
            token_count=20,
        ),
        DocumentChunk(
            chunk_id="c2",
            document_id="doc_zt",
            content="Zero Trust mandates mutual TLS and ephemeral certificate rotation.",
            char_start=0,
            char_end=67,
            token_count=15,
        ),
    ]

    indexed_count = retriever.index_document(chunks)
    assert indexed_count == 2
    assert len(vector_store) == 2

    # Query for Raft
    results = retriever.retrieve("election timers in Raft", top_k=2)
    assert len(results) > 0
    assert results[0].chunk.document_id == "doc_raft"
    assert results[0].rerank_score is not None
