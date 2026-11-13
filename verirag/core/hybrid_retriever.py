"""
Hybrid Retriever Engine.
Unifies BM25 lexical token-matching and dense semantic vector retrieval via Reciprocal Rank Fusion (RRF)
and two-stage cross-encoder reranking.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from verirag.core.bm25 import BM25Index
from verirag.core.chunking import DocumentChunk
from verirag.core.embeddings import BaseEmbeddingProvider
from verirag.core.reranker import CrossEncoderReranker
from verirag.core.vector_store import VectorStore


@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    final_score: float
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rerank_score: Optional[float] = None
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "document_id": self.chunk.document_id,
            "content": self.chunk.content,
            "final_score": round(self.final_score, 4),
            "dense_score": round(self.dense_score, 4) if self.dense_score is not None else None,
            "sparse_score": round(self.sparse_score, 4) if self.sparse_score is not None else None,
            "rerank_score": round(self.rerank_score, 4) if self.rerank_score is not None else None,
            "rank": self.rank,
            "metadata": self.chunk.metadata,
        }


class HybridRetriever:
    """
    Combines sparse (BM25) and dense (vector) retrieval channels using Reciprocal Rank Fusion (RRF),
    followed by an optional cross-encoder reranking stage.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        embedding_provider: BaseEmbeddingProvider,
        reranker: Optional[CrossEncoderReranker] = None,
        rrf_k: int = 60,
        alpha: float = 0.5,
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedding_provider = embedding_provider
        self.reranker = reranker or CrossEncoderReranker()
        self.rrf_k = rrf_k
        self.alpha = alpha

    def index_document(self, chunks: List[DocumentChunk]) -> int:
        """
        Indexes a list of DocumentChunks into both dense vector store and sparse BM25 inverted index.
        """
        if not chunks:
            return 0

        # 1. Sparse Indexing
        bm25_docs = [(c.chunk_id, c.content) for c in chunks]
        self.bm25_index.add_documents(bm25_docs)

        # 2. Dense Indexing
        texts = [c.content for c in chunks]
        vectors = self.embedding_provider.embed_batch(texts)
        self.vector_store.add_batch(chunks, vectors)

        return len(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool_size: int = 20,
        enable_reranking: bool = True,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Executes hybrid retrieval:
        1. Dense retrieval (vector cosine search)
        2. Sparse retrieval (BM25 lexical scoring)
        3. Reciprocal Rank Fusion (RRF)
        4. Cross-Encoder reranking
        """
        # Step 1: Dense Retrieval
        q_vec = self.embedding_provider.embed_text(query)
        dense_hits = self.vector_store.search(
            q_vec, top_k=candidate_pool_size, metadata_filter=metadata_filter
        )
        dense_rank_map = {chunk_id: (rank + 1, score) for rank, (chunk_id, score) in enumerate(dense_hits)}

        # Step 2: Sparse Retrieval
        sparse_hits = self.bm25_index.search(query, top_k=candidate_pool_size)
        sparse_rank_map = {chunk_id: (rank + 1, score) for rank, (chunk_id, score) in enumerate(sparse_hits)}

        # Step 3: Reciprocal Rank Fusion (RRF)
        all_chunk_ids = set(dense_rank_map.keys()) | set(sparse_rank_map.keys())
        rrf_scores: Dict[str, float] = defaultdict(float)

        for chunk_id in all_chunk_ids:
            score = 0.0
            if chunk_id in dense_rank_map:
                r_dense, _ = dense_rank_map[chunk_id]
                score += self.alpha * (1.0 / (self.rrf_k + r_dense))
            if chunk_id in sparse_rank_map:
                r_sparse, _ = sparse_rank_map[chunk_id]
                score += (1.0 - self.alpha) * (1.0 / (self.rrf_k + r_sparse))
            rrf_scores[chunk_id] = score

        # Sort candidate chunks by RRF score
        fused_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        candidate_chunks: List[DocumentChunk] = []
        for chunk_id, _ in fused_candidates[:candidate_pool_size]:
            chunk = self.vector_store.get_chunk(chunk_id)
            if chunk:
                candidate_chunks.append(chunk)

        if not candidate_chunks:
            return []

        # Step 4: Cross-Encoder Reranking
        if enable_reranking and self.reranker:
            reranked = self.reranker.rerank(query, candidate_chunks, top_k=top_k)
            results: List[RetrievalResult] = []
            for rank_idx, (chunk, r_score) in enumerate(reranked):
                d_score = dense_rank_map.get(chunk.chunk_id, (None, None))[1]
                s_score = sparse_rank_map.get(chunk.chunk_id, (None, None))[1]
                results.append(
                    RetrievalResult(
                        chunk=chunk,
                        final_score=r_score,
                        dense_score=d_score,
                        sparse_score=s_score,
                        rerank_score=r_score,
                        rank=rank_idx + 1,
                    )
                )
            return results
        else:
            # Return top-k directly from RRF
            results = []
            for rank_idx, (chunk_id, score) in enumerate(fused_candidates[:top_k]):
                chunk = self.vector_store.get_chunk(chunk_id)
                if not chunk:
                    continue
                d_score = dense_rank_map.get(chunk_id, (None, None))[1]
                s_score = sparse_rank_map.get(chunk_id, (None, None))[1]
                results.append(
                    RetrievalResult(
                        chunk=chunk,
                        final_score=score,
                        dense_score=d_score,
                        sparse_score=s_score,
                        rerank_score=None,
                        rank=rank_idx + 1,
                    )
                )
            return results
