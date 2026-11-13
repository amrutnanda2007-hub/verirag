"""
VeriRAG Core Engine Components.
"""

from verirag.core.chunking import DocumentChunk, RecursiveTokenChunker
from verirag.core.bm25 import BM25Index
from verirag.core.embeddings import BaseEmbeddingProvider, get_embedding_provider
from verirag.core.vector_store import VectorStore
from verirag.core.hybrid_retriever import HybridRetriever
from verirag.core.reranker import CrossEncoderReranker
from verirag.core.hallucination_guard import HallucinationGuard, ClaimVerification
from verirag.core.generator import RAGGenerator
from verirag.core.evaluator import RAGEvaluator

__all__ = [
    "DocumentChunk",
    "RecursiveTokenChunker",
    "BM25Index",
    "BaseEmbeddingProvider",
    "get_embedding_provider",
    "VectorStore",
    "HybridRetriever",
    "CrossEncoderReranker",
    "HallucinationGuard",
    "ClaimVerification",
    "RAGGenerator",
    "RAGEvaluator",
]
