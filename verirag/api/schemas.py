"""
Pydantic Schemas for the VeriRAG REST API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class IngestDocumentRequest(BaseModel):
    document_id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., min_length=10, description="Raw markdown or text content")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class IngestDocumentResponse(BaseModel):
    status: str = "success"
    document_id: str
    chunks_created: int
    total_tokens: int


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="User search or question prompt")
    top_k: int = Field(default=5, ge=1, le=25)
    enable_reranking: bool = Field(default=True)
    enable_verification: bool = Field(default=True)
    metadata_filter: Optional[Dict[str, Any]] = None


class CitationItem(BaseModel):
    index: int
    chunk_id: str
    document_id: str
    quote: Optional[str]
    confidence: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    raw_answer: str
    faithfulness_score: float
    hallucination_index: float
    is_safe: bool
    citations: List[CitationItem]
    retrieved_chunks: List[Dict[str, Any]]


class EvaluateRequest(BaseModel):
    query: str
    retrieved_contexts: List[str]
    generated_answer: str
    ground_truth: str


class EvaluateResponse(BaseModel):
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevance: float
    composite_score: float


class SystemStatsResponse(BaseModel):
    version: str
    total_documents: int
    total_chunks: int
    embedding_provider: str
    embedding_dimension: int
    vector_store_type: str
