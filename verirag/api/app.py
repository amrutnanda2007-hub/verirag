"""
FastAPI Server for VeriRAG.
Provides high-performance RESTful interfaces for document ingestion, hybrid query execution,
and automated hallucination verification.
"""

from contextlib import asynccontextmanager
import logging
from typing import Dict

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from verirag import __version__
from verirag.api.schemas import (
    CitationItem,
    EvaluateRequest,
    EvaluateResponse,
    IngestDocumentRequest,
    IngestDocumentResponse,
    QueryRequest,
    QueryResponse,
    SystemStatsResponse,
)
from verirag.config import settings
from verirag.core.bm25 import BM25Index
from verirag.core.chunking import RecursiveTokenChunker
from verirag.core.embeddings import get_embedding_provider
from verirag.core.evaluator import RAGEvaluator
from verirag.core.generator import RAGGenerator
from verirag.core.hallucination_guard import HallucinationGuard
from verirag.core.hybrid_retriever import HybridRetriever
from verirag.core.reranker import CrossEncoderReranker
from verirag.core.vector_store import VectorStore

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("verirag.api")

# Global singleton engine states
state: Dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing VeriRAG core components...")
    chunker = RecursiveTokenChunker(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    embedder = get_embedding_provider(
        provider_name=settings.embedding_provider,
        model_name=settings.embedding_model,
        api_key=settings.openai_api_key,
        dimension=settings.embedding_dimension,
    )
    vector_store = VectorStore.load_from_file(settings.vector_store_path)
    if settings.bm25_index_path.exists():
        bm25_index = BM25Index.load_from_file(settings.bm25_index_path)
    else:
        bm25_index = BM25Index()

    reranker = CrossEncoderReranker() if settings.enable_reranking else None
    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        embedding_provider=embedder,
        reranker=reranker,
        rrf_k=settings.rrf_k,
        alpha=settings.hybrid_alpha,
    )
    guard = HallucinationGuard(
        entailment_threshold=settings.claim_entailment_threshold,
        max_hallucination_index=settings.max_hallucination_index,
    )
    generator = RAGGenerator(
        hallucination_guard=guard,
        openai_api_key=settings.openai_api_key,
    )
    evaluator = RAGEvaluator(embedding_provider=embedder)

    state["chunker"] = chunker
    state["embedder"] = embedder
    state["vector_store"] = vector_store
    state["bm25_index"] = bm25_index
    state["retriever"] = retriever
    state["guard"] = guard
    state["generator"] = generator
    state["evaluator"] = evaluator
    state["indexed_docs"] = set()

    for rec in vector_store.records.values():
        state["indexed_docs"].add(rec.document_id)

    logger.info(
        f"VeriRAG initialized. Chunks: {len(vector_store)}, Documents: {len(state['indexed_docs'])}"
    )
    yield

    logger.info("Persisting indexes on shutdown...")
    vector_store.save_to_file(settings.vector_store_path)
    bm25_index.save_to_file(settings.bm25_index_path)


app = FastAPI(
    title="VeriRAG API",
    description="Self-Correcting Multi-Modal RAG with Hallucination Guardrails & Citation Verification",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "healthy", "service": "verirag", "version": __version__}


@app.get("/api/v1/stats", response_model=SystemStatsResponse)
def get_stats():
    return SystemStatsResponse(
        version=__version__,
        total_documents=len(state["indexed_docs"]),
        total_chunks=len(state["vector_store"]),
        embedding_provider=settings.embedding_provider,
        embedding_dimension=settings.embedding_dimension,
        vector_store_type="In-Memory JSON / Cosine Similarity",
    )


@app.post("/api/v1/documents/ingest", response_model=IngestDocumentResponse)
def ingest_document(req: IngestDocumentRequest):
    chunker: RecursiveTokenChunker = state["chunker"]
    retriever: HybridRetriever = state["retriever"]

    chunks = chunker.chunk_document(
        document_id=req.document_id, content=req.content, metadata=req.metadata
    )
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No chunks extracted from content"
        )

    retriever.index_document(chunks)
    state["indexed_docs"].add(req.document_id)

    # Auto-save changes
    state["vector_store"].save_to_file(settings.vector_store_path)
    state["bm25_index"].save_to_file(settings.bm25_index_path)

    total_tokens = sum(c.token_count for c in chunks)
    return IngestDocumentResponse(
        document_id=req.document_id,
        chunks_created=len(chunks),
        total_tokens=total_tokens,
    )


@app.post("/api/v1/query", response_model=QueryResponse)
def execute_query(req: QueryRequest):
    retriever: HybridRetriever = state["retriever"]
    generator: RAGGenerator = state["generator"]

    retrieved = retriever.retrieve(
        query=req.query,
        top_k=req.top_k,
        enable_reranking=req.enable_reranking,
        metadata_filter=req.metadata_filter,
    )

    if not retrieved:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No relevant context chunks found. Ingest documents first.",
        )

    result = generator.generate(query=req.query, retrieved_contexts=retrieved)

    citations: List[CitationItem] = []
    for claim in result.guardrail_report.claims:
        if claim.citation_index and claim.supporting_chunk_id:
            chunk = state["vector_store"].get_chunk(claim.supporting_chunk_id)
            doc_id = chunk.document_id if chunk else "unknown"
            citations.append(
                CitationItem(
                    index=claim.citation_index,
                    chunk_id=claim.supporting_chunk_id,
                    document_id=doc_id,
                    quote=claim.supporting_quote,
                    confidence=claim.confidence_score,
                )
            )

    return QueryResponse(
        query=req.query,
        answer=result.verified_answer,
        raw_answer=result.raw_answer,
        faithfulness_score=result.guardrail_report.faithfulness_score,
        hallucination_index=result.guardrail_report.hallucination_index,
        is_safe=result.guardrail_report.is_safe,
        citations=citations,
        retrieved_chunks=[c.to_dict() for c in retrieved],
    )


@app.post("/api/v1/evaluate", response_model=EvaluateResponse)
def evaluate_triad(req: EvaluateRequest):
    evaluator: RAGEvaluator = state["evaluator"]
    res = evaluator.evaluate_sample(
        sample_id="custom_eval",
        query=req.query,
        retrieved_contexts=req.retrieved_contexts,
        generated_answer=req.generated_answer,
        ground_truth_answer=req.ground_truth,
    )
    return EvaluateResponse(
        context_precision=res.context_precision,
        context_recall=res.context_recall,
        faithfulness=res.faithfulness,
        answer_relevance=res.answer_relevance,
        composite_score=res.average_score,
    )
