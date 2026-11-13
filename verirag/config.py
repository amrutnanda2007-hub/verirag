"""
Configuration management for VeriRAG.
Handles environment variables, vector store paths, embedding settings, and thresholds.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Optional


@dataclass
class Settings:
    # Environment
    app_env: str = os.getenv("VERIRAG_ENV", "development")
    log_level: str = os.getenv("VERIRAG_LOG_LEVEL", "INFO")
    
    # Storage & Persistence
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("VERIRAG_DATA_DIR", "./data")))
    vector_store_path: Path = field(default_factory=lambda: Path(os.getenv("VERIRAG_VECTOR_PATH", "./data/vector_store.json")))
    bm25_index_path: Path = field(default_factory=lambda: Path(os.getenv("VERIRAG_BM25_PATH", "./data/bm25_index.json")))

    # Chunking
    chunk_size: int = int(os.getenv("VERIRAG_CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("VERIRAG_CHUNK_OVERLAP", "64"))

    # Embeddings & Model Selection
    embedding_provider: str = os.getenv("VERIRAG_EMBEDDING_PROVIDER", "deterministic")  # "deterministic", "sentence-transformers", "openai"
    embedding_model: str = os.getenv("VERIRAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedding_dimension: int = int(os.getenv("VERIRAG_EMBED_DIM", "384"))

    # Retrieval & RRF Weights
    default_top_k: int = int(os.getenv("VERIRAG_TOP_K", "5"))
    rrf_k: int = int(os.getenv("VERIRAG_RRF_K", "60"))
    hybrid_alpha: float = float(os.getenv("VERIRAG_HYBRID_ALPHA", "0.6"))  # 1.0 = pure dense, 0.0 = pure sparse
    enable_reranking: bool = os.getenv("VERIRAG_ENABLE_RERANKING", "true").lower() in ("1", "true", "yes")

    # Hallucination Guardrails
    enable_verification: bool = os.getenv("VERIRAG_ENABLE_VERIFICATION", "true").lower() in ("1", "true", "yes")
    claim_entailment_threshold: float = float(os.getenv("VERIRAG_CLAIM_THRESHOLD", "0.68"))
    max_hallucination_index: float = float(os.getenv("VERIRAG_MAX_HALLUCINATION_INDEX", "0.25"))

    # API Keys
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY", None)
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY", None)

    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
