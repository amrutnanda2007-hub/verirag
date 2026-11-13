"""
Vector Store and Indexing Module.
Provides dense vector storage, cosine similarity computation, metadata filtering, and disk persistence.
"""

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from verirag.core.chunking import DocumentChunk


@dataclass
class VectorRecord:
    chunk_id: str
    document_id: str
    content: str
    vector: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "vector": self.vector,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorRecord":
        return cls(
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            content=data["content"],
            vector=data["vector"],
            metadata=data.get("metadata", {}),
        )


class VectorStore:
    """
    Lightweight, persistent vector database supporting exact cosine-similarity kNN search
    and arbitrary metadata predicate filtering.
    """

    def __init__(self):
        self.records: Dict[str, VectorRecord] = {}

    def __len__(self) -> int:
        return len(self.records)

    def add(self, chunk: DocumentChunk, vector: List[float]) -> None:
        """Upsert a single chunk and its embedding vector."""
        self.records[chunk.chunk_id] = VectorRecord(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            content=chunk.content,
            vector=vector,
            metadata=chunk.metadata,
        )

    def add_batch(self, chunks: List[DocumentChunk], vectors: List[List[float]]) -> None:
        """Batch upsert chunks and vectors."""
        if len(chunks) != len(vectors):
            raise ValueError("Chunks count and vectors count must match.")
        for chunk, vec in zip(chunks, vectors):
            self.add(chunk, vec)

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Fetch chunk metadata and content by ID."""
        record = self.records.get(chunk_id)
        if not record:
            return None
        return DocumentChunk(
            chunk_id=record.chunk_id,
            document_id=record.document_id,
            content=record.content,
            char_start=record.metadata.get("char_start", 0),
            char_end=record.metadata.get("char_end", len(record.content)),
            token_count=record.metadata.get("token_count", len(record.content) // 4),
            metadata=record.metadata,
        )

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Fast scalar product cosine similarity assuming normalized vectors."""
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        return float(dot)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Calculates cosine similarities against all stored vectors.
        Returns top_k tuples of (chunk_id, similarity_score).
        """
        if not self.records:
            return []

        results: List[Tuple[str, float]] = []

        for chunk_id, record in self.records.items():
            if metadata_filter:
                match = all(
                    record.metadata.get(k) == v for k, v in metadata_filter.items()
                )
                if not match:
                    continue

            sim = self._cosine_similarity(query_vector, record.vector)
            results.append((chunk_id, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def save_to_file(self, file_path: Path) -> None:
        """Save vector records to JSON file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = [rec.to_dict() for rec in self.records.values()]
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2)

    @classmethod
    def load_from_file(cls, file_path: Path) -> "VectorStore":
        """Load vector records from JSON file."""
        store = cls()
        if not file_path.exists():
            return store
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            rec = VectorRecord.from_dict(item)
            store.records[rec.chunk_id] = rec
        return store
