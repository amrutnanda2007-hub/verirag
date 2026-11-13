"""
Recursive Text and Document Chunking Module.
Splits enterprise documents while preserving semantic hierarchy, markdown structure, and metadata.
"""

from dataclasses import dataclass, field
import hashlib
import re
from typing import Any, Dict, List, Optional


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    content: str
    char_start: int
    char_end: int
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        return cls(
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            content=data["content"],
            char_start=data["char_start"],
            char_end=data["char_end"],
            token_count=data["token_count"],
            metadata=data.get("metadata", {}),
        )


class RecursiveTokenChunker:
    """
    Splits text recursively using hierarchical separators to maintain semantic coherence.
    """

    DEFAULT_SEPARATORS = [
        "\n# ",
        "\n## ",
        "\n### ",
        "\n\n",
        "\n",
        ". ",
        "? ",
        "! ",
        "; ",
        ", ",
        " ",
        "",
    ]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: Optional[List[str]] = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def estimate_tokens(self, text: str) -> int:
        """Heuristic token estimator (~4 characters per token)."""
        return max(1, len(text) // 4)

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators: List[str] = []

        for i, s in enumerate(separators):
            if s == "":
                separator = s
                break
            if s in text:
                separator = s
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)

        good_splits: List[str] = []
        for s in splits:
            if not s.strip():
                continue
            if self.estimate_tokens(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if new_separators:
                    sub_splits = self._split_text(s, new_separators)
                    good_splits.extend(sub_splits)
                else:
                    good_splits.append(s)

        # Merge splits with overlap
        current_chunk: List[str] = []
        current_length = 0

        for piece in good_splits:
            piece_len = self.estimate_tokens(piece)
            if current_length + piece_len > self.chunk_size and current_chunk:
                merged = separator.join(current_chunk).strip()
                if merged:
                    final_chunks.append(merged)

                # Keep overlap items from the end of the current chunk
                overlap_chunk: List[str] = []
                overlap_len = 0
                for item in reversed(current_chunk):
                    item_len = self.estimate_tokens(item)
                    if overlap_len + item_len <= self.chunk_overlap:
                        overlap_chunk.insert(0, item)
                        overlap_len += item_len
                    else:
                        break
                current_chunk = overlap_chunk
                current_length = overlap_len

            current_chunk.append(piece)
            current_length += piece_len

        if current_chunk:
            merged = separator.join(current_chunk).strip()
            if merged:
                final_chunks.append(merged)

        return final_chunks

    def chunk_document(
        self,
        document_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """
        Chunks an entire document text into DocumentChunk instances with span indexing.
        """
        metadata = metadata or {}
        raw_chunks = self._split_text(content, self.separators)
        chunks: List[DocumentChunk] = []

        last_search_pos = 0
        for idx, text in enumerate(raw_chunks):
            found_pos = content.find(text, last_search_pos)
            if found_pos != -1:
                char_start = found_pos
                char_end = found_pos + len(text)
                last_search_pos = max(0, char_start + 1)
            else:
                char_start = last_search_pos
                char_end = last_search_pos + len(text)

            chunk_hash = hashlib.sha256(
                f"{document_id}_{idx}_{text[:32]}".encode("utf-8")
            ).hexdigest()[:12]
            chunk_id = f"chk_{chunk_hash}"

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    content=text,
                    char_start=char_start,
                    char_end=char_end,
                    token_count=self.estimate_tokens(text),
                    metadata={
                        **metadata,
                        "chunk_index": idx,
                        "total_chunks": len(raw_chunks),
                    },
                )
            )

        return chunks
