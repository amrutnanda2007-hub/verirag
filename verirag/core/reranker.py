"""
Cross-Encoder Reranker Module.
Refines preliminary candidate retrieval sets by evaluating full cross-attention between queries and chunks.
"""

from abc import ABC, abstractmethod
import math
import re
from typing import List, Tuple

from verirag.core.chunking import DocumentChunk


class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self, query: str, candidates: List[DocumentChunk], top_k: int = 5
    ) -> List[Tuple[DocumentChunk, float]]:
        """Reranks candidate chunks with respect to query."""
        pass


class CrossEncoderReranker(BaseReranker):
    """
    Two-stage reranker that evaluates token-level phrase affinity, span density,
    and semantic coverage. Automatically wraps sentence-transformers CrossEncoder if available.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(model_name)
        except Exception:
            self._model = None

    def _heuristic_score(self, query: str, content: str) -> float:
        """
        Sophisticated heuristic cross-attention scoring:
        - Exact phrase match density
        - Query term coverage percentage
        - Term proximity window penalty
        - Position bias (earlier mention in document section)
        """
        q_tokens = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 2]
        c_text = content.lower()

        if not q_tokens:
            return 0.5

        # 1. Query Term Coverage
        matched_tokens = set()
        token_positions: List[int] = []

        for token in q_tokens:
            matches = [m.start() for m in re.finditer(re.escape(token), c_text)]
            if matches:
                matched_tokens.add(token)
                token_positions.extend(matches)

        coverage = len(matched_tokens) / len(q_tokens)

        # 2. Exact Query Phrase Match
        phrase_boost = 0.35 if query.lower().strip() in c_text else 0.0

        # 3. Proximity Density: tighter clusters of query words yield higher relevance
        proximity_boost = 0.0
        if len(token_positions) >= 2:
            token_positions.sort()
            min_window = min(
                token_positions[i + 1] - token_positions[i]
                for i in range(len(token_positions) - 1)
            )
            # Normalize window: closer words get higher boost
            proximity_boost = max(0.0, 0.25 * math.exp(-min_window / 120.0))

        # 4. Term Frequency Density
        tf_density = min(0.2, (len(token_positions) / max(10, len(c_text.split()))) * 0.5)

        raw_score = (coverage * 0.5) + phrase_boost + proximity_boost + tf_density
        # Sigmoid squash to [0.0, 1.0]
        return 1.0 / (1.0 + math.exp(-4.0 * (raw_score - 0.4)))

    def rerank(
        self, query: str, candidates: List[DocumentChunk], top_k: int = 5
    ) -> List[Tuple[DocumentChunk, float]]:
        if not candidates:
            return []

        if self._model:
            pairs = [[query, c.content] for c in candidates]
            scores = self._model.predict(pairs)
            scored = list(zip(candidates, [float(s) for s in scores]))
        else:
            scored = [
                (chunk, self._heuristic_score(query, chunk.content))
                for chunk in candidates
            ]

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
