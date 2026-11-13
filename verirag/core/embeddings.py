"""
Embedding Provider Module.
Provides modular dense representation engines: deterministic local hashing, SentenceTransformers, and OpenAI.
"""

from abc import ABC, abstractmethod
import hashlib
import math
import re
from typing import List, Optional


class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string into a normalized dense vector."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings into normalized dense vectors."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The vector dimension produced by this provider."""
        pass


class DeterministicEmbeddingProvider(BaseEmbeddingProvider):
    """
    Zero-dependency deterministic dense embedding generator.
    Constructs high-fidelity normalized 384-d vectors using n-gram feature projections and
    semantic hashing. Enables instantaneous testing and deterministic reproducible execution.
    """

    def __init__(self, dimension: int = 384):
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def _normalize(self, vector: List[float]) -> List[float]:
        norm = math.sqrt(sum(x * x for x in vector))
        if norm < 1e-12:
            return [0.0] * self._dim
        return [x / norm for x in vector]

    def embed_text(self, text: str) -> List[float]:
        vec = [0.0] * self._dim
        clean_text = text.lower().strip()
        tokens = re.findall(r"\w+", clean_text)

        if not tokens:
            return [0.0] * self._dim

        for pos, token in enumerate(tokens):
            # Token unigram hash projection
            h_token = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            idx1 = h_token % self._dim
            sign1 = 1.0 if ((h_token >> 8) & 1) == 1 else -1.0
            vec[idx1] += sign1 * 1.5

            # Subword character trigrams
            for i in range(len(token) - 2):
                tri = token[i : i + 3]
                h_tri = int(hashlib.md5(tri.encode("utf-8")).hexdigest(), 16)
                idx2 = h_tri % self._dim
                sign2 = 1.0 if ((h_tri >> 4) & 1) == 1 else -1.0
                vec[idx2] += sign2 * 0.4

            # Bigram contexts
            if pos < len(tokens) - 1:
                bigram = f"{token}_{tokens[pos+1]}"
                h_bi = int(hashlib.sha1(bigram.encode("utf-8")).hexdigest(), 16)
                idx3 = h_bi % self._dim
                vec[idx3] += 0.8

        return self._normalize(vec)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]


class SentenceTransformersEmbeddingProvider(BaseEmbeddingProvider):
    """HuggingFace sentence-transformers integration."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. Run `pip install sentence-transformers`."
            )

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> List[float]:
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [e.tolist() for e in embeddings]


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI Embeddings API integration."""

    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model_name = model_name
        self._dim = 1536 if "large" not in model_name else 3072

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            import urllib.request
            import json

            req = urllib.request.Request(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                data=json.dumps({"input": texts, "model": self.model_name}).encode("utf-8"),
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                return [item["embedding"] for item in res_data["data"]]
        except Exception as e:
            raise RuntimeError(f"OpenAI embedding request failed: {e}")


def get_embedding_provider(
    provider_name: str = "deterministic",
    model_name: str = "all-MiniLM-L6-v2",
    api_key: Optional[str] = None,
    dimension: int = 384,
) -> BaseEmbeddingProvider:
    """Factory creating the appropriate embedding provider."""
    name = provider_name.lower().strip()
    if name == "openai" and api_key:
        return OpenAIEmbeddingProvider(api_key=api_key, model_name=model_name)
    elif name == "sentence-transformers":
        try:
            return SentenceTransformersEmbeddingProvider(model_name=model_name)
        except Exception:
            return DeterministicEmbeddingProvider(dimension=dimension)
    else:
        return DeterministicEmbeddingProvider(dimension=dimension)
