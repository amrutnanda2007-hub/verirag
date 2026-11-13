"""
Pure Python BM25Okapi Inverted Index & Lexical Search Engine.
Provides sparse token-matching retrieval without external heavyweight dependencies.
"""

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Tuple


class BM25Index:
    """
    BM25Okapi sparse retrieval index with token-frequency indexing and IDF caching.
    """

    STOPWORDS: Set[str] = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
        "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
        "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
        "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
        "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
        "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
        "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
        "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
        "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
        "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
        "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
        "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
        "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
        "they've", "this", "those", "through", "to", "too", "under", "until", "up",
        "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
        "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
        "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
        "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
        "yourself", "yourselves",
    }

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_lengths: Dict[str, int] = {}
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.inverted_index: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.idf_cache: Dict[str, float] = {}

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """Lowercases, extracts alphanumeric tokens, and strips common stopwords."""
        raw_tokens = re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())
        return [t for t in raw_tokens if t not in cls.STOPWORDS]

    def add_documents(self, documents: List[Tuple[str, str]]) -> None:
        """
        Adds a batch of (doc_id, text) pairs and recalculates IDF.
        """
        for doc_id, text in documents:
            tokens = self.tokenize(text)
            doc_len = len(tokens)
            self.doc_lengths[doc_id] = doc_len
            self.corpus_size += 1

            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                self.inverted_index[term][doc_id] = count
                self.doc_freqs[term] += 1

        total_tokens = sum(self.doc_lengths.values())
        self.avg_doc_len = total_tokens / max(1, self.corpus_size)
        self._calculate_idf()

    def _calculate_idf(self) -> None:
        """Computes Robertson-Spärck Jones IDF with additive smoothing."""
        self.idf_cache.clear()
        for term, df in self.doc_freqs.items():
            # Standard Lucene/BM25Okapi IDF formula with +1 smoothing
            idf = math.log(((self.corpus_size - df + 0.5) / (df + 0.5)) + 1.0)
            self.idf_cache[term] = max(1e-4, idf)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Calculates BM25 scores for a query across all indexed documents.
        Returns top_k tuples of (doc_id, score).
        """
        query_tokens = self.tokenize(query)
        if not query_tokens or self.corpus_size == 0:
            return []

        scores: Dict[str, float] = defaultdict(float)

        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            idf = self.idf_cache.get(term, 0.0)
            postings = self.inverted_index[term]

            for doc_id, tf in postings.items():
                doc_len = self.doc_lengths.get(doc_id, 0)
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (
                    1.0 - self.b + self.b * (doc_len / max(1e-5, self.avg_doc_len))
                )
                scores[doc_id] += idf * (numerator / max(1e-5, denominator))

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    def save_to_file(self, file_path: Path) -> None:
        """Serializes index state to disk."""
        data = {
            "k1": self.k1,
            "b": self.b,
            "corpus_size": self.corpus_size,
            "avg_doc_len": self.avg_doc_len,
            "doc_lengths": self.doc_lengths,
            "doc_freqs": dict(self.doc_freqs),
            "inverted_index": {k: dict(v) for k, v in self.inverted_index.items()},
            "idf_cache": self.idf_cache,
        }
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_file(cls, file_path: Path) -> "BM25Index":
        """Loads index state from disk."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        idx = cls(k1=data["k1"], b=data["b"])
        idx.corpus_size = data["corpus_size"]
        idx.avg_doc_len = data["avg_doc_len"]
        idx.doc_lengths = data["doc_lengths"]
        idx.doc_freqs = defaultdict(int, data["doc_freqs"])
        idx.inverted_index = defaultdict(
            dict, {k: defaultdict(int, v) for k, v in data["inverted_index"].items()}
        )
        idx.idf_cache = data["idf_cache"]
        return idx
