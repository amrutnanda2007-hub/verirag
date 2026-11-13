"""
Unit Tests for BM25Okapi Lexical Index.
"""

from verirag.core.bm25 import BM25Index


def test_bm25_tokenize():
    tokens = BM25Index.tokenize("Hello, World! This is a test of BM25 tokenization.")
    assert "world" in tokens
    assert "test" in tokens
    assert "bm25" in tokens
    assert "is" not in tokens  # Stopword removed


def test_bm25_search_ranking():
    index = BM25Index()
    docs = [
        ("doc_1", "The Raft consensus algorithm elects a leader to manage log replication."),
        ("doc_2", "Zero Trust Architecture eliminates traditional perimeter network security."),
        ("doc_3", "Apache Kafka uses partitioned commit logs to achieve high throughput event streaming."),
    ]
    index.add_documents(docs)

    results = index.search("Raft leader election", top_k=3)
    assert len(results) > 0
    top_doc_id, top_score = results[0]
    assert top_doc_id == "doc_1"
    assert top_score > 0.0

    kafka_results = index.search("Kafka event streaming", top_k=3)
    assert kafka_results[0][0] == "doc_3"
