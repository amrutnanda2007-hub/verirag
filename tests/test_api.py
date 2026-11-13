"""
Integration Tests for VeriRAG FastAPI Endpoints.
"""

from fastapi.testclient import TestClient
from verirag.api.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "verirag"


def test_ingest_and_query_flow():
    # 1. Ingest document
    ingest_payload = {
        "document_id": "test_raft_doc",
        "content": (
            "Raft is a consensus protocol designed for manageability. "
            "It relies on leader election and replicated write-ahead logs. "
            "Heartbeat intervals keep followers from initiating new elections."
        ),
        "metadata": {"category": "distributed-systems"},
    }
    r_ingest = client.post("/api/v1/documents/ingest", json=ingest_payload)
    assert r_ingest.status_code == 200
    d_ingest = r_ingest.json()
    assert d_ingest["status"] == "success"
    assert d_ingest["chunks_created"] >= 1

    # 2. Query document
    query_payload = {
        "query": "How do heartbeat intervals affect Raft leader elections?",
        "top_k": 2,
        "enable_reranking": True,
        "enable_verification": True,
    }
    r_query = client.post("/api/v1/query", json=query_payload)
    assert r_query.status_code == 200
    d_query = r_query.json()
    assert len(d_query["retrieved_chunks"]) > 0
    assert d_query["faithfulness_score"] >= 0.0
    assert "answer" in d_query


def test_stats_endpoint():
    response = client.get("/api/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total_chunks"] >= 1
    assert data["total_documents"] >= 1
