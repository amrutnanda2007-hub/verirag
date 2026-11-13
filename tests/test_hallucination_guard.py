"""
Unit Tests for Hallucination Guardrails & Citation Verifier.
"""

from verirag.core.chunking import DocumentChunk
from verirag.core.hallucination_guard import HallucinationGuard, VerificationStatus


def test_hallucination_guard_verified_claim():
    guard = HallucinationGuard(entailment_threshold=0.5)
    chunks = [
        DocumentChunk(
            chunk_id="chk_01",
            document_id="doc_raft",
            content="Raft uses randomized election timers between 150ms and 300ms to elect a cluster leader.",
            char_start=0,
            char_end=85,
            token_count=20,
        )
    ]

    answer = "Raft uses randomized election timers between 150ms and 300ms to elect a cluster leader."
    report = guard.verify_answer(answer, chunks)

    assert report.total_claims == 1
    assert report.verified_claims == 1
    assert report.faithfulness_score == 1.0
    assert report.hallucination_index == 0.0
    assert report.is_safe is True
    assert "[1]" in report.verified_text


def test_hallucination_guard_unsupported_claim():
    guard = HallucinationGuard(entailment_threshold=0.6)
    chunks = [
        DocumentChunk(
            chunk_id="chk_01",
            document_id="doc_raft",
            content="Raft uses randomized election timers to avoid split votes in distributed clusters.",
            char_start=0,
            char_end=80,
            token_count=18,
        )
    ]

    hallucinated_answer = "The spacecraft landed safely on Mars after travelling for nine months through interplanetary space."
    report = guard.verify_answer(hallucinated_answer, chunks)

    assert report.total_claims == 1
    assert report.verified_claims == 0
    assert report.unsupported_claims == 1
    assert report.faithfulness_score == 0.0
    assert report.hallucination_index == 1.0
    assert report.is_safe is False
    assert report.claims[0].status == VerificationStatus.UNSUPPORTED
