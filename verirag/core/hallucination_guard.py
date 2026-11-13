"""
Hallucination Guardrails & Claim Verification Engine.
Decomposes generated LLM responses into atomic factual statements, verifies each claim against
retrieved source contexts, and produces verifiable citation links and faithfulness scores.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from verirag.core.chunking import DocumentChunk


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    UNSUPPORTED = "UNSUPPORTED"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass
class ClaimVerification:
    claim_id: str
    claim_text: str
    status: VerificationStatus
    confidence_score: float
    supporting_chunk_id: Optional[str] = None
    supporting_quote: Optional[str] = None
    citation_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "status": self.status.value,
            "confidence_score": round(self.confidence_score, 4),
            "supporting_chunk_id": self.supporting_chunk_id,
            "supporting_quote": self.supporting_quote,
            "citation_index": self.citation_index,
        }


@dataclass
class GuardrailReport:
    total_claims: int
    verified_claims: int
    unsupported_claims: int
    faithfulness_score: float
    hallucination_index: float
    is_safe: bool
    verified_text: str
    claims: List[ClaimVerification] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_claims": self.total_claims,
            "verified_claims": self.verified_claims,
            "unsupported_claims": self.unsupported_claims,
            "faithfulness_score": round(self.faithfulness_score, 4),
            "hallucination_index": round(self.hallucination_index, 4),
            "is_safe": self.is_safe,
            "verified_text": self.verified_text,
            "claims": [c.to_dict() for c in self.claims],
        }


class HallucinationGuard:
    """
    Evaluates factual alignment between generated responses and retrieved source contexts.
    Detects ungrounded assertions, assigns citation indices, and computes faithfulness metrics.
    """

    def __init__(
        self,
        entailment_threshold: float = 0.60,
        max_hallucination_index: float = 0.25,
    ):
        self.entailment_threshold = entailment_threshold
        self.max_hallucination_index = max_hallucination_index

    @staticmethod
    def _extract_sentences(text: str) -> List[str]:
        """Splits answer text into discrete factual sentences."""
        # Clean markdown bullets and numbered lists
        clean = re.sub(r"^\s*[\*\-\d\.]+\s+", "", text, flags=re.MULTILINE)
        raw_sentences = re.split(r"(?<=[.!?])\s+", clean)
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 8]
        return sentences

    @staticmethod
    def _calculate_token_entailment(
        claim: str, context: str
    ) -> Tuple[float, Optional[str]]:
        """
        Calculates lexical entailment and finds best supporting snippet in context.
        """
        claim_tokens = [w.lower() for w in re.findall(r"\w+", claim) if len(w) > 2]
        if not claim_tokens:
            return 0.0, None

        # Find best matching sentence in context
        context_sentences = re.split(r"(?<=[.!?])\s+", context)
        best_overlap = 0.0
        best_sentence = None

        for c_sent in context_sentences:
            c_tokens = set(re.findall(r"\w+", c_sent.lower()))
            if not c_tokens:
                continue

            matched = sum(1 for t in claim_tokens if t in c_tokens)
            overlap_ratio = matched / len(claim_tokens)

            # Sequence match bonus for sub-phrases
            if len(claim_tokens) >= 3:
                for k in range(len(claim_tokens) - 2):
                    tri = f"{claim_tokens[k]} {claim_tokens[k+1]} {claim_tokens[k+2]}"
                    if tri in c_sent.lower():
                        overlap_ratio = min(1.0, overlap_ratio + 0.25)
                        break

            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_sentence = c_sent.strip()

        return best_overlap, best_sentence

    def verify_answer(
        self, generated_answer: str, context_chunks: List[DocumentChunk]
    ) -> GuardrailReport:
        """
        Verifies every statement in the generated answer against provided evidence chunks.
        Appends bracketed citation numbers to sentences that are verified.
        """
        sentences = self._extract_sentences(generated_answer)
        if not sentences:
            return GuardrailReport(
                total_claims=0,
                verified_claims=0,
                unsupported_claims=0,
                faithfulness_score=1.0,
                hallucination_index=0.0,
                is_safe=True,
                verified_text=generated_answer,
                claims=[],
            )

        claims: List[ClaimVerification] = []
        verified_count = 0
        unsupported_count = 0

        citation_map: Dict[str, int] = {}
        annotated_sentences: List[str] = []

        for idx, sentence in enumerate(sentences):
            best_score = 0.0
            best_chunk: Optional[DocumentChunk] = None
            best_quote: Optional[str] = None

            for chunk in context_chunks:
                score, quote = self._calculate_token_entailment(sentence, chunk.content)
                if score > best_score:
                    best_score = score
                    best_chunk = chunk
                    best_quote = quote

            claim_id = f"clm_{idx + 1:03d}"

            if best_score >= self.entailment_threshold and best_chunk:
                status = VerificationStatus.VERIFIED
                verified_count += 1

                if best_chunk.chunk_id not in citation_map:
                    citation_map[best_chunk.chunk_id] = len(citation_map) + 1
                cit_idx = citation_map[best_chunk.chunk_id]

                claims.append(
                    ClaimVerification(
                        claim_id=claim_id,
                        claim_text=sentence,
                        status=status,
                        confidence_score=best_score,
                        supporting_chunk_id=best_chunk.chunk_id,
                        supporting_quote=best_quote,
                        citation_index=cit_idx,
                    )
                )
                annotated_sentences.append(f"{sentence} [{cit_idx}]")
            else:
                status = VerificationStatus.UNSUPPORTED
                unsupported_count += 1
                claims.append(
                    ClaimVerification(
                        claim_id=claim_id,
                        claim_text=sentence,
                        status=status,
                        confidence_score=best_score,
                        supporting_chunk_id=None,
                        supporting_quote=None,
                        citation_index=None,
                    )
                )
                annotated_sentences.append(sentence)

        total = len(sentences)
        faithfulness = verified_count / max(1, total)
        hallucination_index = 1.0 - faithfulness
        is_safe = hallucination_index <= self.max_hallucination_index

        verified_text = " ".join(annotated_sentences)

        return GuardrailReport(
            total_claims=total,
            verified_claims=verified_count,
            unsupported_claims=unsupported_count,
            faithfulness_score=faithfulness,
            hallucination_index=hallucination_index,
            is_safe=is_safe,
            verified_text=verified_text,
            claims=claims,
        )
