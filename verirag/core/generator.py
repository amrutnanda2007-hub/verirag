"""
RAG Generator & Response Synthesis Engine.
Formats retrieved context, invokes LLM generation, and coordinates with HallucinationGuard.
"""

from dataclasses import dataclass
import json
import os
import re
from typing import Any, Dict, List, Optional
import urllib.request

from verirag.core.chunking import DocumentChunk
from verirag.core.hallucination_guard import GuardrailReport, HallucinationGuard
from verirag.core.hybrid_retriever import RetrievalResult


@dataclass
class GenerationResult:
    query: str
    raw_answer: str
    verified_answer: str
    guardrail_report: GuardrailReport
    retrieved_contexts: List[RetrievalResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "raw_answer": self.raw_answer,
            "verified_answer": self.verified_answer,
            "guardrail_report": self.guardrail_report.to_dict(),
            "retrieved_contexts": [c.to_dict() for c in self.retrieved_contexts],
        }


class RAGGenerator:
    """
    Synthesizes grounded answers using retrieved contexts and guarantees citation accuracy.
    """

    SYSTEM_PROMPT = """You are VeriRAG, a verified enterprise question-answering assistant.
Your answers MUST be strictly grounded in the provided Source Passages.
Every claim must be supported by the passages. If the passages do not contain enough information to answer truthfully, state clearly: "I cannot answer this based on the provided documents."
Do not extrapolate or speculate."""

    def __init__(
        self,
        hallucination_guard: Optional[HallucinationGuard] = None,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
    ):
        self.guard = hallucination_guard or HallucinationGuard()
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model_name = model_name

    def _build_context_prompt(self, contexts: List[RetrievalResult]) -> str:
        prompt_parts: List[str] = []
        for idx, res in enumerate(contexts):
            doc_id = res.chunk.document_id
            prompt_parts.append(f"[Source Passage {idx + 1} | Document: {doc_id}]\n{res.chunk.content}\n")
        return "\n".join(prompt_parts)

    def _local_synthesizer(self, query: str, contexts: List[RetrievalResult]) -> str:
        """
        Deterministic, offline extraction synthesizer.
        Finds the highest-scoring sentences directly answering the query keywords.
        """
        if not contexts:
            return "I cannot answer this question because no relevant documents were retrieved."

        q_terms = set(re.findall(r"\w+", query.lower()))
        matched_sentences: List[Tuple[float, str]] = []

        for res in contexts:
            sentences = re.split(r"(?<=[.!?])\s+", res.chunk.content)
            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) < 15:
                    continue
                s_terms = set(re.findall(r"\w+", s_clean.lower()))
                common = q_terms.intersection(s_terms)
                score = len(common) / max(1, len(q_terms))
                if score > 0.2:
                    matched_sentences.append((score, s_clean))

        matched_sentences.sort(key=lambda x: x[0], reverse=True)

        if not matched_sentences:
            top_content = contexts[0].chunk.content.strip()
            first_sentence = top_content.split(". ")[0] + "."
            return f"Based on the retrieved documentation: {first_sentence}"

        # Deduplicate and build answer
        seen = set()
        selected: List[str] = []
        for _, sent in matched_sentences[:3]:
            if sent not in seen:
                seen.add(sent)
                selected.append(sent)

        return " ".join(selected)

    def _call_openai(self, query: str, context_str: str) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Source Passages:\n{context_str}\n\nQuestion: {query}\n\nAnswer:",
                },
            ],
            "temperature": 0.1,
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_api_key}",
            },
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()

    def generate(
        self, query: str, retrieved_contexts: List[RetrievalResult]
    ) -> GenerationResult:
        context_str = self._build_context_prompt(retrieved_contexts)

        # 1. Synthesize raw answer
        if self.openai_api_key:
            try:
                raw_answer = self._call_openai(query, context_str)
            except Exception:
                raw_answer = self._local_synthesizer(query, retrieved_contexts)
        else:
            raw_answer = self._local_synthesizer(query, retrieved_contexts)

        # 2. Run Hallucination Guardrail Verification
        chunks = [res.chunk for res in retrieved_contexts]
        guard_report = self.guard.verify_answer(raw_answer, chunks)

        return GenerationResult(
            query=query,
            raw_answer=raw_answer,
            verified_answer=guard_report.verified_text,
            guardrail_report=guard_report,
            retrieved_contexts=retrieved_contexts,
        )
