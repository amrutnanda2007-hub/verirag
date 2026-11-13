"""
RAG Triad Evaluator Module.
Calculates Context Precision, Context Recall, Faithfulness, and Answer Relevance benchmark scores.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from verirag.core.embeddings import BaseEmbeddingProvider


@dataclass
class EvalSampleResult:
    sample_id: str
    query: str
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevance: float
    average_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "query": self.query,
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevance": round(self.answer_relevance, 4),
            "average_score": round(self.average_score, 4),
        }


@dataclass
class BenchmarkReport:
    total_samples: int
    mean_context_precision: float
    mean_context_recall: float
    mean_faithfulness: float
    mean_answer_relevance: float
    composite_rag_score: float
    samples: List[EvalSampleResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "mean_context_precision": round(self.mean_context_precision, 4),
            "mean_context_recall": round(self.mean_context_recall, 4),
            "mean_faithfulness": round(self.mean_faithfulness, 4),
            "mean_answer_relevance": round(self.mean_answer_relevance, 4),
            "composite_rag_score": round(self.composite_rag_score, 4),
            "samples": [s.to_dict() for s in self.samples],
        }

    def to_markdown(self) -> str:
        lines = [
            "# VeriRAG Triad Benchmark Report",
            "",
            "| Metric | Score | Target Standard |",
            "| :--- | :---: | :---: |",
            f"| **Context Precision** | `{self.mean_context_precision:.4f}` | > 0.85 |",
            f"| **Context Recall** | `{self.mean_context_recall:.4f}` | > 0.80 |",
            f"| **Faithfulness (Groundedness)** | `{self.mean_faithfulness:.4f}` | > 0.90 |",
            f"| **Answer Relevance** | `{self.mean_answer_relevance:.4f}` | > 0.85 |",
            f"| **Composite RAG Index** | `{self.composite_rag_score:.4f}` | > 0.85 |",
            "",
            "## Per-Sample Breakdown",
            "",
            "| Sample ID | Query | Precision | Recall | Faithfulness | Relevance | Avg |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
        for s in self.samples:
            lines.append(
                f"| `{s.sample_id}` | {s.query[:35]}... | {s.context_precision:.2f} | {s.context_recall:.2f} | {s.faithfulness:.2f} | {s.answer_relevance:.2f} | **{s.average_score:.2f}** |"
            )
        return "\n".join(lines)


class RAGEvaluator:
    """
    Automated evaluation harness for RAG pipelines without requiring external SaaS test accounts.
    """

    def __init__(self, embedding_provider: BaseEmbeddingProvider):
        self.embedding_provider = embedding_provider

    def _cosine(self, vec_a: List[float], vec_b: List[float]) -> float:
        return max(0.0, min(1.0, sum(a * b for a, b in zip(vec_a, vec_b))))

    def evaluate_sample(
        self,
        sample_id: str,
        query: str,
        retrieved_contexts: List[str],
        generated_answer: str,
        ground_truth_answer: str,
    ) -> EvalSampleResult:
        # 1. Context Precision: are top chunks aligned with the ground truth?
        gt_terms = set(re.findall(r"\w+", ground_truth_answer.lower()))
        precisions: List[float] = []
        for rank, ctx in enumerate(retrieved_contexts):
            c_terms = set(re.findall(r"\w+", ctx.lower()))
            overlap = len(gt_terms.intersection(c_terms)) / max(1, len(gt_terms))
            if overlap > 0.15:
                precisions.append(1.0 / (rank + 1))

        context_precision = sum(precisions) / max(1, len(retrieved_contexts))
        context_precision = min(1.0, context_precision * 1.5)  # normalize

        # 2. Context Recall: does the combined context cover the ground truth?
        all_context_terms = set(re.findall(r"\w+", " ".join(retrieved_contexts).lower()))
        recall_hits = sum(1 for t in gt_terms if t in all_context_terms)
        context_recall = recall_hits / max(1, len(gt_terms))

        # 3. Faithfulness: are generated claims supported by context?
        ans_terms = set(re.findall(r"\w+", generated_answer.lower()))
        faithfulness = sum(1 for t in ans_terms if t in all_context_terms) / max(1, len(ans_terms))

        # 4. Answer Relevance: semantic alignment between query and generated answer
        q_vec = self.embedding_provider.embed_text(query)
        a_vec = self.embedding_provider.embed_text(generated_answer)
        answer_relevance = self._cosine(q_vec, a_vec)

        avg_score = (context_precision + context_recall + faithfulness + answer_relevance) / 4.0

        return EvalSampleResult(
            sample_id=sample_id,
            query=query,
            context_precision=context_precision,
            context_recall=context_recall,
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            average_score=avg_score,
        )

    def run_benchmark(self, dataset_path: Path) -> BenchmarkReport:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        results: List[EvalSampleResult] = []
        for idx, item in enumerate(dataset):
            sid = item.get("id", f"smp_{idx+1:03d}")
            res = self.evaluate_sample(
                sample_id=sid,
                query=item["query"],
                retrieved_contexts=item.get("contexts", []),
                generated_answer=item.get("answer", ""),
                ground_truth_answer=item.get("ground_truth", ""),
            )
            results.append(res)

        total = len(results)
        m_prec = sum(r.context_precision for r in results) / max(1, total)
        m_rec = sum(r.context_recall for r in results) / max(1, total)
        m_faith = sum(r.faithfulness for r in results) / max(1, total)
        m_rel = sum(r.answer_relevance for r in results) / max(1, total)
        composite = (m_prec + m_rec + m_faith + m_rel) / 4.0

        return BenchmarkReport(
            total_samples=total,
            mean_context_precision=m_prec,
            mean_context_recall=m_rec,
            mean_faithfulness=m_faith,
            mean_answer_relevance=m_rel,
            composite_rag_score=composite,
            samples=results,
        )
