# VeriRAG Architecture Specification

## 1. System Pipeline

```mermaid
sequenceDiagram
    participant User
    participant API as FastAPI / CLI
    participant Retriever as HybridRetriever
    participant BM25 as BM25Index
    participant Vec as VectorStore
    participant Rerank as CrossEncoder
    participant LLM as RAGGenerator
    participant Guard as HallucinationGuard

    User->>API: Query("How does Raft elect a leader?")
    API->>Retriever: retrieve(query, top_k=5)
    par Sparse Lookup
        Retriever->>BM25: search(tokens)
    and Dense Lookup
        Retriever->>Vec: search(embedding_vector)
    end
    Retriever->>Retriever: Reciprocal Rank Fusion (RRF)
    Retriever->>Rerank: rerank(candidate_passages)
    Rerank-->>Retriever: Ranked passages
    Retriever-->>API: Top-K Context Passages
    API->>LLM: generate(query, passages)
    LLM-->>Guard: raw_answer
    Guard->>Guard: Decompose claims & evaluate entailment
    Guard-->>API: Verified answer + Citation tags + Confidence report
    API-->>User: Response with Citations & Guardrail report
```

## 2. Ingestion Mechanics
1. **Document Loading:** Ingests raw markdown or text.
2. **Recursive Splitting:** Employs hierarchical separators (`\n## `, `\n### `, `\n\n`, `\n`, `. `, ` `) respecting logical section boundaries.
3. **Dual Indexing:** Simultaneously populates the sparse inverted index with normalized term frequencies and the dense vector store with normalized embeddings.

## 3. Grounding Verification Algorithm
- Each statement $S_i$ in the output is scored against each retrieved passage $C_j$:
  $$\text{EntailmentScore}(S_i, C_j) = \frac{|\text{Tokens}(S_i) \cap \text{Tokens}(C_j)|}{|\text{Tokens}(S_i)|} + \text{N-GramBonus}$$
- If $\max_j \text{EntailmentScore}(S_i, C_j) \ge \tau$, the claim is tagged as `VERIFIED` and attributed to passage index $j$.
- If the overall `HallucinationIndex` exceeds 0.25, the system raises a safety flag.
