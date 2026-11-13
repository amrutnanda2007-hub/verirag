# VeriRAG Triad Benchmark Report

| Metric | Score | Target Standard |
| :--- | :---: | :---: |
| **Context Precision** | `0.9583` | > 0.85 |
| **Context Recall** | `0.7278` | > 0.80 |
| **Faithfulness (Groundedness)** | `0.6378` | > 0.90 |
| **Answer Relevance** | `0.3873` | > 0.85 |
| **Composite RAG Index** | `0.6778` | > 0.85 |

## Per-Sample Breakdown

| Sample ID | Query | Precision | Recall | Faithfulness | Relevance | Avg |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `eval_001` | How does Raft leader election handl... | 0.92 | 0.93 | 0.78 | 0.46 | **0.77** |
| `eval_002` | What are the core NIST tenets of Ze... | 0.92 | 0.50 | 0.46 | 0.49 | **0.59** |
| `eval_003` | What is the formula and purpose of ... | 1.00 | 0.76 | 0.63 | 0.36 | **0.69** |
| `eval_004` | How does envelope encryption protec... | 1.00 | 0.72 | 0.68 | 0.24 | **0.66** |