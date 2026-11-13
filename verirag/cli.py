"""
VeriRAG Command Line Interface (CLI).
Command-line toolkit for indexing documents, running queries with citation verification,
benchmarking RAG triad metrics, and launching the REST API.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import List

from verirag import __version__
from verirag.config import settings
from verirag.core.bm25 import BM25Index
from verirag.core.chunking import RecursiveTokenChunker
from verirag.core.embeddings import get_embedding_provider
from verirag.core.evaluator import RAGEvaluator
from verirag.core.generator import RAGGenerator
from verirag.core.hallucination_guard import HallucinationGuard
from verirag.core.hybrid_retriever import HybridRetriever
from verirag.core.reranker import CrossEncoderReranker
from verirag.core.vector_store import VectorStore

# ANSI color codes
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_banner():
    banner = f"""{CYAN}{BOLD}
 __      __        _ _____         _____ 
 \\ \\    / /       (_)  __ \\   /\\  / ____|
  \\ \\  / /__ _ __  _| |__) | /  \\| |  __ 
   \\ \\/ / _ \\ '__| | |  _  / / /\\ \\ | |_ |
    \\  /  __/ |  | | | | \\ \\/ ____ \\ |__| |
     \\/ \\___|_|  |_|_|_|  \\_/_/    \\_\\_____|
     Self-Correcting RAG & Citation Guard (v{__version__})
{RESET}"""
    print(banner)


def get_engine():
    chunker = RecursiveTokenChunker(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    embedder = get_embedding_provider(
        provider_name=settings.embedding_provider,
        model_name=settings.embedding_model,
        api_key=settings.openai_api_key,
        dimension=settings.embedding_dimension,
    )
    vector_store = VectorStore.load_from_file(settings.vector_store_path)
    if settings.bm25_index_path.exists():
        bm25_index = BM25Index.load_from_file(settings.bm25_index_path)
    else:
        bm25_index = BM25Index()

    reranker = CrossEncoderReranker() if settings.enable_reranking else None
    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        embedding_provider=embedder,
        reranker=reranker,
        rrf_k=settings.rrf_k,
        alpha=settings.hybrid_alpha,
    )
    guard = HallucinationGuard(
        entailment_threshold=settings.claim_entailment_threshold,
        max_hallucination_index=settings.max_hallucination_index,
    )
    generator = RAGGenerator(
        hallucination_guard=guard,
        openai_api_key=settings.openai_api_key,
    )
    evaluator = RAGEvaluator(embedding_provider=embedder)

    return chunker, embedder, vector_store, bm25_index, retriever, guard, generator, evaluator


def cmd_ingest(args):
    target_path = Path(args.path)
    if not target_path.exists():
        print(f"{RED}Error: Path {target_path} does not exist.{RESET}")
        sys.exit(1)

    chunker, _, vector_store, bm25_index, retriever, _, _, _ = get_engine()

    files_to_process: List[Path] = []
    if target_path.is_file():
        files_to_process.append(target_path)
    else:
        files_to_process = list(target_path.glob("**/*.md")) + list(
            target_path.glob("**/*.txt")
        )

    print(f"{CYAN}Ingesting {len(files_to_process)} document(s)...{RESET}")
    total_chunks = 0

    for f in files_to_process:
        doc_id = f.stem
        content = f.read_text(encoding="utf-8")
        chunks = chunker.chunk_document(document_id=doc_id, content=content)
        retriever.index_document(chunks)
        total_chunks += len(chunks)
        print(f"  {GREEN}✓{RESET} {f.name} -> {len(chunks)} chunks indexed")

    vector_store.save_to_file(settings.vector_store_path)
    bm25_index.save_to_file(settings.bm25_index_path)

    print(
        f"\n{BOLD}{GREEN}Successfully indexed {total_chunks} total chunks into VeriRAG!{RESET}"
    )


def cmd_query(args):
    _, _, vector_store, _, retriever, _, generator, _ = get_engine()

    if len(vector_store) == 0:
        print(
            f"{YELLOW}Warning: Knowledge base is empty. Ingest documents first using `verirag ingest`.{RESET}"
        )
        sys.exit(1)

    print(f"\n{BOLD}Query:{RESET} {args.query}\n")
    retrieved = retriever.retrieve(
        query=args.query, top_k=args.top_k, enable_reranking=not args.no_rerank
    )

    if not retrieved:
        print(f"{RED}No matching context passages found.{RESET}")
        return

    print(f"{CYAN}--- Retrieved Context Passages ({len(retrieved)}) ---{RESET}")
    for res in retrieved:
        score_str = f"Rerank Score: {res.final_score:.4f}"
        print(
            f"[{BOLD}Passage {res.rank}{RESET} | Doc: {res.chunk.document_id} | {score_str}]"
        )
        print(f"  \"{res.chunk.content[:140]}...\"\n")

    res = generator.generate(query=args.query, retrieved_contexts=retrieved)

    print(f"{CYAN}--- Verified Answer ---{RESET}")
    print(f"{BOLD}{res.verified_answer}{RESET}\n")

    report = res.guardrail_report
    status_color = GREEN if report.is_safe else RED
    print(f"{CYAN}--- Hallucination Guardrail Report ---{RESET}")
    print(
        f"  Total Claims:         {report.total_claims} (Verified: {report.verified_claims}, Unsupported: {report.unsupported_claims})"
    )
    print(
        f"  Faithfulness Score:   {status_color}{report.faithfulness_score * 100:.1f}%{RESET}"
    )
    print(
        f"  Hallucination Index:  {status_color}{report.hallucination_index:.3f}{RESET}"
    )
    print(f"  Verification Status:  {status_color}{'SAFE' if report.is_safe else 'ALERT - UNGROUNDED'}{RESET}")

    if report.claims:
        print(f"\n  {BOLD}Claim Verification Table:{RESET}")
        for c in report.claims:
            color = GREEN if c.status.value == "VERIFIED" else RED
            cit = f"[{c.citation_index}]" if c.citation_index else "[-]"
            print(f"    {color}[{c.status.value}]{RESET} {cit} {c.claim_text}")
            if c.supporting_quote:
                print(f"      {CYAN}↳ Quote:{RESET} \"{c.supporting_quote[:90]}...\"")


def cmd_benchmark(args):
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"{RED}Error: Dataset {dataset_path} not found.{RESET}")
        sys.exit(1)

    _, embedder, _, _, _, _, _, evaluator = get_engine()
    print(f"{CYAN}Executing RAG Triad benchmark suite on {dataset_path.name}...{RESET}")
    report = evaluator.run_benchmark(dataset_path)

    print("\n" + report.to_markdown())

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(report.to_markdown(), encoding="utf-8")
        print(f"\n{GREEN}✓ Benchmark report saved to {out_path}{RESET}")


def cmd_serve(args):
    try:
        import uvicorn
        print(f"{GREEN}Starting VeriRAG API on http://{args.host}:{args.port}{RESET}")
        uvicorn.run("verirag.api.app:app", host=args.host, port=args.port, reload=args.reload)
    except ImportError:
        print(f"{RED}uvicorn is required to run the server. Run `pip install uvicorn`.{RESET}")


def main():
    print_banner()
    parser = argparse.ArgumentParser(
        prog="verirag",
        description="VeriRAG: Self-Correcting RAG with Hallucination Guardrails",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest and chunk documents into the knowledge base")
    p_ingest.add_argument("path", help="File or folder path containing markdown/text documents")

    # Query
    p_query = subparsers.add_parser("query", help="Query the RAG engine with citation verification")
    p_query.add_argument("query", help="Question prompt")
    p_query.add_argument("--top-k", type=int, default=3, help="Number of passages to retrieve")
    p_query.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder reranking")

    # Benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run RAG Triad evaluation benchmarks")
    p_bench.add_argument("--dataset", default="sample_data/eval_gold_dataset.json", help="Path to evaluation JSON")
    p_bench.add_argument("--output", default="docs/benchmark_results.md", help="Path to save markdown report")

    # Serve
    p_serve = subparsers.add_parser("serve", help="Launch FastAPI REST server")
    p_serve.add_argument("--host", default="127.0.0.1", help="Host interface")
    p_serve.add_argument("--port", type=int, default=8000, help="Port")
    p_serve.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()
    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "serve":
        cmd_serve(args)


if __name__ == "__main__":
    main()
