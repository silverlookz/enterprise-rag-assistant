"""End-to-end RAG pipeline: ingest -> retrieve -> generate."""
from __future__ import annotations

import time
from pathlib import Path

from .generator import Answer, generate
from .ingest import build_chunks
from .retriever import HybridRetriever

DEFAULT_DOCS = Path(__file__).resolve().parents[2] / "data" / "docs"


class RAGPipeline:
    def __init__(self, docs_dir: str | Path = DEFAULT_DOCS, chunk_size: int = 180,
                 overlap: int = 40, top_k: int = 4, use_openai: bool | None = None):
        self.chunks = build_chunks(docs_dir, chunk_size, overlap)
        self.retriever = HybridRetriever(self.chunks, use_openai=use_openai)
        self.top_k = top_k

    def ask(self, question: str) -> dict:
        start = time.perf_counter()
        hits = self.retriever.search(question, k=self.top_k)
        answer: Answer = generate(question, hits)
        return {
            "question": question,
            "answer": answer.text,
            "grounded": answer.grounded,
            "citations": answer.citations,
            "mode": answer.mode,
            "retrieval_backend": f"bm25+{self.retriever.dense_backend}",
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        }


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What is the maximum loan-to-value for an uninsured mortgage?"
    result = RAGPipeline().ask(q)
    print(f"\nQ: {result['question']}\n\nA: {result['answer']}\n")
    for c in result["citations"]:
        print(f"  [{c['id']}] {c['source']} § {c['section']}")
