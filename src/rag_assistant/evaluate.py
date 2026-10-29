"""Retrieval + answer evaluation harness.

Metrics on a labelled question set (data/eval/questions.jsonl):
  * Hit@k        - did the expected source appear in the top-k chunks?
  * MRR          - mean reciprocal rank of the first correct source
  * Groundedness - share of answerable questions that got a cited answer
  * Refusal acc. - share of unanswerable questions correctly refused
"""
from __future__ import annotations

import json
from pathlib import Path

from .pipeline import RAGPipeline

EVAL_FILE = Path(__file__).resolve().parents[2] / "data" / "eval" / "questions.jsonl"


def load_eval_set(path: str | Path = EVAL_FILE) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def evaluate(pipeline: RAGPipeline, items: list[dict], k: int = 4) -> dict:
    hits, rr, grounded, answerable, refused, unanswerable = 0, 0.0, 0, 0, 0, 0
    rows = []
    for item in items:
        expected = item.get("expected_source")
        retrieved = [r.chunk.source for r in pipeline.retriever.search(item["question"], k=k)]
        result = pipeline.ask(item["question"])
        if expected:
            answerable += 1
            rank = next((i for i, s in enumerate(retrieved, 1) if s == expected), None)
            hits += rank is not None
            rr += 1 / rank if rank else 0
            grounded += result["grounded"]
        else:
            unanswerable += 1
            refused += not result["grounded"]
        rows.append({"question": item["question"], "expected": expected,
                     "top_source": retrieved[0] if retrieved else None,
                     "grounded": result["grounded"]})
    return {
        "n": len(items),
        f"hit@{k}": round(hits / max(answerable, 1), 3),
        "mrr": round(rr / max(answerable, 1), 3),
        "groundedness": round(grounded / max(answerable, 1), 3),
        "refusal_accuracy": round(refused / max(unanswerable, 1), 3),
        "rows": rows,
    }


if __name__ == "__main__":
    report = evaluate(RAGPipeline(), load_eval_set())
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
