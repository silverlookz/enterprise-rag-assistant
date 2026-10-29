"""Hybrid retrieval: BM25 (lexical) + vector similarity, fused with RRF.

Lexical search catches exact policy terms ("LTV", "Form 4"), vector search
catches paraphrases. Reciprocal Rank Fusion combines both without having to
calibrate their score scales against each other.

The dense side uses OpenAI embeddings through LangChain when an API key is
configured, and falls back to TF-IDF so the project runs fully offline.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from .ingest import Chunk

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "to", "in", "on", "for",
    "and", "or", "what", "how", "do", "does", "can", "i", "we", "it", "be", "by",
    "with", "at", "as", "that", "this", "which", "who", "when", "my", "our",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    lexical_rank: int | None
    dense_rank: int | None


class _TfidfEmbedder:
    def __init__(self, texts: list[str]):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(texts)

    def doc_matrix(self) -> np.ndarray:
        return self.matrix

    def similarity(self, query: str) -> np.ndarray:
        q = self.vectorizer.transform([query])
        return (self.matrix @ q.T).toarray().ravel()


class _OpenAIEmbedder:  # pragma: no cover - needs network + key
    def __init__(self, texts: list[str], model: str):
        from langchain_openai import OpenAIEmbeddings

        self.client = OpenAIEmbeddings(model=model)
        vecs = np.array(self.client.embed_documents(texts))
        self.matrix = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    def similarity(self, query: str) -> np.ndarray:
        q = np.array(self.client.embed_query(query))
        return self.matrix @ (q / np.linalg.norm(q))


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], rrf_k: int = 60, use_openai: bool | None = None):
        if not chunks:
            raise ValueError("Retriever needs at least one chunk")
        self.chunks = chunks
        self.rrf_k = rrf_k
        texts = [f"{c.section}. {c.text}" for c in chunks]
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])

        if use_openai is None:
            use_openai = bool(os.getenv("OPENAI_API_KEY"))
        if use_openai:
            model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            self.embedder = _OpenAIEmbedder(texts, model)
            self.dense_backend = f"openai:{model}"
        else:
            self.embedder = _TfidfEmbedder(texts)
            self.dense_backend = "tfidf"

    @staticmethod
    def _ranks(scores: np.ndarray, depth: int) -> dict[int, int]:
        order = np.argsort(-scores)[:depth]
        return {int(idx): rank for rank, idx in enumerate(order, start=1) if scores[idx] > 0}

    def search(self, query: str, k: int = 4, depth: int = 20) -> list[RetrievedChunk]:
        lexical = self._ranks(np.asarray(self.bm25.get_scores(tokenize(query))), depth)
        dense = self._ranks(self.embedder.similarity(query), depth)

        fused: dict[int, float] = {}
        for ranking in (lexical, dense):
            for idx, rank in ranking.items():
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)

        best = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [
            RetrievedChunk(self.chunks[i], s, lexical.get(i), dense.get(i)) for i, s in best
        ]
