"""FastAPI service exposing the RAG pipeline.

Run:  uvicorn api.main:app --reload
Docs: http://localhost:8000/docs
"""
from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag_assistant.pipeline import RAGPipeline

app = FastAPI(title="Enterprise RAG Assistant", version="1.0.0")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


@app.get("/health")
def health() -> dict:
    p = get_pipeline()
    return {"status": "ok", "chunks_indexed": len(p.chunks),
            "retrieval_backend": f"bm25+{p.retriever.dense_backend}"}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    return get_pipeline().ask(req.question)
