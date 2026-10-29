"""Answer generation with mandatory citations.

LLM mode: LangChain + OpenAI, with a prompt that forbids answering outside the
retrieved context and requires [n] citations.
Offline mode: extractive answer built from the highest-overlap sentences, so
the pipeline (and CI) works with no API key.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .retriever import RetrievedChunk, tokenize

NO_ANSWER = "I couldn't find this in the indexed documents."

SYSTEM_PROMPT = """You are an internal policy assistant.
Answer ONLY from the numbered context passages below.
Cite every claim with the passage number in square brackets, e.g. [1] or [2][3].
If the context does not contain the answer, reply exactly: "{no_answer}"
Be concise: 2-5 sentences."""


@dataclass
class Answer:
    text: str
    citations: list[dict]
    mode: str
    grounded: bool


def _format_context(hits: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{i}] ({h.chunk.citation})\n{h.chunk.text}" for i, h in enumerate(hits, start=1)
    )


def _citations(hits: list[RetrievedChunk]) -> list[dict]:
    return [
        {"id": i, "source": h.chunk.source, "section": h.chunk.section,
         "chunk_id": h.chunk.chunk_id, "score": round(h.score, 4)}
        for i, h in enumerate(hits, start=1)
    ]


def extractive_answer(question: str, hits: list[RetrievedChunk], max_sentences: int = 3,
                      min_overlap: int = 2) -> Answer:
    q_terms = set(tokenize(question))
    scored = []
    for i, hit in enumerate(hits, start=1):
        for sent in re.split(r"(?<=[.!?])\s+", hit.chunk.text):
            overlap = len(q_terms & set(tokenize(sent)))
            if overlap:
                scored.append((overlap, -i, sent.strip(), i))
    scored.sort(reverse=True)
    if not scored or scored[0][0] < min(min_overlap, len(q_terms)):
        return Answer(NO_ANSWER, [], "extractive", grounded=False)
    picked = scored[:max_sentences]
    text = " ".join(f"{s} [{i}]" for _, _, s, i in picked)
    used = sorted({i for *_, i in picked})
    cites = [c for c in _citations(hits) if c["id"] in used]
    return Answer(text, cites, "extractive", grounded=True)


def llm_answer(question: str, hits: list[RetrievedChunk]) -> Answer:  # pragma: no cover
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT.format(no_answer=NO_ANSWER)),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])
    llm = ChatOpenAI(model=os.getenv("CHAT_MODEL", "gpt-4o-mini"), temperature=0)
    text = (prompt | llm).invoke(
        {"context": _format_context(hits), "question": question}
    ).content.strip()
    used = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    cites = [c for c in _citations(hits) if c["id"] in used]
    return Answer(text, cites, "llm", grounded=text != NO_ANSWER and bool(cites))


def generate(question: str, hits: list[RetrievedChunk]) -> Answer:
    if not hits:
        return Answer(NO_ANSWER, [], "none", grounded=False)
    if os.getenv("OPENAI_API_KEY"):
        return llm_answer(question, hits)
    return extractive_answer(question, hits)
