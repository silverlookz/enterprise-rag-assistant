"""Document loading and chunking.

Splits Markdown/text documents into overlapping, heading-aware chunks so every
chunk carries its source file and section title. That metadata is what makes
citations possible downstream.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_EXTENSIONS = {".md", ".txt"}


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    section: str
    metadata: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"{self.source} § {self.section}"


def load_documents(folder: str | Path) -> dict[str, str]:
    """Return {filename: text} for every supported file in a folder."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Document folder not found: {folder}")
    docs = {
        p.name: p.read_text(encoding="utf-8")
        for p in sorted(folder.iterdir())
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    }
    if not docs:
        raise ValueError(f"No .md or .txt documents found in {folder}")
    return docs


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split on Markdown headings; returns [(heading, body)]."""
    sections: list[tuple[str, str]] = []
    current_heading, buffer = "Introduction", []
    for line in text.splitlines():
        match = re.match(r"^#{1,6}\s+(.*)", line)
        if match:
            if any(l.strip() for l in buffer):
                sections.append((current_heading, "\n".join(buffer).strip()))
            current_heading, buffer = match.group(1).strip(), []
        else:
            buffer.append(line)
    if any(l.strip() for l in buffer):
        sections.append((current_heading, "\n".join(buffer).strip()))
    return sections


def _window(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if len(words) <= size:
        return [words]
    step = max(1, size - overlap)
    windows = []
    for start in range(0, len(words), step):
        windows.append(words[start : start + size])
        if start + size >= len(words):
            break
    return windows


def chunk_document(
    source: str, text: str, chunk_size: int = 180, overlap: int = 40
) -> list[Chunk]:
    """Heading-aware sliding-window chunking (sizes in words)."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks: list[Chunk] = []
    for s_idx, (heading, body) in enumerate(_split_sections(text)):
        for w_idx, words in enumerate(_window(body.split(), chunk_size, overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{Path(source).stem}-{s_idx}-{w_idx}",
                    text=" ".join(words),
                    source=source,
                    section=heading,
                )
            )
    return chunks


def build_chunks(folder: str | Path, chunk_size: int = 180, overlap: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []
    for name, text in load_documents(folder).items():
        chunks.extend(chunk_document(name, text, chunk_size, overlap))
    return chunks
