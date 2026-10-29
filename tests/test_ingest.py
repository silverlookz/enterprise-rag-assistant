import pytest

from rag_assistant.ingest import build_chunks, chunk_document

DOC = "# Title\nintro text\n## Section A\n" + " ".join(f"w{i}" for i in range(400))


def test_chunks_keep_section_metadata():
    chunks = chunk_document("doc.md", DOC, chunk_size=100, overlap=20)
    assert {c.section for c in chunks} == {"Title", "Section A"}
    assert all(c.source == "doc.md" for c in chunks)


def test_overlap_between_windows():
    chunks = [c for c in chunk_document("doc.md", DOC, 100, 20) if c.section == "Section A"]
    assert chunks[0].text.split()[-20:] == chunks[1].text.split()[:20]


def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_document("doc.md", DOC, chunk_size=50, overlap=50)


def test_sample_corpus_loads():
    assert len(build_chunks("data/docs")) > 10
