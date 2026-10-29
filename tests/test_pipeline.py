import pytest
from fastapi.testclient import TestClient

from rag_assistant.evaluate import evaluate, load_eval_set
from rag_assistant.generator import NO_ANSWER
from rag_assistant.pipeline import RAGPipeline


@pytest.fixture(scope="module")
def pipeline():
    return RAGPipeline(use_openai=False)


def test_answer_is_cited(pipeline):
    r = pipeline.ask("What is the maximum loan-to-value for an uninsured mortgage?")
    assert r["grounded"] and "80 percent" in r["answer"]
    assert r["citations"][0]["source"] == "mortgage_underwriting_policy.md"


def test_out_of_scope_is_refused(pipeline):
    r = pipeline.ask("What is the dress code for the downtown office?")
    assert r["answer"] == NO_ANSWER and not r["grounded"]


def test_eval_quality_gate(pipeline):
    report = evaluate(pipeline, load_eval_set())
    assert report["hit@4"] >= 0.9
    assert report["mrr"] >= 0.8
    assert report["refusal_accuracy"] >= 0.5


def test_api():
    from api.main import app
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/ask", json={"question": "How long are call recordings retained?"})
    assert r.status_code == 200 and "two years" in r.json()["answer"]
