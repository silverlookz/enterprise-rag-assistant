"""Streamlit chat UI.  Run:  streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag_assistant.pipeline import RAGPipeline  # noqa: E402

st.set_page_config(page_title="Enterprise RAG Assistant", page_icon="📚", layout="wide")


@st.cache_resource
def load():
    return RAGPipeline()


pipeline = load()
st.title("📚 Enterprise RAG Assistant")
st.caption(f"{len(pipeline.chunks)} chunks indexed · retrieval: bm25+{pipeline.retriever.dense_backend}")

examples = ["What is the maximum loan-to-value for an uninsured mortgage?",
            "When must a large cash transaction be reported?",
            "Can restricted client data be sent to external AI services?"]
with st.sidebar:
    st.subheader("Try a question")
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["q"] = ex

question = st.chat_input("Ask about the policy documents") or st.session_state.pop("q", None)
if question:
    st.chat_message("user").write(question)
    result = pipeline.ask(question)
    with st.chat_message("assistant"):
        st.write(result["answer"])
        if result["citations"]:
            with st.expander("Sources"):
                for c in result["citations"]:
                    st.markdown(f"**[{c['id']}]** `{c['source']}` — {c['section']} (score {c['score']})")
        st.caption(f"mode: {result['mode']} · {result['latency_ms']} ms")
