import logging
from pathlib import Path

import streamlit as st

from config import DATA_DIR
from rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="RAG QA System", layout="wide")

# --- Initialise session state ---
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()
if "history" not in st.session_state:
    st.session_state.history = []
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# --- Sidebar ---
st.sidebar.title("Document Ingestion")

uploaded_files = st.sidebar.file_uploader(
    "Upload documents",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
)

reset_store = st.sidebar.checkbox("Reset vector store before ingesting", value=True)

if st.sidebar.button("Ingest documents", type="primary"):
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one file first.")
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        if reset_store:
            uploaded_names = {f.name for f in uploaded_files}
            for old_file in DATA_DIR.iterdir():
                if old_file.is_file() and old_file.name not in uploaded_names:
                    old_file.unlink()

        saved = []
        for f in uploaded_files:
            dest = DATA_DIR / f.name
            dest.write_bytes(f.getvalue())
            saved.append(f.name)

        with st.spinner("Ingesting documents ..."):
            count = st.session_state.pipeline.ingest(
                str(DATA_DIR), reset=reset_store
            )
        st.session_state.chunk_count = count
        st.sidebar.success(f"Ingested {len(saved)} file(s) — {count} chunk(s) indexed.")

st.sidebar.divider()
st.sidebar.metric("Indexed chunks", st.session_state.chunk_count)

# --- Main area ---
st.title("RAG QA System")

with st.form("qa_form"):
    cols = st.columns([4, 1])
    question = cols[0].text_input(
        "Ask a question about your documents", label_visibility="collapsed"
    )
    ask = cols[1].form_submit_button("Ask", type="primary", use_container_width=True)

if ask and question.strip():
    with st.spinner("Searching and generating answer ..."):
        result = st.session_state.pipeline.query(question.strip())
    st.session_state.history.append({"question": question.strip(), **result})

# Display history
for i, entry in enumerate(reversed(st.session_state.history)):
    with st.container():
        st.markdown(f"**Q{i+1}:** {entry['question']}")
        st.markdown(f"**Answer:** {entry['answer']}")
        if entry["sources"]:
            st.markdown(f"**Sources:** {', '.join(entry['sources'])}")

        with st.expander(f"Retrieved chunks ({len(entry['retrieved_chunks'])} raw)"):
            for j, chunk in enumerate(entry["retrieved_chunks"]):
                st.markdown(f"**Chunk {j+1}**  —  `{chunk['chunk_id']}`")
                st.markdown(f"- Source: {chunk['source']}  |  Page: {chunk['page']}  |  Distance: {chunk['distance']:.4f}")
                st.text(chunk["text"][:500])
                st.divider()
        st.divider()

if not st.session_state.history:
    st.info("Upload documents in the sidebar and ingest them, then ask a question.")
