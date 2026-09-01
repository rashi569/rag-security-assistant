import os
import tempfile
import streamlit as st

from ingest import ingest_file
from rag_chain import ask

st.set_page_config(page_title="RAG Security Assistant", page_icon="🔒", layout="wide")

st.title("🔒 Document Q&A Assistant with Security Guardrails")
st.caption(
    "Upload documents, ask questions. Answers are grounded only in what you upload. "
    "Prompt injection attempts - in your questions OR hidden in documents - are flagged, not silently followed."
)

with st.sidebar:
    st.header("📄 Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
    )

    if uploaded_files and st.button("Ingest documents"):
        for uploaded_file in uploaded_files:
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                result = ingest_file(tmp_path, display_name=uploaded_file.name)

            os.unlink(tmp_path)

            st.success(f"✅ {result['filename']}: {result['chunks_ingested']} chunks ingested")
            if result["flagged_chunks"]:
                st.warning(
                    f"⚠️ {len(result['flagged_chunks'])} chunk(s) in this document "
                    f"contained patterns resembling prompt injection attempts. "
                    f"They were still ingested (so you can inspect them) but flagged for retrieval-time warnings."
                )

    st.divider()
    st.caption(
        "This app runs fully locally: embeddings via sentence-transformers, "
        "LLM via Google Gemini API, vector store via ChromaDB. Requires GEMINI_API_KEY."
    )

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Ask a question about your uploaded documents...")

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        if entry["security"]["query_flagged"] or entry["security"]["context_flagged_sources"] or entry["security"]["response_leaked"]:
            with st.expander("⚠️ Security notices for this exchange"):
                sec = entry["security"]
                if sec["query_flagged"]:
                    st.write(f"- Your question matched injection patterns: {', '.join(sec['query_flag_reasons'])}")
                if sec["context_flagged_sources"]:
                    st.write(f"- Retrieved context included previously flagged document(s): {', '.join(sec['context_flagged_sources'])}")
                if sec["response_leaked"]:
                    st.write(f"- Response may have leaked internal instructions: {sec['response_leaked_phrases']}")
        if entry["sources"]:
            st.caption(f"Sources: {', '.join(set(entry['sources']))}")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = ask(question)
            except Exception as e:
                st.error(
                    f"Error generating answer: {e}\n\n"
                    "If this is an auth/connection error, make sure GEMINI_API_KEY is set "
                    "(e.g. `export GEMINI_API_KEY=your-key-here`, get one at "
                    "https://aistudio.google.com/apikey)."
                )
                st.stop()

        st.write(result["answer"])

        sec = result["security"]
        if sec["query_flagged"] or sec["context_flagged_sources"] or sec["response_leaked"]:
            with st.expander("⚠️ Security notices for this exchange", expanded=True):
                if sec["query_flagged"]:
                    st.write(f"- Your question matched injection patterns: {', '.join(sec['query_flag_reasons'])}")
                if sec["context_flagged_sources"]:
                    st.write(f"- Retrieved context included previously flagged document(s): {', '.join(sec['context_flagged_sources'])}")
                if sec["response_leaked"]:
                    st.write(f"- Response may have leaked internal instructions: {sec['response_leaked_phrases']}")

        if result["sources"]:
            st.caption(f"Sources: {', '.join(set(result['sources']))}")

    st.session_state.history.append({
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"],
        "security": sec,
    })
