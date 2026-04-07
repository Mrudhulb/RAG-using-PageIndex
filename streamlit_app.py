"""
streamlit_app.py
----------------
Streamlit UI for the LangGraph + PageIndex vectorless PDF Q&A app.

Run with:
    streamlit run streamlit_app.py

Features
--------
- Upload a PDF (stored to a temp file during the session)
- Displays ingestion mode (PageIndex cloud or local BM25) and page count
- Accept a natural-language question
- Displays the LLM-generated answer with cited source pages
- Shows the retrieved page excerpts in an expandable section
- Caches the ingested document in st.session_state to avoid re-indexing
  on every question
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

# Make sure `app/` is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

# Read keys once at module level so all blocks can access them
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
PAGEINDEX_KEY = os.getenv("PAGEINDEX_API_KEY", "").strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PageIndex RAG — PDF Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Sidebar — configuration & info
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Configuration")

    st.markdown(
        """
        **Required**
        - `OPENROUTER_API_KEY` in `.env` for answer generation.

        **Optional (but recommended)**
        - `PAGEINDEX_API_KEY` in `.env` for cloud-based vectorless retrieval.
        - Without it the app uses local **BM25** retrieval automatically.
        """
    )

    # Live mode indicator
    retrieval_mode = "PageIndex Cloud" if PAGEINDEX_KEY else "Local BM25"
    st.markdown(f"**Retrieval mode:** `{retrieval_mode}`")

    if OPENROUTER_KEY:
        st.success("OpenRouter API key detected ✓")
    else:
        st.error("OPENROUTER_API_KEY not set — answers will fail.")

    st.divider()

    FREE_MODELS = [
        "meta-llama/llama-3.2-3b-instruct:free",
        "meta-llama/llama-3.1-8b-instruct:free",
        "google/gemma-3-4b-it:free",
        "google/gemma-3-12b-it:free",
        "microsoft/phi-3-mini-128k-instruct:free",
        "qwen/qwen-2.5-7b-instruct:free",
        "deepseek/deepseek-r1:free",
    ]

    selected_model = st.selectbox(
        "Free model (OpenRouter)",
        options=FREE_MODELS,
        index=0,
        help="All models listed are free-tier on OpenRouter.",
    )

    st.divider()
    top_k = st.slider(
        "Top-K pages to retrieve",
        min_value=1,
        max_value=15,
        value=int(os.getenv("TOP_K", "5")),
        help="How many page chunks to pass to the LLM as context.",
    )

    st.divider()

    # Local cache manager
    st.markdown("**Local document cache**")
    from app.cache import list_cached, CACHE_DIR
    cached = list_cached()
    if cached:
        st.caption(f"{len(cached)} document(s) cached in `{CACHE_DIR}`")
        for entry in cached:
            st.caption(f"• {entry['file'][:12]}… | {entry['mode']} | {entry['pages']} pages | {entry['size_kb']} KB")
        if st.button("Clear all cache"):
            import shutil
            shutil.rmtree(str(CACHE_DIR), ignore_errors=True)
            st.success("Cache cleared.")
            st.rerun()
    else:
        st.caption("No cached documents yet.")

    st.divider()
    st.caption("Powered by LangGraph · PageIndex · LangChain · OpenRouter")


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "ingested_doc" not in st.session_state:
    st.session_state.ingested_doc = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of (question, answer, chunks)


# ---------------------------------------------------------------------------
# Main — title
# ---------------------------------------------------------------------------

st.title("📄 PDF Q&A with PageIndex + LangGraph")
st.markdown(
    "Upload a PDF, then ask any question about it. "
    "The app retrieves the most relevant pages and generates a grounded answer."
)

st.divider()


# ---------------------------------------------------------------------------
# Step 1 — PDF upload
# ---------------------------------------------------------------------------

st.subheader("Step 1 — Upload a PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    help="The PDF will be indexed once; you can ask multiple questions without re-uploading.",
)

if uploaded_file is not None:
    # Check if this is a new file
    new_file = uploaded_file.name != st.session_state.pdf_name

    if new_file:
        st.session_state.ingested_doc = None
        st.session_state.chat_history = []

        # Save to a temp file that persists for the session
        suffix = Path(uploaded_file.name).suffix or ".pdf"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded_file.read())
        tmp.flush()
        tmp.close()

        st.session_state.pdf_path = tmp.name
        st.session_state.pdf_name = uploaded_file.name

    # Ingest (only if not already done for this file)
    if st.session_state.ingested_doc is None:
        from app.cache import load as cache_load, save as cache_save

        cached_doc = cache_load(st.session_state.pdf_path)
        if cached_doc is not None:
            st.session_state.ingested_doc = cached_doc
            st.success(
                f"⚡ Loaded **{uploaded_file.name}** from local cache — "
                f"**{cached_doc.page_count} pages** — mode: `{cached_doc.mode}`"
            )
        else:
            with st.spinner(
                f"Indexing **{uploaded_file.name}** … "
                f"({'uploading to PageIndex cloud' if PAGEINDEX_KEY else 'building BM25 index locally'})"
            ):
                try:
                    from app.ingestion import ingest_pdf
                    doc = ingest_pdf(st.session_state.pdf_path)
                    cache_save(st.session_state.pdf_path, doc)
                    st.session_state.ingested_doc = doc
                    st.success(
                        f"✅ Indexed **{uploaded_file.name}** — "
                        f"**{doc.page_count} pages** — mode: `{doc.mode}` — saved to local cache"
                    )
                except Exception as exc:
                    st.error(f"❌ Ingestion failed: {exc}")
                    logger.exception("Ingestion failed")
    else:
        doc = st.session_state.ingested_doc
        st.info(
            f"📌 **{st.session_state.pdf_name}** already indexed — "
            f"**{doc.page_count} pages** — mode: `{doc.mode}`"
        )


# ---------------------------------------------------------------------------
# Step 2 — Ask a question
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Step 2 — Ask a Question")

if st.session_state.ingested_doc is None:
    st.warning("Please upload and index a PDF first.")
else:
    with st.form(key="qa_form", clear_on_submit=False):
        question = st.text_area(
            "Your question",
            placeholder="e.g. What are the main findings of this document?",
            height=100,
        )
        submit_btn = st.form_submit_button("Get Answer", type="primary")

    if submit_btn:
        if not question.strip():
            st.warning("Please enter a question.")
        elif not OPENROUTER_KEY:
            st.error("OPENROUTER_API_KEY is not set. Cannot generate an answer.")
        else:
            with st.spinner("Retrieving relevant pages and generating answer …"):
                try:
                    from app.graph import run_rag_pipeline

                    result = run_rag_pipeline(
                        pdf_path=st.session_state.pdf_path,
                        query=question,
                        top_k=top_k,
                        preloaded_doc=st.session_state.ingested_doc,
                        model=selected_model,
                    )

                    answer = result.get("answer", "")
                    chunks = result.get("retrieved_chunks") or []
                    error = result.get("error", "")

                    # Store in chat history
                    st.session_state.chat_history.append(
                        (question, answer, chunks, error)
                    )

                except Exception as exc:
                    logger.exception("Pipeline error")
                    st.error(f"Pipeline error: {exc}")


# ---------------------------------------------------------------------------
# Step 3 — Display answers (newest first)
# ---------------------------------------------------------------------------

if st.session_state.chat_history:
    st.divider()
    st.subheader("Step 3 — Answers")

    for i, entry in enumerate(reversed(st.session_state.chat_history)):
        question, answer, chunks, error = entry
        turn_number = len(st.session_state.chat_history) - i

        with st.container():
            st.markdown(f"**Q{turn_number}: {question}**")

            if error and not answer:
                st.error(f"Error: {error}")
            else:
                st.markdown(answer)

                if error:
                    st.warning(f"Note: {error}")

            # Source pages expander
            display_chunks = [c for c in chunks if c.source != "pageindex_answer"]
            if display_chunks:
                chunks = display_chunks
            if chunks:
                page_nums = sorted({c.page_number for c in chunks if c.page_number > 0})
                page_label = ", ".join(f"p.{n}" for n in page_nums)
                with st.expander(
                    f"📑 Source pages ({page_label}) — {len(chunks)} chunk(s) retrieved"
                ):
                    for chunk in sorted(chunks, key=lambda c: c.page_number):
                        score_str = f"{chunk.score:.4f}" if chunk.score else "N/A"
                        st.markdown(
                            f"**Page {chunk.page_number}** "
                            f"*(score: {score_str}, source: {chunk.source})*"
                        )
                        excerpt = chunk.text[:800] + ("…" if len(chunk.text) > 800 else "")
                        st.text_area(
                            label=f"page_{chunk.page_number}_q{turn_number}",
                            value=excerpt,
                            height=150,
                            label_visibility="collapsed",
                            key=f"chunk_{turn_number}_{chunk.page_number}_{i}",
                        )

            st.divider()


# ---------------------------------------------------------------------------
# Clear history button
# ---------------------------------------------------------------------------

if st.session_state.chat_history:
    if st.button("🗑️ Clear conversation history"):
        st.session_state.chat_history = []
        st.rerun()
