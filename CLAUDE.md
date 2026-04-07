# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app (opens http://localhost:8501)
streamlit run streamlit_app.py
```

## Environment Configuration (`.env`)

| Variable | Required | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | LLM inference via OpenRouter |
| `PAGEINDEX_API_KEY` | No | Cloud retrieval (falls back to local BM25 if absent) |
| `OPENROUTER_MODEL` | No | Defaults to a free model (e.g., Mistral 7B) |
| `LLM_TEMPERATURE` | No | Defaults to 0.2 |

## Architecture

The app is a PDF Q&A (RAG) system with **no vector embeddings**. Retrieval is tree/keyword-based.

### Layer Overview

```
streamlit_app.py         ← UI entry point, session state management
app/
  graph.py               ← LangGraph workflow (ingest → retrieve → generate)
  ingestion.py           ← PDF loading; routes to PageIndex cloud or BM25
  retrieval.py           ← Query routing; PageIndex API polling or BM25 scoring
  llm.py                 ← OpenRouter ChatOpenAI setup and RAG prompt construction
```

### LangGraph Workflow (`app/graph.py`)

Three nodes operate on a shared `RAGState` TypedDict:

1. **`ingest_node`** — loads PDF and builds an index (`IngestedDocument`)
2. **`retrieve_node`** — fetches top-K relevant chunks (`List[RetrievedChunk]`)
3. **`generate_node`** — calls LLM with retrieved context to produce the answer

`run_rag_pipeline()` accepts a pre-built `IngestedDocument` to skip re-ingestion on follow-up questions within the same session (cached in `st.session_state`).

### Retrieval Modes (`app/ingestion.py` + `app/retrieval.py`)

**PageIndex Cloud** (when `PAGEINDEX_API_KEY` is set):
- Uploads PDF via `submit_document()`, polls `is_retrieval_ready()` (up to 5 min)
- Queries via `submit_query()` / `get_retrieval()` polling (up to 2 min)
- Returns tree-structured nodes flattened to page-level `RetrievedChunk` objects
- Falls back to local BM25 if the API returns no nodes

**Local BM25** (fallback, fully offline):
- Extracts text page-by-page with `pypdf` (falls back to `pdfplumber` for scanned PDFs)
- Builds an in-memory `BM25Okapi` index; no network calls required

### Key Data Structures

```python
# IngestedDocument (app/ingestion.py)
pdf_path: str
mode: str          # 'pageindex' | 'bm25'
pages: List[PageChunk]
doc_id: Optional[str]               # set in PageIndex mode
bm25_index: Optional[BM25Okapi]     # set in BM25 mode
tokenized_corpus: Optional[...]     # set in BM25 mode

# RetrievedChunk (app/retrieval.py)
page_number: int   # 1-based
text: str
score: float
source: str        # 'pageindex' | 'bm25' | 'bm25_fallback'
```

### LLM Layer (`app/llm.py`)

Uses `ChatOpenAI` pointed at `https://openrouter.ai/api/v1`. The RAG prompt instructs the model to answer strictly from retrieved context and cite page numbers. Free-tier models include Mistral 7B, Llama 3 8B, Gemma 3, Qwen 3, and others selectable in the UI sidebar.
