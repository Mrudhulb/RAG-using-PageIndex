# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
# Primary entry point (FastAPI + chat UI)
uvicorn api.main:app --reload          # http://127.0.0.1:8000

# Legacy Streamlit UI (still functional but secondary)
python -m streamlit run streamlit_app.py
```

## Environment Variables

Copy `.env.example` to `.env`. Required:

| Variable | Required | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | Free key at openrouter.ai/keys |
| `PAGEINDEX_API_KEY` | No | Falls back to local BM25 if absent |
| `LLM_TEMPERATURE` | No | Default 0.2 |
| `LANGCHAIN_TRACING_V2` | No | Set `true` to enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | No | LangSmith API key (ls__...) |
| `LANGCHAIN_PROJECT` | No | Default `pageindex-rag` |
| `TAVILY_API_KEY` | No | Enables web search tool in agent mode |

Default model: `openai/gpt-oss-20b:free`. Free models change frequently — query `GET https://openrouter.ai/api/v1/models` filtered by `pricing.prompt == "0"` to get the current list.

## Architecture

Two entry points share the same `app/` core:

```
api/main.py          FastAPI server — mounts frontend/ as static root, loads cached docs on startup
api/session.py       In-memory session store: chat history (List[dict]) + active IngestedDocument per session_id
api/routes/chat.py   POST /api/chat, GET/DELETE /api/history
api/routes/documents.py  CRUD /api/documents + select/deselect active doc per session

app/graph.py         LangGraph ChatState workflow (see below)
app/ingestion.py     PDF → IngestedDocument (PageIndex cloud or BM25)
app/retrieval.py     Query → List[RetrievedChunk]
app/llm.py           ChatOpenAI pointed at openrouter.ai/api/v1; tiktoken_model_name="gpt-3.5-turbo" required for non-OpenAI model names
app/cache.py         Disk cache at .cache/<md5>.pkl; auto-loaded on startup; pageindex_client excluded from pickle

frontend/            Vanilla HTML/CSS/JS + marked.js (CDN) for markdown rendering
streamlit_app.py     Legacy UI — still wired to the same app/ core
```

## LangGraph Workflow (`app/graph.py`)

State: `ChatState(TypedDict)` — `query`, `chat_history`, `active_doc`, `model`, `route`, `retrieved_chunks`, `answer`, `error`, `messages` (Annotated with `operator.add`), `iterations`

```
START → classifier_node ──"general"──► general_node                          → END
                        ├─"document"─► retrieve_node → generate_node         → END
                        └─"agent"───► agent_node ↔ tool_node (ReAct, max 5)
                                                └─► agent_final_node         → END
```

- **classifier_node**: routes to "general", "document", or "agent". No `active_doc` + web query → "agent". `active_doc` set + doc question → "document". Default fallback when `active_doc` present → "document".
- **general_node**: builds messages from `chat_history` + `query`; falls back to merged HumanMessage if SystemMessage is rejected (e.g. Gemma).
- **retrieve_node**: calls `app/retrieval.retrieve(active_doc, query, top_k=5)`.
- **generate_node**: if `chunks[0].source == "pageindex_answer"` the PageIndex API already answered — skip the LLM call and return directly. Otherwise builds RAG prompt with chat history context.
- **agent_node**: LLM with bound tools; on first call builds messages from `chat_history` + `query`; on re-entry messages already include tool results.
- **tool_node**: `langgraph.prebuilt.ToolNode` — executes any tool calls emitted by `agent_node`.
- **agent_final_node**: walks `messages` in reverse to find the last AIMessage with no pending `tool_calls`; returns it as `answer`.

Public entry point: `run_chat_pipeline(query, chat_history, active_doc, model) -> dict`.

## Agent Tools (`app/tools.py`)

When `classifier_node` routes to "agent", a ReAct loop runs (max 5 iterations):

```
agent_node → tool_node → agent_node → ... → agent_final_node → END
```

Available tools (loaded based on env vars):
- `tavily_search` — requires `TAVILY_API_KEY`
- `browser_navigate`, `browser_get_text`, `browser_screenshot` — requires `playwright install chromium`

## LangSmith Tracing

Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` in `.env`.
LangChain/LangGraph instruments automatically — no code changes needed.
Project: `pageindex-rag` (set via `LANGCHAIN_PROJECT`).

## Retrieval Layer (`app/ingestion.py` + `app/retrieval.py`)

**PageIndex mode** (`PAGEINDEX_API_KEY` set):
- Ingestion: `submit_document()` → poll `get_tree()` until `status == "completed"` (max 90s). **Do not** use `is_retrieval_ready()` — it returns `False` on free-tier keys.
- Retrieval: `chat_completions(messages, doc_id=doc_id)` — PageIndex handles retrieval + generation internally. A synthetic `RetrievedChunk(source="pageindex_answer")` is inserted at index 0 to carry the answer through.
- BM25 over tree nodes is used only for source page display, not for the answer.

**BM25 mode** (no API key):
- `pypdf` extracts text page-by-page; falls back to `pdfplumber` if >50% pages are empty.
- `BM25Okapi` index built in memory; stored in `IngestedDocument.bm25_index`.

**`IngestedDocument`** key fields: `pdf_path`, `mode`, `pages: List[PageChunk]`, `doc_id` (PageIndex), `pageindex_client` (excluded from pickle), `bm25_index`, `tokenized_corpus`.

## Document Cache (`app/cache.py`)

- Cache key = MD5 of the PDF file bytes → `.cache/<hash>.pkl`
- `pageindex_client` is set to `None` before pickling and recreated from env on `load()`.
- `get_doc_handle(file_path)` returns the MD5 string used as `doc_handle` in all API responses.
- On FastAPI startup, all `.cache/*.pkl` files are loaded and registered into the session store so previously indexed PDFs are immediately available.
