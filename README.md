# PageIndex RAG Chat

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about your PDF documents — with **no vector embeddings** required. Built with [LangGraph](https://github.com/langchain-ai/langgraph), [PageIndex](https://pageindex.ai), and [FastAPI](https://fastapi.tiangolo.com), served through a clean chat UI.

---

## Features

- **Vectorless RAG** — uses PageIndex tree-based retrieval or local BM25 (no embeddings, no vector DB)
- **Smart 3-way routing** — LangGraph automatically classifies each question as *general*, *document*, or *agent (web search)*
- **Web search agent** — ReAct loop with Tavily search + Playwright browser tools for real-time information
- **General chatbot** — works as a full assistant even without a PDF loaded
- **Persistent document memory** — previously indexed PDFs reload automatically on server restart
- **Chat history** — full conversation context passed to the LLM on every turn
- **Free LLM inference** — powered by OpenRouter free-tier models
- **LangSmith tracing** — full observability of every graph run out of the box
- **FastAPI backend** — clean REST API with auto-reload dev server
- **Chat UI** — bubble-style interface with markdown rendering, source page viewer, and document manager

---

## Architecture

```
Browser (Chat UI)
  └── HTTP REST
      └── FastAPI Server (api/)
              └── LangGraph Workflow (app/graph.py)
                      ├── classifier_node → general_node                          (direct LLM answer)
                      ├── classifier_node → retrieve_node → generate_node         (RAG)
                      └── classifier_node → agent_node ↔ tool_node (ReAct loop)
                                                       └── agent_final_node       (web answer)
                                                  PageIndex Cloud | Local BM25
```

### LangGraph Nodes

| Node | Role |
|---|---|
| `classifier_node` | LLM decides: *general*, *document*, or *agent* |
| `general_node` | Answers with full chat history, no retrieval |
| `retrieve_node` | Fetches top-K relevant pages from the active document |
| `generate_node` | Builds answer from retrieved context + chat history |
| `agent_node` | ReAct LLM step — decides whether to call tools |
| `tool_node` | Executes tool calls (Tavily search, Playwright browser) |
| `agent_final_node` | Extracts the final answer from the agent message history |

### Retrieval Modes

| Mode | When | How |
|---|---|---|
| **PageIndex Cloud** | `PAGEINDEX_API_KEY` set | Tree index via `chat_completions` API |
| **Local BM25** | No API key | `pypdf`/`pdfplumber` + `rank-bm25` scoring |

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/Mrudhulb/RAG-using-PageIndex.git
cd RAG-using-PageIndex
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Playwright browser (for web browsing tools)

```bash
playwright install chromium
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your keys:

```env
# Required — get a free key at https://openrouter.ai/keys
OPENROUTER_API_KEY=sk-or-...

# Optional — get a free key at https://pageindex.ai
# Without this, the app uses local BM25 retrieval automatically
PAGEINDEX_API_KEY=...

# Optional — enables web search in the agent route
# Get a free key at https://tavily.com
TAVILY_API_KEY=tvly-...

# Optional — enables LangSmith tracing at smith.langchain.com
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=pageindex-rag

# LLM temperature (default: 0.2)
LLM_TEMPERATURE=0.2
```

### 5. Run

```bash
uvicorn api.main:app --reload
```

Open **http://127.0.0.1:8000** in your browser.

---

## Usage

1. **General chat** — type any question without a PDF. The chatbot answers from its training knowledge with full conversation memory.

2. **PDF Q&A**
   - Click **Upload PDF** in the sidebar
   - Wait for indexing to complete
   - Click **Select** on the document
   - Ask questions — automatically routed to document retrieval or general knowledge

3. **Web search** — ask anything requiring real-time information (e.g. *"What are the latest AI news?"*). The agent automatically uses Tavily search and Playwright to browse the web.

4. **Multiple documents** — upload several PDFs, select whichever is relevant per question

5. **Persistent documents** — restart the server; previously indexed PDFs reappear automatically

---

## Project Structure

```
├── api/
│   ├── main.py              # FastAPI app, startup loader, static mount
│   ├── session.py           # In-memory session store (chat history + active doc)
│   └── routes/
│       ├── chat.py          # POST /api/chat, GET/DELETE /api/history
│       └── documents.py     # CRUD /api/documents
├── app/
│   ├── graph.py             # LangGraph workflow (3-way routing + ReAct agent loop)
│   ├── tools.py             # Agent tools: Tavily search + Playwright browser
│   ├── ingestion.py         # PDF ingestion (PageIndex or BM25)
│   ├── retrieval.py         # Retrieval logic
│   ├── llm.py               # OpenRouter LLM setup
│   └── cache.py             # Disk cache for indexed documents
├── frontend/
│   ├── index.html           # Chat UI
│   ├── style.css            # Styles
│   └── app.js               # Frontend logic + markdown rendering
├── .env.example             # Environment variable template
├── requirements.txt
└── README.md
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chat` | Send a message, get an answer |
| `GET` | `/api/history` | Get session chat history |
| `DELETE` | `/api/history` | Clear session chat history |
| `POST` | `/api/documents` | Upload and index a PDF |
| `GET` | `/api/documents` | List all cached documents |
| `DELETE` | `/api/documents/{handle}` | Remove a document |
| `POST` | `/api/documents/{handle}/select` | Set active document for session |
| `POST` | `/api/documents/deselect` | Clear active document |

**Chat request/response:**

```json
// POST /api/chat
{ "query": "What are the key findings?", "model": "openai/gpt-oss-20b:free", "session_id": "uuid" }

// Response
{ "answer": "...", "route": "document", "sources": [...], "session_id": "uuid" }
```

---

## Free Models (OpenRouter)

| Model | Notes |
|---|---|
| `openai/gpt-oss-20b:free` | Default — fast, reliable |
| `openai/gpt-oss-120b:free` | Higher quality |
| `meta-llama/llama-3.3-70b-instruct:free` | Strong reasoning |
| `google/gemma-3-27b-it:free` | Google flagship free model |
| `nvidia/nemotron-3-super-120b-a12b:free` | High quality, currently active |

> Free model availability changes frequently. Visit [openrouter.ai/models](https://openrouter.ai/models?q=:free) for the current list.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| Workflow | LangGraph |
| LLM | LangChain + OpenRouter (free models) |
| Retrieval | PageIndex cloud API or rank-bm25 |
| PDF Parsing | pypdf + pdfplumber |
| Web Search | Tavily |
| Browser Automation | Playwright (headless Chromium) |
| Observability | LangSmith |
| Frontend | Vanilla HTML/CSS/JS + marked.js |

---

## License

MIT
