"""
api/main.py
-----------
FastAPI application entry point.

Run with:
    uvicorn api.main:app --reload
"""

from __future__ import annotations

# load_dotenv MUST come first — before any langchain/langsmith import so that
# LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, etc. are visible when LangChain
# initialises its global callback manager.
from dotenv import load_dotenv
load_dotenv()

import logging
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: load cached documents + log LangSmith tracing status
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: runs startup logic before yield, shutdown after."""

    # ── LangSmith tracing status ─────────────────────────────────────────
    tracing_on = os.getenv("LANGCHAIN_TRACING_V2", "").lower() in ("true", "1")
    project     = os.getenv("LANGCHAIN_PROJECT", "default")
    logger.info(
        "LangSmith tracing: %s (project=%s)",
        "ENABLED" if tracing_on else "DISABLED",
        project,
    )
    if tracing_on and not os.getenv("LANGCHAIN_API_KEY", "").strip():
        logger.warning(
            "LANGCHAIN_TRACING_V2=true but LANGCHAIN_API_KEY is not set — "
            "traces will NOT be sent."
        )

    # ── Load cached documents ─────────────────────────────────────────────
    logger.info("Loading cached documents on startup …")

    from app.cache import list_cached, load as cache_load, get_doc_handle, CACHE_DIR
    from api import session as sess

    entries = list_cached()
    loaded = 0

    for entry in entries:
        pkl_path = entry.get("path", "")
        if not pkl_path:
            continue

        try:
            with open(pkl_path, "rb") as f:
                doc = pickle.load(f)

            pdf_path = getattr(doc, "pdf_path", "")
            if not pdf_path or not os.path.isfile(pdf_path):
                doc_handle = Path(pkl_path).stem
                filename = Path(pdf_path).name if pdf_path else doc_handle

                if doc.mode == "pageindex" and doc.doc_id:
                    api_key = os.getenv("PAGEINDEX_API_KEY", "").strip()
                    if api_key:
                        try:
                            from pageindex import PageIndexClient  # type: ignore
                            doc.pageindex_client = PageIndexClient(api_key=api_key)
                        except Exception:
                            pass

                meta = {
                    "filename": filename,
                    "pages": doc.page_count,
                    "mode": doc.mode,
                    "pdf_path": pdf_path,
                }
                sess.register_doc(doc_handle, doc, meta)
                loaded += 1
            else:
                reloaded = cache_load(pdf_path)
                if reloaded is None:
                    continue
                doc_handle = get_doc_handle(pdf_path)
                meta = {
                    "filename": Path(pdf_path).name,
                    "pages": reloaded.page_count,
                    "mode": reloaded.mode,
                    "pdf_path": pdf_path,
                }
                sess.register_doc(doc_handle, reloaded, meta)
                loaded += 1
        except Exception as exc:
            logger.warning("Could not load cached entry %s: %s", pkl_path, exc)

    logger.info("Startup complete — %d cached document(s) loaded.", loaded)

    yield  # ── application runs ──


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PageIndex RAG Chat",
    description="FastAPI + LangGraph chatbot with PDF RAG capabilities.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# API routes (mounted under /api)
# ---------------------------------------------------------------------------

from api.routes.chat import router as chat_router
from api.routes.documents import router as documents_router

app.include_router(chat_router, prefix="/api")
app.include_router(documents_router, prefix="/api")

# ---------------------------------------------------------------------------
# Static files (frontend) — mounted last so API routes take priority
# ---------------------------------------------------------------------------

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    logger.warning("Frontend directory not found at %s — static serving disabled.", FRONTEND_DIR)
