"""
api/main.py
-----------
FastAPI application entry point.

Run with:
    uvicorn api.main:app --reload
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PageIndex RAG Chat",
    description="FastAPI + LangGraph chatbot with PDF RAG capabilities.",
    version="1.0.0",
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
# Startup: load cached documents
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Load all previously cached documents into the session store."""
    logger.info("Loading cached documents on startup …")

    from app.cache import list_cached, load as cache_load, get_doc_handle, CACHE_DIR
    from api import session as sess

    entries = list_cached()
    loaded = 0

    for entry in entries:
        pkl_path = entry.get("path", "")
        if not pkl_path:
            continue

        # Try to find the original pdf_path from the pickled doc
        try:
            import pickle
            with open(pkl_path, "rb") as f:
                doc = pickle.load(f)

            pdf_path = getattr(doc, "pdf_path", "")
            if not pdf_path or not os.path.isfile(pdf_path):
                # pdf_path is stored in doc; reload via cache_load using pkl stem as handle
                # We can register with the handle derived from the pkl filename
                doc_handle = Path(pkl_path).stem  # the md5 hash IS the stem
                filename = Path(pdf_path).name if pdf_path else doc_handle

                # Recreate live client for pageindex mode
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


# ---------------------------------------------------------------------------
# Static files (frontend) — mounted last so API routes take priority
# ---------------------------------------------------------------------------

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    logger.warning("Frontend directory not found at %s — static serving disabled.", FRONTEND_DIR)
