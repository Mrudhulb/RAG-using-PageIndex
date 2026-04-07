"""
api/routes/documents.py
-----------------------
Document management endpoints.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from api import session as sess
from app.cache import get_doc_handle, load as cache_load, save as cache_save, delete as cache_delete
from app.ingestion import ingest_pdf

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_DIR = Path(__file__).parent.parent.parent / ".cache" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class SelectDocRequest(BaseModel):
    session_id: str


# ---------------------------------------------------------------------------
# POST /api/documents — upload and ingest a PDF
# ---------------------------------------------------------------------------

@router.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    """Accept a PDF upload, ingest it, cache it, and register in the session store."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    dest_path = UPLOAD_DIR / file.filename
    try:
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}")

    pdf_path = str(dest_path)

    # Check disk cache first
    doc = cache_load(pdf_path)
    if doc is None:
        try:
            doc = ingest_pdf(pdf_path)
            cache_save(pdf_path, doc)
        except Exception as exc:
            logger.exception("Ingestion failed for %s: %s", pdf_path, exc)
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

    doc_handle = get_doc_handle(pdf_path)
    meta = {
        "filename": file.filename,
        "pages": doc.page_count,
        "mode": doc.mode,
        "pdf_path": pdf_path,
    }
    sess.register_doc(doc_handle, doc, meta)

    logger.info("Uploaded and registered doc_handle=%s (%s)", doc_handle, file.filename)
    return {
        "doc_handle": doc_handle,
        "filename": file.filename,
        "pages": doc.page_count,
        "mode": doc.mode,
    }


# ---------------------------------------------------------------------------
# GET /api/documents
# ---------------------------------------------------------------------------

@router.get("/documents")
async def list_documents():
    """Return all registered/cached documents with metadata."""
    return {"documents": sess.list_docs()}


# ---------------------------------------------------------------------------
# DELETE /api/documents/{doc_handle}
# ---------------------------------------------------------------------------

@router.delete("/documents/{doc_handle}")
async def delete_document(doc_handle: str):
    """Remove a document from cache and the session store."""
    doc = sess.get_doc(doc_handle)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    pdf_path = getattr(doc, "pdf_path", "")
    if pdf_path:
        cache_delete(pdf_path)
        # Remove the uploaded file if it lives in our uploads dir
        try:
            upload_file = UPLOAD_DIR / Path(pdf_path).name
            if upload_file.exists():
                upload_file.unlink()
        except Exception:
            pass

    sess.remove_doc(doc_handle)
    return {"status": "deleted", "doc_handle": doc_handle}


# ---------------------------------------------------------------------------
# POST /api/documents/{doc_handle}/select
# ---------------------------------------------------------------------------

@router.post("/documents/{doc_handle}/select")
async def select_document(doc_handle: str, body: SelectDocRequest):
    """Set the active document for a session."""
    if sess.get_doc(doc_handle) is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    sess.set_active_doc(body.session_id, doc_handle)
    return {"status": "selected", "doc_handle": doc_handle, "session_id": body.session_id}


# ---------------------------------------------------------------------------
# POST /api/documents/deselect
# ---------------------------------------------------------------------------

@router.post("/documents/deselect")
async def deselect_document(body: SelectDocRequest):
    """Clear the active document for a session (back to general chat)."""
    sess.clear_active_doc(body.session_id)
    return {"status": "deselected", "session_id": body.session_id}
