"""
api/session.py
--------------
In-memory session store for the FastAPI chatbot.

Sessions expire after 2 hours of inactivity.
Documents are registered globally by doc_handle (md5 of the original pdf path).
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Any
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SESSION_TTL_SECONDS = 2 * 60 * 60  # 2 hours

# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------

# session_id -> {"history": List[dict], "active_doc_id": str|None, "last_active": float}
_sessions: Dict[str, dict] = {}

# doc_handle -> IngestedDocument
_docs: Dict[str, Any] = {}

# doc_handle -> metadata dict
_doc_meta: Dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _now() -> float:
    return time.time()


def _evict_expired() -> None:
    """Remove sessions that have been inactive for longer than SESSION_TTL_SECONDS."""
    cutoff = _now() - SESSION_TTL_SECONDS
    expired = [sid for sid, s in _sessions.items() if s["last_active"] < cutoff]
    for sid in expired:
        del _sessions[sid]
        logger.debug("Evicted expired session: %s", sid)


def get_or_create_session(session_id: str) -> dict:
    """Return existing session or create a new one. Touches last_active."""
    _evict_expired()
    if session_id not in _sessions:
        _sessions[session_id] = {
            "history": [],
            "active_doc_id": None,
            "last_active": _now(),
        }
        logger.info("Created new session: %s", session_id)
    else:
        _sessions[session_id]["last_active"] = _now()
    return _sessions[session_id]


def add_message(session_id: str, role: str, content: str) -> None:
    """Append a message to the session's chat history."""
    session = get_or_create_session(session_id)
    session["history"].append({"role": role, "content": content})


def get_history(session_id: str) -> List[dict]:
    """Return the chat history for a session."""
    session = get_or_create_session(session_id)
    return session["history"]


def clear_history(session_id: str) -> None:
    """Clear chat history for a session (keeps active_doc)."""
    session = get_or_create_session(session_id)
    session["history"] = []
    logger.info("Cleared history for session: %s", session_id)


def set_active_doc(session_id: str, doc_handle: str) -> None:
    """Set the active document for a session."""
    session = get_or_create_session(session_id)
    session["active_doc_id"] = doc_handle
    logger.info("Session %s → active_doc_id=%s", session_id, doc_handle)


def clear_active_doc(session_id: str) -> None:
    """Remove the active document from a session (back to general chat)."""
    session = get_or_create_session(session_id)
    session["active_doc_id"] = None
    logger.info("Session %s active_doc cleared.", session_id)


def get_active_doc(session_id: str):
    """Return the IngestedDocument for the session's active doc, or None."""
    session = get_or_create_session(session_id)
    doc_id = session.get("active_doc_id")
    if not doc_id:
        return None
    return _docs.get(doc_id)


# ---------------------------------------------------------------------------
# Document registry
# ---------------------------------------------------------------------------

def register_doc(doc_handle: str, doc, meta: Optional[dict] = None) -> None:
    """
    Register an IngestedDocument in the global doc store.

    Parameters
    ----------
    doc_handle: md5 hash of the original pdf path.
    doc:        IngestedDocument instance.
    meta:       Optional metadata dict (filename, pages, mode, etc.).
    """
    _docs[doc_handle] = doc
    _doc_meta[doc_handle] = meta or {}
    logger.info("Registered doc_handle=%s", doc_handle)


def get_doc(doc_handle: str):
    """Return the IngestedDocument for doc_handle, or None."""
    return _docs.get(doc_handle)


def remove_doc(doc_handle: str) -> None:
    """Remove a document from the registry."""
    _docs.pop(doc_handle, None)
    _doc_meta.pop(doc_handle, None)
    # Remove this doc from any sessions that have it active
    for session in _sessions.values():
        if session.get("active_doc_id") == doc_handle:
            session["active_doc_id"] = None
    logger.info("Removed doc_handle=%s from registry.", doc_handle)


def list_docs() -> List[dict]:
    """Return metadata for all registered documents."""
    result = []
    for doc_handle, doc in _docs.items():
        meta = _doc_meta.get(doc_handle, {})
        result.append({
            "doc_handle": doc_handle,
            "filename": meta.get("filename", doc_handle),
            "pages": meta.get("pages", getattr(doc, "page_count", 0)),
            "mode": meta.get("mode", getattr(doc, "mode", "unknown")),
            "pdf_path": meta.get("pdf_path", getattr(doc, "pdf_path", "")),
        })
    return result
