"""
api/routes/chat.py
------------------
Chat endpoints: POST /api/chat, GET /api/history, DELETE /api/history
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api import session as sess
from app.graph import run_chat_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str
    model: str = "openai/gpt-oss-20b:free"
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    route: str
    sources: list
    session_id: str


# ---------------------------------------------------------------------------
# POST /api/chat
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Run the chat pipeline and return an answer."""
    session_id = req.session_id or ""
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required.")

    history = sess.get_history(session_id)
    active_doc = sess.get_active_doc(session_id)

    try:
        result = run_chat_pipeline(
            query=req.query,
            chat_history=history,
            active_doc=active_doc,
            model=req.model,
        )
    except Exception as exc:
        logger.exception("Chat pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    # Surface errors as the answer so the UI shows them
    answer = result["answer"]
    if not answer and result.get("error"):
        answer = f"⚠️ {result['error']}"

    # Persist messages
    sess.add_message(session_id, "user", req.query)
    sess.add_message(session_id, "assistant", answer)

    # Serialise sources
    sources = []
    for chunk in result.get("retrieved_chunks") or []:
        if getattr(chunk, "source", "") == "pageindex_answer":
            continue
        sources.append({
            "page_number": chunk.page_number,
            "text": chunk.text[:400],
            "score": round(chunk.score, 4),
            "source": chunk.source,
        })

    return ChatResponse(
        answer=answer,
        route=result.get("route", "general"),
        sources=sources,
        session_id=session_id,
    )


# ---------------------------------------------------------------------------
# GET /api/history
# ---------------------------------------------------------------------------

@router.get("/history")
async def get_history(session_id: str):
    """Return chat history for a session."""
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required.")
    history = sess.get_history(session_id)
    return {"history": history, "session_id": session_id}


# ---------------------------------------------------------------------------
# DELETE /api/history
# ---------------------------------------------------------------------------

@router.delete("/history")
async def clear_history(session_id: str):
    """Clear chat history for a session."""
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required.")
    sess.clear_history(session_id)
    return {"status": "cleared", "session_id": session_id}
