# ADR-010: Global Document Registry with Session-Scoped Active Selection

**Status:** Accepted  
**Date:** 2026-04-11  
**Deciders:** Project team

---

## Context

Multiple browser sessions can use the application simultaneously. Each session needs to independently select which document it's querying. Documents are expensive to ingest (PageIndex: 10–90s; BM25: full PDF parse), so re-ingesting per session is unacceptable.

## Decision

Two-tier document management:

1. **Global registry** (`_docs: Dict[str, Any]`, `_doc_meta: Dict[str, dict]` in `api/session.py`): All ingested documents are registered once, process-wide, keyed by `doc_handle` (content MD5). Any session can reference any registered doc.

2. **Session-scoped active selection** (`session["active_doc_id"]`): Each session stores a pointer to its currently selected `doc_handle`. `get_active_doc(session_id)` resolves the pointer to the `IngestedDocument` object.

Startup loader (`api/main.py` lifespan) re-populates the global registry from `.cache/*.pkl` on process start.

## Consequences

**Positive:**
- One ingestion per unique document, shared across all sessions
- Sessions can switch documents freely without re-ingestion
- Startup restore means documents survive server restarts

**Negative:**
- **No tenant isolation** — all sessions can access all documents (no per-user ownership)
- If a doc is deleted (`DELETE /api/documents/{handle}`), all sessions pointing to it silently lose their active doc (cleared in `remove_doc()`)
- Global dict is not thread-safe under concurrent writes without locking (CPython GIL provides partial protection for simple dict ops, but not for read-modify-write sequences)
- In multi-replica deployments, each replica has an independent registry — a document uploaded to replica A is invisible to replica B

**Scaling path:**
- Move document registry to a shared store (Redis, database) keyed by `doc_handle`
- Add per-user ownership/access control layer
