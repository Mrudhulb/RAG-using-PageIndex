# ADR-003: FastAPI + Vanilla JS Frontend over Streamlit

**Status:** Accepted  
**Date:** 2026-04-11  
**Deciders:** Project team

---

## Context

The application was originally built with Streamlit as the UI layer. Streamlit bundles server-side state and UI rendering in a single Python process, which caused friction with multi-session chat management, custom component styling, and clean API separation for potential future clients.

## Decision

Replace Streamlit with:
- **FastAPI** as the HTTP server with explicit REST endpoints
- **Vanilla HTML/CSS/JS** (+ marked.js for markdown rendering) served as static files from `frontend/`
- Static files mounted at `/` after API routes, so `GET /` serves `index.html`

## Consequences

**Positive:**
- REST API-first — decoupled frontend; any client (mobile, CLI, other frontends) can consume `/api/*`
- Session management is explicit (`session_id` UUID passed per request)
- FastAPI auto-generates OpenAPI docs at `/docs`
- CORS middleware gives flexibility for cross-origin clients
- Clean uvicorn/ASGI stack — no hidden Streamlit reactivity

**Negative:**
- `streamlit>=1.38.0` remains in `requirements.txt` as a dead dependency — bloats installs and the Docker image by ~150MB
- Frontend state (session UUID, document list) managed in vanilla JS `app.js` — no reactive framework
- No hot-reload for frontend assets in development

**Action required:**
- Remove `streamlit` from `requirements.txt` (tracked separately)
