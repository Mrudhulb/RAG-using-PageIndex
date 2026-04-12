# ADR-009: LangSmith Tracing as Opt-In via Environment Variables

**Status:** Accepted  
**Date:** 2026-04-11  
**Deciders:** Project team

---

## Context

LangGraph graph execution needs observability — per-node latency, token usage, and error tracing. LangSmith provides this natively for LangChain/LangGraph. However, tracing should not be a hard dependency (not all deployments have LangSmith accounts, and traces add latency).

## Decision

Tracing is opt-in: enabled only when **both** `LANGCHAIN_TRACING_V2=true` (or `LANGSMITH_TRACING=true`) **and** `LANGCHAIN_API_KEY` (or `LANGSMITH_API_KEY`) are set.

Per-request LangSmith client (`ls.Client`) is created with `tracing_context()` wrapping `graph.invoke()`. The client is flushed (`client.flush()`) before the FastAPI response returns, ensuring traces appear in the dashboard synchronously.

Supports both legacy (`LANGCHAIN_*`) and current (`LANGSMITH_*`) env var naming.

Failure modes:
- If tracing setup fails but `graph.invoke()` succeeded → use the result, log warning
- If `graph.invoke()` itself failed inside the tracing context → re-invoke without tracing

## Consequences

**Positive:**
- Zero-impact when not configured — no imports, no overhead
- Per-request client ensures per-project trace routing
- `client.flush()` guarantees traces are visible before response (avoids "missing traces" issue)
- Dual env-var naming supports older and newer langsmith SDK versions

**Negative:**
- `client.flush()` adds latency to every traced request
- Per-request client creation has overhead vs. a global client
- Dual-invocation risk: if tracing context crashes mid-graph, the re-invoke runs the full graph twice (potential duplicate side effects with external APIs)

**Env vars:**
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=pageindex-rag
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```
