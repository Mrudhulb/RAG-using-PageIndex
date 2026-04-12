# ADR-002: LangGraph StateGraph for Workflow Orchestration

**Status:** Accepted  
**Date:** 2026-04-11  
**Deciders:** Project team

---

## Context

The application needs to support three distinct query-handling paths (general LLM, document RAG, ReAct agent) plus a classifier that selects between them. The ReAct agent requires a loop (agent → tools → agent → ...) with a cycle-break condition. This logic needs to be testable, observable via LangSmith, and composable.

## Decision

Use `langgraph.graph.StateGraph` with a `ChatState` TypedDict as the shared state schema. The compiled graph is a process-level singleton (`get_graph()`).

**Graph topology:**
```
START → classifier_node → (conditional) → general_node → END
                                        → retrieve_node → generate_node → END
                                        → agent_node ↔ tool_node (≤5 iterations)
                                                     → agent_final_node → END
```

**State fields:** query, chat_history, active_doc, model, route, retrieved_chunks, answer, error, messages (accumulated via `operator.add`), iterations.

## Consequences

**Positive:**
- Clean separation of concerns — each node is a pure function (state-in, dict-out)
- Conditional edges encode routing logic declaratively
- `messages: Annotated[List, operator.add]` enables natural ReAct message accumulation
- LangSmith tracing wraps the entire `graph.invoke()` call, capturing all node spans

**Negative:**
- `_graph_instance` global singleton is not thread-safe for hot-reloading (requires process restart to pick up graph changes)
- `ChatState` is a `TypedDict` — no domain-object encapsulation; state fields are loosely typed
- The classifier makes an LLM call on every request (cost/latency even for simple queries)

**Alternatives considered:**
- Plain Python `if/elif` routing — rejected: no observability, hard to add new paths
- Celery/async task queues — rejected: over-engineered for single-server deployment
