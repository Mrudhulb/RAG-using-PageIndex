# ADR-008: ReAct Agent with Hard Iteration Cap (5)

**Status:** Accepted  
**Date:** 2026-04-11  
**Deciders:** Project team

---

## Context

The agent path uses a ReAct (Reason + Act) loop where the LLM can call tools (Tavily search, Playwright browser) repeatedly until it has enough information to answer. Without a termination condition beyond the LLM deciding to stop, the loop could run indefinitely if the model keeps emitting tool calls.

## Decision

Cap the ReAct loop at **5 iterations** using an `iterations` counter in `ChatState`.

```python
# In _route_after_agent:
if last_msg and getattr(last_msg, "tool_calls", None) and iterations < 5:
    return "tool_node"
return "agent_final_node"
```

## Consequences

**Positive:**
- Prevents runaway tool-calling loops on free-tier models (which may be more likely to loop)
- Bounds latency and API cost per request
- Explicit, auditable limit visible in the graph routing code

**Negative:**
- Complex research tasks requiring more than 5 tool invocations will terminate early
- Cap is hardcoded — no per-request or per-user override
- `agent_final_node` may return "I was unable to find an answer" if capped before resolution

**Configuration path:**
- Move cap to an environment variable (`AGENT_MAX_ITERATIONS`, default 5) for easier tuning
