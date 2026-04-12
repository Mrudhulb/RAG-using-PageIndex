# ADR-006: OpenRouter Free-Tier Models via OpenAI-Compatible API

**Status:** Accepted  
**Date:** 2026-04-11  
**Deciders:** Project team

---

## Context

The application requires an LLM for classification, generation, and agent reasoning. Direct OpenAI or Anthropic APIs incur per-token costs. A zero-cost inference option is needed for development and demos.

## Decision

Use [OpenRouter](https://openrouter.ai) with its OpenAI-compatible API endpoint (`https://openrouter.ai/api/v1`) via `langchain-openai.ChatOpenAI`. Default model: `openai/gpt-oss-20b:free`.

```python
ChatOpenAI(
    model=resolved_model,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    tiktoken_model_name="gpt-3.5-turbo",   # tokenizer proxy
    default_headers={
        "HTTP-Referer": "https://github.com/pageindex-rag",
        "X-Title": "PageIndex RAG",
    },
)
```

The `tiktoken_model_name="gpt-3.5-turbo"` workaround is required because tiktoken doesn't recognise OpenRouter model names.

## Consequences

**Positive:**
- Zero inference cost on free-tier models
- Drop-in LangChain integration via `ChatOpenAI`
- Model can be overridden per-request via `ChatRequest.model` and `OPENROUTER_MODEL` env var

**Negative:**
- Free model availability changes frequently without notice — hardcoded default may become unavailable
- Free tier has rate limits and may have higher latency than paid tiers
- tiktoken tokenizer mismatch — token counts reported to LangSmith may be inaccurate
- OpenRouter is an intermediary — adds one network hop vs. calling model providers directly

**Risk mitigation:**
- `OPENROUTER_MODEL` env var allows switching models without code changes
- Model selection exposed in the chat UI and request body
