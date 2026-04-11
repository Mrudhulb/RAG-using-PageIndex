"""
app/llm.py
----------
LLM setup and answer-generation prompt.

Uses langchain-openai (ChatOpenAI) pointed at the OpenRouter API,
which provides free-tier models via an OpenAI-compatible endpoint.

Required env vars:
  OPENROUTER_API_KEY  — get yours free at https://openrouter.ai/keys

Optional env vars:
  OPENROUTER_MODEL    — default: mistralai/mistral-7b-instruct:free
  LLM_TEMPERATURE     — default: 0.2

Popular free models on OpenRouter:
  mistralai/mistral-7b-instruct:free
  meta-llama/llama-3-8b-instruct:free
  google/gemma-3-4b-it:free
  qwen/qwen3-8b:free
"""

from __future__ import annotations

import os
import logging
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-oss-20b:free"

# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def get_llm(
    model: str | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    """
    Return a ChatOpenAI instance configured for OpenRouter.

    Parameters
    ----------
    model:       Override the model name (falls back to OPENROUTER_MODEL env
                 var, then 'mistralai/mistral-7b-instruct:free').
    temperature: Override temperature (falls back to LLM_TEMPERATURE env var,
                 then 0.2).
    """
    resolved_model = model or os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
    resolved_temp = temperature if temperature is not None else float(
        os.getenv("LLM_TEMPERATURE", "0.2")
    )

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. "
            "Get a free key at https://openrouter.ai/keys and add it to your .env file."
        )

    logger.info("Initialising LLM via OpenRouter: model=%s, temperature=%s", resolved_model, resolved_temp)
    return ChatOpenAI(
        model=resolved_model,
        temperature=resolved_temp,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        # tiktoken doesn't know OpenRouter model names — use a known tokenizer
        tiktoken_model_name="gpt-3.5-turbo",
        default_headers={
            "HTTP-Referer": "https://github.com/pageindex-rag",
            "X-Title": "PageIndex RAG",
        },
    )


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful PDF assistant. \
You answer questions strictly based on the retrieved page excerpts provided. \
If the answer cannot be found in the excerpts, say so honestly. \
Always cite the page numbers that support your answer."""

def build_rag_prompt(query: str, chunks) -> List:
    """
    Build a list of LangChain messages for the RAG Q&A call.

    Parameters
    ----------
    query:  The user's question.
    chunks: List of RetrievedChunk objects.
    """
    context_parts = []
    for chunk in chunks:
        header = f"[Page {chunk.page_number}]"
        context_parts.append(f"{header}\n{chunk.text}")

    context = "\n\n---\n\n".join(context_parts)

    human_content = (
        f"Context (retrieved pages):\n\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer (cite page numbers):"
    )

    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]


# ---------------------------------------------------------------------------
# Convenience: single-call answer generation
# ---------------------------------------------------------------------------

def generate_answer(query: str, chunks, llm: BaseChatModel | None = None) -> str:
    """
    Generate an answer for *query* given *chunks*.

    Parameters
    ----------
    query:  User question.
    chunks: List[RetrievedChunk]
    llm:    Optional pre-built LLM instance; creates one if not supplied.
    """
    if llm is None:
        llm = get_llm()

    messages = build_rag_prompt(query, chunks)
    response = llm.invoke(messages)
    return response.content
