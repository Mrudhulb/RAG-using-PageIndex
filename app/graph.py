"""
app/graph.py
------------
LangGraph workflow for the FastAPI chatbot.

Graph topology
--------------

  [START] → classifier_node → (route) → general_node  → END
                                       → retrieve_node → generate_node → END

State schema: ChatState (TypedDict)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class ChatState(TypedDict, total=False):
    query: str
    chat_history: List[dict]        # [{"role": "user"|"assistant", "content": str}]
    active_doc: Optional[Any]       # IngestedDocument or None
    model: str                      # OpenRouter model name
    route: str                      # "general" | "document" — set by classifier
    retrieved_chunks: Optional[List[Any]]
    answer: str
    error: str


# ---------------------------------------------------------------------------
# Node: classifier_node
# ---------------------------------------------------------------------------

def classifier_node(state: ChatState) -> Dict[str, Any]:
    """
    Decide whether the query requires the PDF document or can be answered
    from general knowledge.

    If no active_doc is loaded, always route to "general".
    """
    active_doc = state.get("active_doc")
    if active_doc is None:
        logger.info("[classifier_node] No active_doc — routing to general.")
        return {"route": "general"}

    query: str = state.get("query", "")
    model: str = state.get("model", "")

    system_prompt = (
        "You are a router. The user has a PDF document loaded. "
        "Decide if their question requires the PDF document to answer ('document') "
        "or can be answered from general knowledge ('general'). "
        "Reply with exactly one word: general or document."
    )

    try:
        from app.llm import get_llm
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = get_llm(model=model or None)
        try:
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=query)]
            response = llm.invoke(messages)
        except Exception:
            messages = [HumanMessage(content=f"{system_prompt}\n\n{query}")]
            response = llm.invoke(messages)
        raw = response.content.strip().lower()
        route = "document" if "document" in raw else "general"
        logger.info("[classifier_node] LLM classified query as: %s (raw=%r)", route, raw)
        return {"route": route}
    except Exception as exc:
        logger.warning("[classifier_node] Classification failed (%s) — defaulting to document.", exc)
        return {"route": "document"}


# ---------------------------------------------------------------------------
# Conditional routing after classifier
# ---------------------------------------------------------------------------

def _route_after_classifier(state: ChatState) -> str:
    route = state.get("route", "general")
    active_doc = state.get("active_doc")
    if route == "document" and active_doc is not None:
        return "retrieve_node"
    return "general_node"


# ---------------------------------------------------------------------------
# Node: general_node
# ---------------------------------------------------------------------------

def general_node(state: ChatState) -> Dict[str, Any]:
    """
    Answer a general (non-document) question using the LLM with chat history.
    """
    query: str = state.get("query", "")
    chat_history: List[dict] = state.get("chat_history", [])
    model: str = state.get("model", "")

    try:
        from app.llm import get_llm
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        llm = get_llm(model=model or None)

        # Prepend system prompt into first human message for models that
        # don't support SystemMessage (e.g. Gemma via Google AI Studio)
        sys_content = "You are a helpful assistant."
        messages = []
        history_msgs = []
        for msg in chat_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                history_msgs.append(HumanMessage(content=content))
            elif role == "assistant":
                history_msgs.append(AIMessage(content=content))

        first_human = f"{sys_content}\n\n{query}" if not history_msgs else query
        try:
            messages = [SystemMessage(content=sys_content)] + history_msgs + [HumanMessage(content=query)]
            response = llm.invoke(messages)
        except Exception:
            # Fallback: no system message, merge into first human turn
            messages = history_msgs + [HumanMessage(content=first_human)]
            response = llm.invoke(messages)

        answer = response.content.strip()
        logger.info("[general_node] Generated general answer (%d chars).", len(answer))
        return {"answer": answer, "route": "general", "error": ""}
    except Exception as exc:
        logger.exception("[general_node] LLM call failed: %s", exc)
        return {"answer": "", "route": "general", "error": f"LLM error: {exc}"}


# ---------------------------------------------------------------------------
# Node: retrieve_node
# ---------------------------------------------------------------------------

def retrieve_node(state: ChatState) -> Dict[str, Any]:
    """
    Retrieve relevant chunks from the active document.
    """
    active_doc = state.get("active_doc")
    query: str = state.get("query", "")

    if active_doc is None:
        return {"retrieved_chunks": [], "error": "No active document."}
    if not query.strip():
        return {"retrieved_chunks": [], "error": "Query is empty."}

    try:
        from app.retrieval import retrieve
        chunks = retrieve(active_doc, query, top_k=5)
        logger.info("[retrieve_node] Retrieved %d chunks.", len(chunks))
        return {"retrieved_chunks": chunks, "error": ""}
    except Exception as exc:
        logger.exception("[retrieve_node] Retrieval failed: %s", exc)
        return {"retrieved_chunks": [], "error": f"Retrieval error: {exc}"}


# ---------------------------------------------------------------------------
# Node: generate_node
# ---------------------------------------------------------------------------

def generate_node(state: ChatState) -> Dict[str, Any]:
    """
    Generate an answer from retrieved chunks, incorporating chat history.
    """
    chunks = state.get("retrieved_chunks") or []
    query: str = state.get("query", "")
    chat_history: List[dict] = state.get("chat_history", [])
    model: str = state.get("model", "")

    if not chunks:
        return {
            "answer": (
                "No relevant content was found in the document for your question. "
                "Please try rephrasing your question."
            ),
            "route": "document",
            "error": "",
        }

    # PageIndex mode: answer already generated by chat_completions
    if chunks[0].source == "pageindex_answer":
        answer = chunks[0].text
        logger.info("[generate_node] Using PageIndex answer (%d chars).", len(answer))
        return {"answer": answer, "route": "document", "error": ""}

    logger.info("[generate_node] Building RAG prompt with %d chunks.", len(chunks))

    try:
        from app.llm import get_llm, SYSTEM_PROMPT
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        llm = get_llm(model=model or None)

        # Build context string from chunks
        context_parts = []
        for chunk in chunks:
            context_parts.append(f"[Page {chunk.page_number}]\n{chunk.text}")
        context = "\n\n---\n\n".join(context_parts)

        history_msgs = []
        for msg in chat_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                history_msgs.append(HumanMessage(content=content))
            elif role == "assistant":
                history_msgs.append(AIMessage(content=content))

        human_content = (
            f"Context (retrieved pages):\n\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer (cite page numbers):"
        )
        try:
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + history_msgs + [HumanMessage(content=human_content)]
            response = llm.invoke(messages)
        except Exception:
            messages = history_msgs + [HumanMessage(content=f"{SYSTEM_PROMPT}\n\n{human_content}")]
            response = llm.invoke(messages)
        answer = response.content.strip()
        logger.info("[generate_node] Answer generated (%d chars).", len(answer))
        return {"answer": answer, "route": "document", "error": ""}
    except Exception as exc:
        logger.exception("[generate_node] LLM call failed: %s", exc)
        return {"answer": "", "route": "document", "error": f"LLM error: {exc}"}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Construct and compile the LangGraph chat+RAG workflow."""
    builder = StateGraph(ChatState)

    builder.add_node("classifier_node", classifier_node)
    builder.add_node("general_node", general_node)
    builder.add_node("retrieve_node", retrieve_node)
    builder.add_node("generate_node", generate_node)

    builder.add_edge(START, "classifier_node")

    builder.add_conditional_edges(
        "classifier_node",
        _route_after_classifier,
        {
            "general_node": "general_node",
            "retrieve_node": "retrieve_node",
        },
    )

    builder.add_edge("general_node", END)
    builder.add_edge("retrieve_node", "generate_node")
    builder.add_edge("generate_node", END)

    graph = builder.compile()
    logger.info("LangGraph chat graph compiled successfully.")
    return graph


# ---------------------------------------------------------------------------
# Cached singleton graph
# ---------------------------------------------------------------------------

_graph_instance = None


def get_graph():
    """Return the compiled graph, building it once per process."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = build_graph()
    return _graph_instance


# ---------------------------------------------------------------------------
# High-level convenience function
# ---------------------------------------------------------------------------

def run_chat_pipeline(
    query: str,
    chat_history: List[dict],
    active_doc=None,
    model: str = "",
) -> Dict[str, Any]:
    """
    Run the full chat pipeline and return the result.

    Parameters
    ----------
    query:        User's question.
    chat_history: List of {"role": ..., "content": ...} dicts.
    active_doc:   IngestedDocument or None (for general chat).
    model:        OpenRouter model name.

    Returns
    -------
    dict with keys: answer, route, retrieved_chunks, error
    """
    initial_state: ChatState = {
        "query": query,
        "chat_history": chat_history,
        "active_doc": active_doc,
        "model": model,
        "route": "",
        "retrieved_chunks": None,
        "answer": "",
        "error": "",
    }

    graph = get_graph()
    final_state = graph.invoke(initial_state)

    return {
        "answer": final_state.get("answer", ""),
        "route": final_state.get("route", "general"),
        "retrieved_chunks": final_state.get("retrieved_chunks") or [],
        "error": final_state.get("error", ""),
    }
