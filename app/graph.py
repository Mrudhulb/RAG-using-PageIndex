"""
app/graph.py
------------
LangGraph workflow for the FastAPI chatbot.

Graph topology
--------------

  [START] → classifier_node → (route) → general_node         → END
                                       → retrieve_node → generate_node → END
                                       → agent_node ↔ tool_node (ReAct loop, max 5 iterations)
                                                    → agent_final_node → END

State schema: ChatState (TypedDict)
"""

from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, Dict, List, Optional

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
    chat_history: List[dict]                    # [{"role": "user"|"assistant", "content": str}]
    active_doc: Optional[Any]                   # IngestedDocument or None
    model: str                                  # OpenRouter model name
    route: str                                  # "general" | "document" | "agent"
    retrieved_chunks: Optional[List[Any]]
    answer: str
    error: str
    # Agent ReAct loop fields
    messages: Annotated[List[Any], operator.add]  # accumulates tool call / response messages
    iterations: int                               # track ReAct loop count to enforce max


# ---------------------------------------------------------------------------
# Node: classifier_node
# ---------------------------------------------------------------------------

def classifier_node(state: ChatState) -> Dict[str, Any]:
    """
    Decide whether the query requires:
      - the PDF document ("document")
      - an agent with web/browser tools ("agent")
      - a plain LLM answer ("general")

    Routing rules:
      1. No active_doc + query needs web info → "agent"
      2. No active_doc + conversational/simple → "general"
      3. active_doc set + query is about document → "document"
      4. active_doc set but ambiguous → "document" (safe default)
    """
    active_doc = state.get("active_doc")
    query: str = state.get("query", "")
    model: str = state.get("model", "")

    system_prompt = (
        "You are a router. Classify the user's query into exactly one of three categories:\n"
        "  general  — conversational, simple knowledge, or factual questions answerable from training data\n"
        "  document — questions about the currently loaded PDF document\n"
        "  agent    — questions that need web search, current events, browsing URLs, or real-time information\n"
    )

    if active_doc is None:
        system_prompt += (
            "\nNo PDF document is currently loaded. "
            "If the query requires up-to-date or web information, reply 'agent'. "
            "Otherwise reply 'general'.\n"
        )
    else:
        system_prompt += (
            "\nA PDF document IS loaded. "
            "If the question is clearly about the document content, reply 'document'. "
            "If it needs live web data, reply 'agent'. "
            "Otherwise reply 'general'.\n"
        )

    system_prompt += "Reply with exactly one word: general, document, or agent."

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
        if "agent" in raw:
            route = "agent"
        elif "document" in raw and active_doc is not None:
            route = "document"
        elif "document" in raw and active_doc is None:
            # LLM said document but there's none — fall back to agent or general
            route = "agent" if _looks_like_web_query(query) else "general"
        else:
            route = "general"

        logger.info("[classifier_node] LLM classified query as: %s (raw=%r)", route, raw)
        return {"route": route}
    except Exception as exc:
        logger.warning("[classifier_node] Classification failed (%s) — using fallback.", exc)
        # Fallback: document if active_doc, else general
        fallback = "document" if active_doc is not None else "general"
        return {"route": fallback}


def _looks_like_web_query(query: str) -> bool:
    """Heuristic: does the query likely require live web information?"""
    web_keywords = [
        "latest", "current", "today", "news", "recent", "now", "price",
        "weather", "stock", "live", "real-time", "update", "2024", "2025",
        "2026", "http", "www.", ".com", ".org", ".net", "website", "search",
    ]
    q_lower = query.lower()
    return any(kw in q_lower for kw in web_keywords)


# ---------------------------------------------------------------------------
# Conditional routing after classifier
# ---------------------------------------------------------------------------

def _route_after_classifier(state: ChatState) -> str:
    route = state.get("route", "general")
    if route == "document" and state.get("active_doc"):
        return "retrieve_node"
    elif route == "agent":
        return "agent_node"
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

        sys_content = "You are a helpful assistant."
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
# Node: agent_node  (ReAct step — LLM decides whether to call tools)
# ---------------------------------------------------------------------------

def agent_node(state: ChatState) -> Dict[str, Any]:
    """
    ReAct agent with Tavily + Playwright tools.

    On the first call (messages is empty) the initial message list is built
    from chat_history + query.  On subsequent calls the messages list already
    contains the tool results appended by tool_node.
    """
    from app.tools import get_all_agent_tools
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from app.llm import get_llm

    tools = get_all_agent_tools()
    llm = get_llm(model=state.get("model") or None)
    llm_with_tools = llm.bind_tools(tools)

    current_messages = state.get("messages") or []
    iterations = state.get("iterations", 0)

    if not current_messages:
        # First call — build initial message list
        system = SystemMessage(content=(
            "You are a helpful assistant with access to web search and browser tools. "
            "Use tavily_search to find current information. "
            "Use browser_navigate to read specific web pages. "
            "Answer the user's question thoroughly."
        ))
        history_msgs = []
        for msg in state.get("chat_history", []):
            if msg["role"] == "user":
                history_msgs.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                history_msgs.append(AIMessage(content=msg["content"]))

        try:
            current_messages = [system] + history_msgs + [HumanMessage(content=state["query"])]
        except Exception:
            current_messages = history_msgs + [
                HumanMessage(content=f"You are a helpful assistant.\n\n{state['query']}")
            ]

    response = llm_with_tools.invoke(current_messages)
    new_messages = [response]

    return {
        "messages": new_messages,
        "iterations": iterations + 1,
        "route": "agent",
    }


# ---------------------------------------------------------------------------
# Node: tool_node  (executes tool calls produced by agent_node)
# ---------------------------------------------------------------------------

def get_tool_node():
    """Build and return a ToolNode loaded with all agent tools."""
    from langgraph.prebuilt import ToolNode
    from app.tools import get_all_agent_tools
    return ToolNode(get_all_agent_tools())


# ---------------------------------------------------------------------------
# Node: agent_final_node  (extracts final answer from accumulated messages)
# ---------------------------------------------------------------------------

def agent_final_node(state: ChatState) -> Dict[str, Any]:
    """
    Extract the final plain-text answer from the agent message history.

    The last AIMessage that has no pending tool_calls is taken as the answer.
    """
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
            return {"answer": msg.content, "route": "agent", "error": ""}
    return {"answer": "I was unable to find an answer.", "route": "agent", "error": ""}


# ---------------------------------------------------------------------------
# Conditional routing after agent_node
# ---------------------------------------------------------------------------

def _route_after_agent(state: ChatState) -> str:
    messages = state.get("messages", [])
    iterations = state.get("iterations", 0)
    last_msg = messages[-1] if messages else None
    # Loop back to tools if the LLM emitted tool calls AND under max iterations
    if last_msg and getattr(last_msg, "tool_calls", None) and iterations < 5:
        return "tool_node"
    return "agent_final_node"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Construct and compile the LangGraph chat+RAG+agent workflow."""
    builder = StateGraph(ChatState)

    builder.add_node("classifier_node", classifier_node)
    builder.add_node("general_node", general_node)
    builder.add_node("retrieve_node", retrieve_node)
    builder.add_node("generate_node", generate_node)
    builder.add_node("agent_node", agent_node)
    builder.add_node("tool_node", get_tool_node())
    builder.add_node("agent_final_node", agent_final_node)

    builder.add_edge(START, "classifier_node")

    builder.add_conditional_edges(
        "classifier_node",
        _route_after_classifier,
        {
            "general_node": "general_node",
            "retrieve_node": "retrieve_node",
            "agent_node": "agent_node",
        },
    )

    builder.add_edge("general_node", END)
    builder.add_edge("retrieve_node", "generate_node")
    builder.add_edge("generate_node", END)

    builder.add_conditional_edges(
        "agent_node",
        _route_after_agent,
        {
            "tool_node": "tool_node",
            "agent_final_node": "agent_final_node",
        },
    )
    builder.add_edge("tool_node", "agent_node")   # ReAct loop
    builder.add_edge("agent_final_node", END)

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
        "messages": [],
        "iterations": 0,
    }

    graph = get_graph()
    final_state = graph.invoke(initial_state)

    return {
        "answer": final_state.get("answer", ""),
        "route": final_state.get("route", "general"),
        "retrieved_chunks": final_state.get("retrieved_chunks") or [],
        "error": final_state.get("error", ""),
    }
