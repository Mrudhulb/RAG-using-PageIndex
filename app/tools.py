"""
app/tools.py
------------
LangChain tools for the ReAct agent node:
  - Tavily web search (requires TAVILY_API_KEY)
  - Playwright browser tools (requires `playwright install chromium`)

Usage
-----
    from app.tools import get_all_agent_tools
    tools = get_all_agent_tools()
"""

from __future__ import annotations

import os
import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tavily search tool
# ---------------------------------------------------------------------------

def get_search_tool():
    """Return a TavilySearch tool instance."""
    from langchain_tavily import TavilySearch

    return TavilySearch(
        max_results=5,
        name="tavily_search",
        description=(
            "Search the web for current information, news, facts, "
            "or anything requiring up-to-date knowledge."
        ),
    )


# ---------------------------------------------------------------------------
# Playwright browser tools
# ---------------------------------------------------------------------------

@tool
def browser_navigate(url: str) -> str:
    """
    Navigate to a URL and return the page title plus the first 3000 characters
    of visible text content. Useful for reading a specific web page.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return (
            "Error: playwright is not installed. "
            "Run `pip install playwright` and then `playwright install chromium`."
        )
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto(url, timeout=30000)
                title = page.title()
                text = page.inner_text("body")
                text_preview = text[:3000]
            finally:
                browser.close()
        return f"Page title: {title}\n\nVisible text:\n{text_preview}"
    except Exception as exc:
        logger.warning("[browser_navigate] Failed for %s: %s", url, exc)
        return f"Error navigating to {url}: {exc}"


@tool
def browser_get_text(url: str, css_selector: str = "body") -> str:
    """
    Navigate to a URL and return the text content of the element matching the
    given CSS selector (default: 'body'). Useful for extracting specific
    sections of a page.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return (
            "Error: playwright is not installed. "
            "Run `pip install playwright` and then `playwright install chromium`."
        )
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto(url, timeout=30000)
                element = page.query_selector(css_selector)
                if element is None:
                    text = f"No element found matching selector '{css_selector}'."
                else:
                    text = element.inner_text()
            finally:
                browser.close()
        return text
    except Exception as exc:
        logger.warning("[browser_get_text] Failed for %s (selector=%s): %s", url, css_selector, exc)
        return f"Error getting text from {url} with selector '{css_selector}': {exc}"


@tool
def browser_screenshot(url: str) -> str:
    """
    Navigate to a URL and return a text description of the page including its
    title and a preview of visible text. (Screenshots cannot be returned as
    tool output, so a textual description is provided instead.)
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return (
            "Error: playwright is not installed. "
            "Run `pip install playwright` and then `playwright install chromium`."
        )
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto(url, timeout=30000)
                title = page.title()
                text = page.inner_text("body")
            finally:
                browser.close()
        return (
            f"Screenshot taken of {url} - page title: {title}, "
            f"visible text preview: {text[:500]}"
        )
    except Exception as exc:
        logger.warning("[browser_screenshot] Failed for %s: %s", url, exc)
        return f"Error taking screenshot of {url}: {exc}"


def get_browser_tools() -> list:
    """Return all Playwright browser tools."""
    return [browser_navigate, browser_get_text, browser_screenshot]


# ---------------------------------------------------------------------------
# Combined tool list
# ---------------------------------------------------------------------------

def get_all_agent_tools() -> list:
    """
    Return all available agent tools.

    - Tavily search is included only when TAVILY_API_KEY is set.
    - Playwright tools are always included (they fail gracefully if not installed).
    """
    tools = []
    if os.getenv("TAVILY_API_KEY"):
        try:
            tools.append(get_search_tool())
            logger.info("[tools] Tavily search tool loaded.")
        except Exception as exc:
            logger.warning("[tools] Failed to load Tavily tool: %s", exc)
    else:
        logger.info("[tools] TAVILY_API_KEY not set — skipping Tavily search tool.")
    tools.extend(get_browser_tools())
    return tools
