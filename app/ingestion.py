"""
app/ingestion.py
----------------
PDF ingestion layer.

Two modes depending on environment configuration:

  1. PageIndex cloud mode (PAGEINDEX_API_KEY set)
     - Uploads the PDF to the PageIndex API.
     - Waits until the document is ready for retrieval.
     - Returns an IngestedDocument containing the cloud doc_id.

  2. Local BM25 mode (no PAGEINDEX_API_KEY)
     - Extracts text page-by-page with pypdf / pdfplumber.
     - Builds a rank-bm25 index in memory.
     - Returns an IngestedDocument containing the local BM25 index.
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PageChunk:
    """A single page's worth of text with its 1-based page number."""
    page_number: int
    text: str


@dataclass
class IngestedDocument:
    """
    Carries everything downstream nodes need, regardless of mode.

    Attributes
    ----------
    pdf_path:         Absolute path to the source PDF.
    mode:             'pageindex' or 'bm25'.
    pages:            List of PageChunk objects (always populated).
    doc_id:           PageIndex cloud doc_id (pageindex mode only).
    pageindex_client: Live PageIndexClient instance (pageindex mode only).
    bm25_index:       rank-bm25 BM25Okapi object (bm25 mode only).
    tokenized_corpus: Pre-tokenised pages (bm25 mode only).
    """
    pdf_path: str
    mode: str                              # 'pageindex' | 'bm25'
    pages: List[PageChunk] = field(default_factory=list)
    doc_id: Optional[str] = None          # pageindex mode
    pageindex_client: Optional[Any] = None  # pageindex mode
    bm25_index: Optional[Any] = None      # bm25 mode
    tokenized_corpus: Optional[List[List[str]]] = None  # bm25 mode

    @property
    def page_count(self) -> int:
        return len(self.pages)


# ---------------------------------------------------------------------------
# PDF text extraction (shared utility)
# ---------------------------------------------------------------------------

def extract_pages_from_pdf(pdf_path: str) -> List[PageChunk]:
    """
    Extract text from every page of *pdf_path*.

    Tries pypdf first; falls back to pdfplumber for scanned/complex PDFs.
    """
    pages: List[PageChunk] = []

    # --- attempt 1: pypdf ---------------------------------------------------
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(pdf_path)
        for idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append(PageChunk(page_number=idx, text=text.strip()))

        # If pypdf returned mostly empty pages, try pdfplumber instead
        non_empty = sum(1 for p in pages if len(p.text) > 20)
        if non_empty / max(len(pages), 1) < 0.5:
            raise ValueError("pypdf returned too many empty pages; trying pdfplumber")

        logger.info("pypdf extracted %d pages from %s", len(pages), pdf_path)
        return pages

    except Exception as exc:
        logger.warning("pypdf extraction issue (%s); falling back to pdfplumber.", exc)
        pages = []

    # --- attempt 2: pdfplumber ----------------------------------------------
    try:
        import pdfplumber  # type: ignore

        with pdfplumber.open(pdf_path) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                pages.append(PageChunk(page_number=idx, text=text.strip()))

        logger.info("pdfplumber extracted %d pages from %s", len(pages), pdf_path)
        return pages

    except Exception as exc:
        raise RuntimeError(
            f"Could not extract text from PDF '{pdf_path}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# PageIndex cloud ingestion
# ---------------------------------------------------------------------------

def _ingest_pageindex(pdf_path: str, api_key: str) -> IngestedDocument:
    """
    Upload PDF to PageIndex cloud and wait for the tree to be built.

    Uses get_tree (status==completed) instead of is_retrieval_ready, which
    is unreliable on free-tier keys. Q&A is handled via chat_completions.
    """
    from pageindex import PageIndexClient  # type: ignore

    client = PageIndexClient(api_key=api_key)

    logger.info("Uploading '%s' to PageIndex cloud …", pdf_path)
    result = client.submit_document(file_path=pdf_path)
    doc_id: str = result["doc_id"]
    logger.info("PageIndex doc_id: %s — waiting for tree processing …", doc_id)

    # Poll get_tree until status == completed (typically 10-30s)
    max_wait_seconds = 90
    poll_interval = 3
    elapsed = 0
    tree_nodes: List[Dict[str, Any]] = []

    while elapsed < max_wait_seconds:
        try:
            tree = client.get_tree(doc_id)
            if tree.get("status") == "completed":
                tree_nodes = tree.get("result", []) or []
                logger.info("Tree completed — %d nodes.", len(tree_nodes))
                break
        except Exception as exc:
            logger.warning("Tree polling error (will retry): %s", exc)

        time.sleep(poll_interval)
        elapsed += poll_interval
    else:
        logger.warning(
            "Tree not completed after %ds; continuing with local text extraction.",
            max_wait_seconds,
        )

    # Build PageChunks from tree nodes (preferred) or raw PDF extraction
    if tree_nodes:
        pages = [
            PageChunk(
                page_number=int(node.get("page_index", idx + 1)),
                text=(node.get("text") or node.get("title") or "").strip(),
            )
            for idx, node in enumerate(tree_nodes)
        ]
        logger.info("Built %d PageChunks from tree nodes.", len(pages))
    else:
        pages = extract_pages_from_pdf(pdf_path)

    return IngestedDocument(
        pdf_path=pdf_path,
        mode="pageindex",
        pages=pages,
        doc_id=doc_id,
        pageindex_client=client,
    )


# ---------------------------------------------------------------------------
# Local BM25 ingestion
# ---------------------------------------------------------------------------

def _ingest_bm25(pdf_path: str) -> IngestedDocument:
    """Build an in-memory BM25 index over PDF pages."""
    from rank_bm25 import BM25Okapi  # type: ignore

    pages = extract_pages_from_pdf(pdf_path)

    if not pages:
        raise ValueError(f"No text could be extracted from '{pdf_path}'.")

    tokenized_corpus = [page.text.lower().split() for page in pages]
    bm25_index = BM25Okapi(tokenized_corpus)

    logger.info(
        "BM25 index built over %d pages from '%s'.", len(pages), pdf_path
    )

    return IngestedDocument(
        pdf_path=pdf_path,
        mode="bm25",
        pages=pages,
        bm25_index=bm25_index,
        tokenized_corpus=tokenized_corpus,
    )


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def ingest_pdf(pdf_path: str) -> IngestedDocument:
    """
    Ingest a PDF and return an IngestedDocument.

    Uses PageIndex cloud API if PAGEINDEX_API_KEY is set in the environment;
    otherwise uses local BM25 retrieval.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pageindex_api_key = os.getenv("PAGEINDEX_API_KEY", "").strip()

    if pageindex_api_key:
        logger.info("PAGEINDEX_API_KEY found — using PageIndex cloud mode.")
        return _ingest_pageindex(pdf_path, pageindex_api_key)
    else:
        logger.info("No PAGEINDEX_API_KEY — using local BM25 mode.")
        return _ingest_bm25(pdf_path)
