"""
app/retrieval.py
----------------
Retrieval layer — routes to the correct backend based on ingestion mode.

  PageIndex mode  →  submit_query() + poll get_retrieval() → cloud-ranked chunks
  BM25 mode       →  rank-bm25 local scoring over extracted page text

Both paths return a list of RetrievedChunk objects that downstream nodes consume
in a uniform way.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """
    A retrieved page/chunk with its source metadata.

    Attributes
    ----------
    page_number:  1-based page number in the source PDF.
    text:         The page / chunk text content.
    score:        Relevance score (BM25 score or PageIndex confidence).
    source:       'pageindex' | 'bm25'
    """
    page_number: int
    text: str
    score: float
    source: str


# ---------------------------------------------------------------------------
# PageIndex cloud retrieval
# ---------------------------------------------------------------------------

def _retrieve_pageindex(
    ingested_doc,
    query: str,
    top_k: int,
) -> List[RetrievedChunk]:
    """
    Use PageIndex chat_completions (scoped to doc_id) to answer the query,
    and BM25 over the tree nodes to produce source page chunks for display.

    Parameters
    ----------
    ingested_doc:  IngestedDocument with doc_id, pageindex_client, and pages.
    query:         User's question.
    top_k:         Max number of page chunks to return for source display.
    """
    client = ingested_doc.pageindex_client
    doc_id = ingested_doc.doc_id
    pages = ingested_doc.pages

    logger.info("PageIndex chat_completions query (doc_id=%s): %r", doc_id, query)

    # Get the answer from PageIndex (it does retrieval + generation internally)
    response = client.chat_completions(
        messages=[{"role": "user", "content": query}],
        doc_id=doc_id,
    )
    answer_text = (
        response.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    logger.info("PageIndex answer (%d chars).", len(answer_text))

    # Use BM25 over tree pages to find the most relevant pages for source display
    chunks: List[RetrievedChunk] = []
    if pages:
        from rank_bm25 import BM25Okapi  # type: ignore
        tokenized = [p.text.lower().split() for p in pages]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(query.lower().split())
        ranked = sorted(zip(pages, scores), key=lambda x: x[1], reverse=True)
        for page_chunk, score in ranked[:top_k]:
            if page_chunk.text:
                chunks.append(
                    RetrievedChunk(
                        page_number=page_chunk.page_number,
                        text=page_chunk.text,
                        score=float(score),
                        source="pageindex",
                    )
                )

    # Attach the PageIndex answer as a synthetic first chunk so generate_node
    # can pass it through directly without calling the external LLM again.
    if answer_text:
        chunks.insert(
            0,
            RetrievedChunk(
                page_number=0,
                text=answer_text,
                score=1.0,
                source="pageindex_answer",
            ),
        )

    logger.info("PageIndex retrieval returned %d chunks.", len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Local BM25 retrieval
# ---------------------------------------------------------------------------

def _retrieve_bm25(
    bm25_index,
    tokenized_corpus: list,
    pages,
    query: str,
    top_k: int,
) -> List[RetrievedChunk]:
    """
    Score all pages against *query* with BM25 and return the top-k.
    """
    tokenized_query = query.lower().split()
    scores = bm25_index.get_scores(tokenized_query)

    ranked = sorted(
        zip(pages, scores), key=lambda x: x[1], reverse=True
    )

    chunks: List[RetrievedChunk] = []
    for page_chunk, score in ranked[:top_k]:
        if page_chunk.text:  # skip blank pages
            chunks.append(
                RetrievedChunk(
                    page_number=page_chunk.page_number,
                    text=page_chunk.text,
                    score=float(score),
                    source="bm25",
                )
            )

    logger.info("BM25 retrieval returned %d chunks.", len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def retrieve(
    ingested_doc,  # IngestedDocument (avoid circular import with type hint)
    query: str,
    top_k: int = 5,
) -> List[RetrievedChunk]:
    """
    Retrieve the most relevant chunks for *query* from *ingested_doc*.

    Automatically selects PageIndex cloud or local BM25 based on the
    ingestion mode stored in *ingested_doc*.
    """
    if not query.strip():
        raise ValueError("Query must not be empty.")

    if ingested_doc.mode == "pageindex":
        return _retrieve_pageindex(
            ingested_doc=ingested_doc,
            query=query,
            top_k=top_k,
        )
    else:
        return _retrieve_bm25(
            bm25_index=ingested_doc.bm25_index,
            tokenized_corpus=ingested_doc.tokenized_corpus,
            pages=ingested_doc.pages,
            query=query,
            top_k=top_k,
        )
