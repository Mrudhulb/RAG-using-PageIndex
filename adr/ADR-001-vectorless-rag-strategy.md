# ADR-001: Vectorless RAG via PageIndex with Local BM25 Fallback

**Status:** Accepted  
**Date:** 2026-04-11  
**Deciders:** Project team

---

## Context

RAG (Retrieval-Augmented Generation) systems traditionally require a vector database (Pinecone, Weaviate, Chroma, etc.) and an embedding model to convert text into high-dimensional vectors for semantic similarity search. This adds infrastructure cost, API charges for embedding, and operational complexity.

## Decision

Use PageIndex cloud API (tree-based retrieval, no vectors) as the primary retrieval backend. Fall back to an in-process `rank-bm25` BM25 index when `PAGEINDEX_API_KEY` is not set.

- **PageIndex mode:** PDF is uploaded to the PageIndex cloud, which builds a hierarchical tree of content. Retrieval is done via `client.chat_completions(doc_id=...)` — PageIndex performs both retrieval and generation internally. The local app does a secondary BM25 pass over tree nodes only to surface source page references for the UI.
- **BM25 mode:** PDF text extracted with pypdf/pdfplumber, indexed in-memory with `BM25Okapi`. All retrieval and generation happens locally.

## Consequences

**Positive:**
- Zero local ML inference — no GPU, no embedding API cost
- BM25 fallback works fully offline with no API keys
- No vector database to provision or maintain
- pypdf → pdfplumber fallback handles scanned/complex PDFs

**Negative:**
- PageIndex cloud mode incurs external API latency (PDF upload + 10–90s tree build on first ingest)
- BM25 is keyword-based — no semantic similarity for synonyms or paraphrases
- Hybrid mode (PageIndex retrieval + BM25 source pages) is a non-obvious two-step process

**Risks:**
- PageIndex API availability/pricing changes affect primary path
- BM25 quality degrades on domain-specific technical PDFs
