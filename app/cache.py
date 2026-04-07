"""
app/cache.py
------------
Local disk cache for IngestedDocument objects.

Documents are pickled to  .cache/<md5_of_pdf>.pkl  in the project root.
The pageindex_client field is excluded from the pickle (it holds a live API
key / connection) and recreated transparently on load.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / ".cache"


def _md5(file_path: str) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_path(file_path: str) -> Path:
    return CACHE_DIR / f"{_md5(file_path)}.pkl"


def load(file_path: str):
    """
    Return a cached IngestedDocument for *file_path*, or None if not cached.
    Recreates the pageindex_client if the doc was in pageindex mode.
    """
    path = _cache_path(file_path)
    if not path.exists():
        return None

    try:
        with open(path, "rb") as f:
            doc = pickle.load(f)

        # Recreate live client for pageindex mode
        if doc.mode == "pageindex" and doc.doc_id:
            api_key = os.getenv("PAGEINDEX_API_KEY", "").strip()
            if api_key:
                from pageindex import PageIndexClient  # type: ignore
                doc.pageindex_client = PageIndexClient(api_key=api_key)

        logger.info("Cache hit for '%s' (mode=%s, pages=%d).", file_path, doc.mode, doc.page_count)
        return doc
    except Exception as exc:
        logger.warning("Cache load failed for '%s': %s — will re-ingest.", file_path, exc)
        path.unlink(missing_ok=True)
        return None


def save(file_path: str, doc) -> None:
    """
    Persist *doc* to disk. The live pageindex_client is excluded from the pickle.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(file_path)

    # Temporarily remove the unpicklable client
    client = doc.pageindex_client
    doc.pageindex_client = None
    try:
        with open(path, "wb") as f:
            pickle.dump(doc, f)
        logger.info("Cached '%s' → %s", file_path, path.name)
    except Exception as exc:
        logger.warning("Cache save failed: %s", exc)
    finally:
        doc.pageindex_client = client  # restore


def delete(file_path: str) -> None:
    """Remove the cached entry for *file_path* if it exists."""
    _cache_path(file_path).unlink(missing_ok=True)


def get_doc_handle(file_path: str) -> str:
    """Return the md5 hash of *file_path* used as the doc_handle everywhere."""
    return _md5(file_path)


def list_cached() -> list[dict]:
    """Return metadata for all cached documents."""
    if not CACHE_DIR.exists():
        return []
    entries = []
    for pkl in CACHE_DIR.glob("*.pkl"):
        try:
            with open(pkl, "rb") as f:
                doc = pickle.load(f)
            entries.append({
                "file": pkl.stem,
                "mode": doc.mode,
                "pages": doc.page_count,
                "size_kb": round(pkl.stat().st_size / 1024, 1),
                "path": str(pkl),
            })
        except Exception:
            pass
    return entries
