# ADR-005: Content-Hash (MD5) as Document Identity

**Status:** Accepted  
**Date:** 2026-04-11  
**Deciders:** Project team

---

## Context

Uploaded PDFs need a stable, collision-resistant handle for cache lookup, session references, and document registry keys. Options include: sequential IDs, UUIDs, filename-based keys, path-based hash, or content-based hash.

## Decision

Use MD5 hash of the **file content** (not the filename or path) as `doc_handle`. Computed in `app/cache.py`:

```python
def _md5(file_path: str) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
```

The pickle cache file is stored as `.cache/<md5>.pkl` and the session points to `doc_handle = md5`.

**Note:** The comment in `api/session.py` says "md5 hash of the original pdf path" — this is incorrect. The implementation hashes file **content**, not the path. The docstring should be corrected.

## Consequences

**Positive:**
- Content-addressable: same PDF uploaded twice → same handle → cache hit, no re-ingestion
- Rename-safe: renaming the file doesn't change the handle
- Natural deduplication

**Negative:**
- MD5 is not collision-resistant for adversarial inputs (acceptable for internal use)
- Full file read required on every `_cache_path()` call to compute the hash — O(file size)
- Misleading documentation in `api/session.py` (says "path hash")

**Action required:**
- Fix docstring in `api/session.py` to say "MD5 hash of PDF file content"
- Consider caching the hash computation for large files
