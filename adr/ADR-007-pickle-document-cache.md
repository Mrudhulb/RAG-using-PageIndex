# ADR-007: Pickle Serialization for Document Cache

**Status:** Accepted  
**Date:** 2026-04-11  
**Deciders:** Project team

---

## Context

`IngestedDocument` objects contain complex in-memory state: extracted page text, a `BM25Okapi` index object, tokenized corpus arrays, and (for PageIndex mode) a live API client. These objects need to survive server restarts without re-ingesting PDFs. Options include JSON, msgpack, SQLite, or Python pickle.

## Decision

Serialize `IngestedDocument` to disk as pickle files at `.cache/<md5>.pkl`.

The `pageindex_client` field (a live `PageIndexClient` with an API key / connection) is explicitly excluded from pickle:

```python
client = doc.pageindex_client
doc.pageindex_client = None
pickle.dump(doc, f)
doc.pageindex_client = client  # restore
```

On load, the client is reconstructed from `PAGEINDEX_API_KEY` env var.

## Consequences

**Positive:**
- Preserves the entire `BM25Okapi` index object natively — no serialization schema needed
- Fast load (no re-ingestion or re-indexing on restart)
- Simple implementation

**Negative:**
- **Security risk:** Pickle files are executable — a tampered `.cache/*.pkl` file could achieve RCE. The `.cache/` directory should not be exposed publicly or writable by untrusted processes
- **Python version coupling:** Pickle files may be incompatible across Python minor versions (e.g., 3.11 → 3.12 may require re-ingestion)
- `pageindex_client` exclusion pattern is fragile — any future non-picklable field added to `IngestedDocument` will cause silent data loss without a similar guard

**Mitigations:**
- `.cache/` is in `.dockerignore` and `.gitignore`
- Cache directory permissions should be restricted (Docker image sets `chown appuser`)
- Consider migrating to `joblib` or `cloudpickle` for more robust serialization
