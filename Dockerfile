# =============================================================================
# Stage 1 — builder
# Install Python dependencies into an isolated virtual environment.
# This stage never lands in the final image.
# =============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# gcc is needed to compile some wheels (e.g. pdfplumber C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment and install all Python dependencies
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# =============================================================================
# Stage 2 — runtime
# Lean image: venv from builder + Playwright browser + app source only.
# =============================================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# ── Playwright system-level OS dependencies ────────────────────────────────
# playwright install-deps installs the exact set of libs Chromium needs.
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

RUN apt-get update \
    && playwright install-deps chromium \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# ── Playwright browser binaries ────────────────────────────────────────────
# Store in a well-known path so non-root user can read them.
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/playwright-browsers
RUN playwright install chromium

# ── Non-root user ──────────────────────────────────────────────────────────
RUN groupadd --system appgroup \
    && useradd --system --gid appgroup --home /app appuser

# ── Application source ─────────────────────────────────────────────────────
COPY --chown=appuser:appgroup api/      ./api/
COPY --chown=appuser:appgroup app/      ./app/
COPY --chown=appuser:appgroup frontend/ ./frontend/

# ── Persistent data directories ────────────────────────────────────────────
# .cache/uploads/ — uploaded PDFs
# .cache/         — pickled IngestedDocument objects
# These are mounted as a named volume at runtime so data survives restarts.
RUN mkdir -p .cache/uploads data \
    && chown -R appuser:appgroup /app \
    && chown -R appuser:appgroup /opt/playwright-browsers

USER appuser

# ── Server config ──────────────────────────────────────────────────────────
# Azure Container Apps injects PORT (typically 8080).
# Local docker-compose can override with -e PORT=8000.
ENV PORT=8080
EXPOSE 8080

# Single worker per container — scale out via ACA replicas.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
