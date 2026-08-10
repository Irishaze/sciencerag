# ScienceRAG: FastAPI backend (all four spec endpoints) + the real spec §7
# Vite/React frontend (frontend/), built here and served as static files by
# the same backend process — see sciencerag/app.py's FRONTEND_DIST mount.
# Two stages: Node builds the frontend, Python runs the service; only the
# frontend's build OUTPUT crosses into the final image, not Node itself.

# ---- frontend build stage ----
FROM node:22-slim AS frontend-build
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ---- backend runtime stage ----
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.24 /uv /uvx /bin/

WORKDIR /app

# Dependencies first for layer caching — most rebuilds only touch
# application code below, not pyproject.toml/uv.lock. Longer HTTP timeout
# than uv's 30s default — torch/sentence-transformers pull enough large
# wheels that a slow link can time out mid-download otherwise.
ENV UV_HTTP_TIMEOUT=120
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Application code and the runtime data tec_bridge.py / retrieval.py load
# at import time (not just at request time): the internal paper corpus and
# tec_surrogate's trained artifacts.
COPY sciencerag/ ./sciencerag/
COPY corpus/ ./corpus/
COPY tec_surrogate/ ./tec_surrogate/

# Built frontend from the first stage — sciencerag/app.py mounts this at
# /app if the directory exists, so the order (copy after sciencerag/)
# doesn't matter for that check, only that it's present before the server
# actually starts.
COPY --from=frontend-build /frontend/dist ./frontend/dist

ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "sciencerag.app:app", "--host", "0.0.0.0", "--port", "8000"]
