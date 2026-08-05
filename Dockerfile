# ScienceRAG: FastAPI app (sciencerag.app:app) serving all four spec
# endpoints plus the /workbench and /demo static pages (spec §7's v1 web
# layer, MiroFish-style pattern). Reproducible install via uv + uv.lock.
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
#
# KNOWN GAP: tec_surrogate/ is not committed to this git repo (see
# project memory / README) — this COPY only works building from a local
# checkout that already has it on disk. A `git clone` + build on a remote
# host will fail here until tec_surrogate/ is committed (possibly via Git
# LFS for its checkpoint/.mph binaries) or fetched as a separate deploy
# step. corpus/papers/ has a similar but narrower gap: only the 6 seed
# papers are tracked in git (see .gitignore's comment), so a fresh clone
# builds with less corpus coverage than this machine has locally.
COPY sciencerag/ ./sciencerag/
COPY corpus/ ./corpus/
COPY tec_surrogate/ ./tec_surrogate/

ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "sciencerag.app:app", "--host", "0.0.0.0", "--port", "8000"]
