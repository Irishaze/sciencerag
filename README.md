# ScienceRAG

## Overview

ScienceRAG is the scientific RAG service inside the TEC (thermoelectric cooler) simulation closed-loop architecture. It exposes four endpoints that give the Hermes agent evidence-backed priors, validation, and knowledge-graph Q&A.

Full design spec: [docs/spec/sciencerag_spec_zh.md](docs/spec/sciencerag_spec_zh.md).

## Workflow

### `sciencerag.priors` — priors retrieval before an experiment

`POST /sciencerag/priors`: given a research question, retrieves evidence from the internal literature corpus (`corpus/papers/`) and has an LLM extract it into structured priors. Schema: [priors.schema.json](sciencerag/schemas/priors.schema.json).

- Each prior is one of five kinds (`parameter_range` / `material_property` / `scaling_relationship` / `candidate_config` / `caution`), each with its own fixed `value` schema — not a free-form dict.
- Every number in `value`/`notes` must be traceable back to the cited evidence text (exact match or ≤2% relative error); if it can't be traced, the prior is rejected and retried.
- `confidence` is a heuristic ranking score, not a gate — what actually decides whether a prior survives is numeric traceability (deterministic) plus an independent LLM semantic judgment (KEEP/REVIEW/DROP).
- When `allow_external=true` and internal coverage is insufficient, it queries Semantic Scholar abstracts (no full text, no arXiv); hits are tagged `external_unverified` and queued for review.
- `POST /sciencerag/priors/batch_evidence`: given N candidate designs, returns supporting/contradicting/neutral evidence for each, unranked.

### `sciencerag.validate` — simulation result validation and learning

`POST /sciencerag/validate`: called after a simulation run completes. Schema: [validate.schema.json](sciencerag/schemas/validate.schema.json).

1. **Anomaly checks** — `energy_balance` (interface conservation, `n_pairs=1` only), `pde_residual` (PDE residual), `ood` (out-of-distribution latent-space check, requires `latent_state`). Any `blocking` result short-circuits everything downstream.
2. **Result evaluation** — compares against 31 known COMSOL samples and against priors from `sciencerag.priors`, yielding `consistent` / `deviation_found` / `insufficient_benchmark`.
3. **Fine-tune suggestions** (error/uncertainty-driven, omitted if signal is weak) + **knowledge candidate extraction** (only when verdict isn't `deviation_found`) — knowledge candidates are the **only** write path into the knowledge graph, after human approval via `scripts/approve_kg_candidates.py`.

### `sciencerag.report` — report generation

`POST /sciencerag/report`: assembles design parameters, validation results, and the priors used into a cited report (JSON + Markdown), stored under `data/reports/` (not committed to git).

```
GET /sciencerag/reports            # list
GET /sciencerag/reports/{stem}     # fetch one
```

### `sciencerag.ask` — knowledge-graph Q&A

`POST /sciencerag/ask`: first checks the knowledge graph for matching triples; if none, falls back to `sciencerag.priors`' literature retrieval (`fallback_used` is set explicitly in the response). The knowledge graph is a JSON store at `data/kg/graph.json` — conflicting values are kept side by side, never silently overwritten.

**Frontend**: `sciencerag/static/workbench.html` (`/workbench`, static) and `frontend/` (`/app`, Vite/React) are functionally equivalent — both have a Q&A panel (with graph visualization) and a report browser. Literature/knowledge-candidate approval is command-line only in both.

## Quick Start

```bash
uv sync                                          # Python 3.12 deps
uv run uvicorn sciencerag.app:app --port 8000
```

Tests:

```bash
make test-m1-fast   # priors: free — unit tests + schema validation + route smoke test (mocked retrieval)
make test-m1        # priors: + real-call regression fixture (incurs cost)
uv run pytest tests/test_validate_route.py tests/test_validate_schema.py tests/test_validate_regression.py tests/test_validate_m3.py -q
```

Frontend development:

```bash
cd frontend
npm install
npm run dev     # http://localhost:5173, API proxied to the local port-8000 backend via Vite
npm run build    # output to frontend/dist/, auto-mounted at /app when the backend starts
```

Docker deployment:

```bash
docker compose up --build
```

Two-stage build (`node:22-slim` builds the frontend → `python:3.12-slim` runs the service; Node never ships in the final image). Real API keys go in `.env` (see `.env.example`); `data/`, `logs/`, and the PaperQA2 index persist via volumes.

## Status

All four endpoints are implemented and have been exercised with real data, verified end-to-end from a fresh `git clone` + `docker build`. Known limitations are documented in each endpoint's section and in code comments; `docs/spec/sciencerag_spec_zh.md` §9 tracks which open questions now have implementations to review and which are still open. `corpus/papers/` currently has only 6 seed papers committed to git; the rest are managed locally via `.gitignore`.
