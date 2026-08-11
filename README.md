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
- When `allow_external=true` and internal coverage is insufficient, it queries Semantic Scholar and arXiv. Any hit with a downloadable PDF (arXiv always; Semantic Scholar when open-access) gets its full text pulled straight into `corpus/papers/` and is trusted immediately — no approval queue. Semantic Scholar hits with no open-access PDF fall back to abstract-only evidence, tagged `provenance="external_unverified"`.
- `POST /sciencerag/priors/batch_evidence`: given N candidate designs, returns supporting/contradicting/neutral evidence for each, unranked.

### `sciencerag.validate` — simulation result validation and learning

`POST /sciencerag/validate`: called after a simulation run completes. Schema: [validate.schema.json](sciencerag/schemas/validate.schema.json).

1. **Anomaly checks** — `ood` (out-of-distribution latent-space check, requires `latent_state`). A `blocking` result short-circuits everything downstream. (`energy_balance`/`pde_residual` were removed — both depended on tec_surrogate's compositional multi-pair model, which was never a substitute for real multi-pair COMSOL calibration data and only covered up to 20 pairs.)
2. **Result evaluation** — compares against 31 known COMSOL samples and against priors from `sciencerag.priors`, yielding `consistent` / `deviation_found` / `insufficient_benchmark`.
3. **Fine-tune suggestions** (error/uncertainty-driven, omitted if signal is weak) + **knowledge candidate extraction** (only when verdict isn't `deviation_found`) — knowledge candidates are the **only** write path into the knowledge graph, after human approval via `scripts/approve_kg_candidates.py`.

Each candidate carries an `entity_type` (which node type it belongs to — `TECDesign` / `Material` / `SimulationRun` / etc., AI-classified against an ontology generated once from the simulation contract: `scripts/generate_kg_ontology.py` → `data/kg/ontology.json`) separately from its `entity_id` (a deterministic hash of subject+geometry that individuates *which* design/run/material it is — two runs of the identical geometry resolve to the same node). A candidate's object is either a measured number (`object_value`) or a link to another entity (`object_entity_id`) — e.g. every simulation run links to the `TECDesign` it evaluated and the `Material` it used, which is what lets multiple designs actually show up connected in the graph UI instead of as disconnected clusters.

#### Seeding the graph from literature (before any simulation runs exist)

`scripts/seed_kg_from_corpus.py` implements the spec's documented cold-start acceleration (§3.3): instead of waiting for simulation runs to populate the graph, it runs literature-priors queries against the internal corpus and converts the results into knowledge candidates too — `parameter_range`/`material_property` priors become numeric-fact candidates, `scaling_relationship` priors (e.g. "leg_length is positively correlated with total_resistance") become **link** candidates between two parameter entities. (`candidate_config`/`caution` priors don't map onto a single-fact-or-link shape without losing what made them useful, so they're skipped — logged, not silently dropped.)

This does **not** bypass approval — it queues candidates into the exact same `data/kg_candidates/pending/` directory `sciencerag.validate` does, gated by the same `scripts/approve_kg_candidates.py` human checkpoint. Literature-sourced triples get attributed to the source paper's DOI (`KGSource(type="paper", doi=...)`) rather than a simulation run.

```bash
uv run python scripts/seed_kg_from_corpus.py                 # real retrieval + LLM calls, ~1-2 min per query
uv run python scripts/approve_kg_candidates.py --list-pending
uv run python scripts/approve_kg_candidates.py --pending <stem> --list          # preview only
uv run python scripts/approve_kg_candidates.py --pending <stem> --approve-all   # actually writes
```

### `sciencerag.report` — report generation

`POST /sciencerag/report`: assembles design parameters, validation results, and the priors used into a cited report (JSON + Markdown), stored under `data/reports/` (not committed to git). PDF is rendered on demand from the stored Markdown (`markdown` → HTML → `xhtml2pdf`, pure Python, no system libraries needed) rather than stored — free-text fields that reach the report are HTML-escaped before PDF conversion, closing an SSRF found during review where raw `<img src="...">` in a field like `task_context.objective` made the PDF renderer fetch that URL server-side.

```
GET /sciencerag/reports            # list
GET /sciencerag/reports/{stem}     # fetch one (JSON)
GET /sciencerag/reports/{stem}/pdf # fetch one (PDF, rendered on demand)
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
