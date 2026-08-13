# ScienceRAG MCP Adapter

This adapter lets Hermes call the installed ScienceRAG service (ask / priors /
validate / report / kg_approval) through its local FastAPI API.

Default paths / settings:

- Repo: `D:\comsol\ScienceRAG` (`SCIENCERAG_HOME`)
- API: `http://127.0.0.1:8000` (`SCIENCERAG_API_BASE_URL`, auto-launched if unset/unreachable)
- Launch command: `uv run uvicorn sciencerag.app:app --port 8000`, run from `SCIENCERAG_HOME`

Hermes starts this MCP server through `D:\Hermes\run-sciencerag-mcp-for-hermes.ps1`.

## Requirements on the target machine

Unlike TEC/COMSOL, ScienceRAG isn't a self-contained offline app yet — the
following need to exist under `SCIENCERAG_HOME` before this adapter is useful:

- Python 3.12 + [`uv`](https://docs.astral.sh/uv/) on PATH (or `SCIENCERAG_UV_EXE` set)
- `uv sync` already run once (installs fastapi/uvicorn/paper-qa/torch/sentence-transformers/etc.)
- `corpus/papers/` populated (the retrieval index ScienceRAG reads from)
- `.env` with whatever LLM/embedding API keys `sciencerag/common/config.py` expects
- Frontend build at `frontend/dist/` if the `/app` UI should be reachable too (optional — the MCP tools only need the JSON API)

## Tools

- `sciencerag_launch` / `sciencerag_status` — start the API (if not already running) and check readiness
- `sciencerag_ask` — `POST /sciencerag/ask`
- `sciencerag_graph` — `GET /sciencerag/graph`
- `sciencerag_priors` / `sciencerag_priors_batch_evidence` — `POST /sciencerag/priors[...]`
- `sciencerag_validate` — `POST /sciencerag/validate`
- `sciencerag_report` — `POST /sciencerag/report` (pass the full request body as a dict — it nests the `evaluation`/`update_package` from a prior `sciencerag_validate` call)
- `sciencerag_list_reports` / `sciencerag_get_report` / `sciencerag_export_report_pdf`
- `sciencerag_kg_candidates_pending` / `sciencerag_kg_candidate_detail` / `sciencerag_approve_kg_candidate`
