# Hermes integration

MCP adapter + install scripts that let Hermes (an AI desktop app running on
the target Windows machine) call this project's FastAPI service.

- `ScienceRAG_MCP/` — thin `FastMCP` adapter wrapping `sciencerag/app.py`'s
  routes (`/sciencerag/ask`, `/priors`, `/validate`, `/report`,
  `/kg_candidates/...`) as MCP tools. Pinned to `mcp==1.28.1` to match the
  version already installed in the shared `COMSOL_Multiphysics_MCP/.venv`
  this and the sibling TEC/COMSOL adapters run under.
- `Hermes-MCP-Setup/` — `Install-MCP-For-Hermes.ps1` (writes the
  `mcp_servers` entries into Hermes's `config.yaml`) plus the `run-*.ps1`
  launcher scripts, and the install README for whoever sets this up on the
  target machine.

This is the source of truth for these files. The actual delivery package
(TEC Multiphysics Studio installer, the COMSOL `.venv`, and everything
else too large/binary for git) is assembled separately and distributed
outside this repo — copy the current versions of `ScienceRAG_MCP/` and
`Hermes-MCP-Setup/` from here into that package when repackaging, rather
than editing the delivery copy directly.

ScienceRAG itself isn't offline-packaged yet: on the target machine,
`ScienceRAG_MCP` expects this repo checked out at `SCIENCERAG_HOME`
(default `D:\comsol\ScienceRAG`) with `uv sync` already run, `corpus/papers/`
populated (~735MB, gitignored — copy separately), `DEEPSEEK_API_KEY` (LLM)
and `OPENAI_API_KEY` (embedding) set in `.env`, and ideally `.pqa_index/`
(the PaperQA2 retrieval cache, also gitignored) copied over too — without
it, the first `sciencerag_priors`/`sciencerag_ask` call re-embeds the whole
corpus from scratch. `deploy-sciencerag-windows.ps1` in this directory
checks all of the above and does a smoke-test launch.
