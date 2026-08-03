"""ScienceRAG FastAPI app: a single process, one router per endpoint (§2/§7)."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from sciencerag.ask.router import router as ask_router
from sciencerag.priors.router import router as priors_router
from sciencerag.report.router import router as report_router
from sciencerag.validate.router import router as validate_router

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="ScienceRAG")
app.include_router(priors_router)
app.include_router(validate_router)
app.include_router(report_router)
app.include_router(ask_router)


@app.get("/demo")
def demo_page() -> FileResponse:
    """Ad-hoc demo UI for /sciencerag/priors — not part of the official
    roadmap (the real Web frontend is spec §7 / milestone M5, and is built
    around sciencerag.ask, not priors). Just a same-origin static page so
    it can call the API with no CORS setup."""
    return FileResponse(STATIC_DIR / "demo.html")


@app.get("/workbench")
def workbench_page() -> FileResponse:
    """spec §7 v1 web layer: 问答 (sciencerag.ask) + 图谱可视化 + 报告浏览
    (sciencerag.report), as one static page + vanilla JS — same "no build
    tooling" scope decision as /demo, not the eventual Vite/React frontend
    spec §7 describes. The 文献/知识候选审批 panel is deliberately not here:
    spec §7 explicitly allows a v1 CLI substitute
    (scripts/approve_kg_candidates.py), which is what M5/M6 actually ship."""
    return FileResponse(STATIC_DIR / "workbench.html")
