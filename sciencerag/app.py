"""ScienceRAG FastAPI app: a single process, one router per endpoint (§2/§7)."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException
from starlette.types import Scope

from sciencerag.ask.router import router as ask_router
from sciencerag.priors.router import router as priors_router
from sciencerag.report.router import router as report_router
from sciencerag.validate.router import router as validate_router

STATIC_DIR = Path(__file__).resolve().parent / "static"
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

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
    """Minimal single-page fallback (问答 + 图谱可视化 + 报告浏览), kept
    around after /app (the real spec §7 Vite/React frontend, see below)
    shipped — useful when frontend/dist hasn't been built (e.g. local dev
    without running `npm run build` first) and as a lighter-weight
    reference implementation. The 文献/知识候选审批 panel is deliberately not
    here in either frontend: spec §7 explicitly allows a v1 CLI substitute
    (scripts/approve_kg_candidates.py), which is what M5/M6 actually ship."""
    return FileResponse(STATIC_DIR / "workbench.html")


class SPAStaticFiles(StaticFiles):
    """Plain StaticFiles 404s on any sub-path that isn't a literal file —
    fine for asset requests (/app/assets/...) but breaks the frontend's
    client-side routes (/app/ask, /app/graph, ...): those only exist in the
    browser via react-router, so a real HTTP request for them (a page
    refresh, a typed/bookmarked/shared URL) has no matching file and 404s
    before React ever loads to handle the route itself. Falls back to
    index.html for anything not found, same standard pattern every SPA
    host uses, so react-router's basename-scoped routes resolve correctly
    regardless of how the browser arrived at them.

    Starlette's StaticFiles *raises* HTTPException(404) in the not-found
    case (see staticfiles.py's get_response) rather than returning a 404
    Response — an override that only checked a returned status code would
    silently never trigger. Confirmed by testing an actual 404 request
    against both versions of this override, not just reading the source."""

    async def get_response(self, path: str, scope: Scope):
        try:
            return await super().get_response(path, scope)
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
            return await super().get_response("index.html", scope)


if FRONTEND_DIST.exists():
    # Real spec §7 Vite/React frontend (frontend/), built separately
    # (`npm run build`) and served as static files from this same process —
    # same "backend does endpoint forwarding, no separate session service"
    # shape spec §7 describes, just same-process instead of a second Flask
    # service, since there's no session state to manage in v1. Mounted at
    # /app, not "/", so it doesn't shadow /demo, /workbench, or the JSON
    # API routes above.
    app.mount("/app", SPAStaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")

    @app.get("/")
    def root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/app")
