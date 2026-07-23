"""ScienceRAG FastAPI app: a single process, one router per endpoint (§2/§7)."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from sciencerag.priors.router import router as priors_router

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="ScienceRAG")
app.include_router(priors_router)


@app.get("/demo")
def demo_page() -> FileResponse:
    """Ad-hoc demo UI for /sciencerag/priors — not part of the official
    roadmap (the real Web frontend is spec §7 / milestone M5, and is built
    around sciencerag.ask, not priors). Just a same-origin static page so
    it can call the API with no CORS setup."""
    return FileResponse(STATIC_DIR / "demo.html")
