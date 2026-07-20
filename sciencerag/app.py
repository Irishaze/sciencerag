"""ScienceRAG FastAPI app: a single process, one router per endpoint (§2/§7)."""

from fastapi import FastAPI

from sciencerag.priors.router import router as priors_router

app = FastAPI(title="ScienceRAG")
app.include_router(priors_router)
