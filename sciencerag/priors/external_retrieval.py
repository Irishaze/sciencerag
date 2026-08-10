"""External literature retrieval (spec §3.2/§3.5, §9 OQ#1) — M6.

Component A supplements the internal corpus with Semantic Scholar's free
search API when `allow_external=True` and internal coverage looks thin
(retrieval.py triggers this on `coverage.gaps` being non-empty). Results
are tagged `provenance="external_unverified"` end to end (Prior.provenance,
spec §3.5) — nothing here promotes a result to "trusted"; that's
scripts/approve_external_papers.py's job, after human review or the
preset-rule auto-promotion spec §3.5 describes.

v1 scope, narrower than spec §3.5's full ingestion pipeline: only the
returned *abstract* is used as evidence text, not the full paper — full
PDF download+parse (spec §3.5 steps 2-4) isn't implemented for external
hits. A paper that gets promoted to "trusted" should go through the normal
internal ingestion pipeline (corpus/papers/) for real full-text coverage,
not keep running off its abstract indefinitely. Papers with no abstract
are skipped outright rather than passed through with empty evidence.

Network failures (timeout, rate limit, DNS) degrade to "no external
augmentation this call" rather than failing the whole priors request —
external retrieval is a best-effort supplement, not a hard dependency.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
from pathlib import Path

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
REQUEST_TIMEOUT_SECONDS = 15.0
DEFAULT_LIMIT = 5

PENDING_PATH = Path("data/external_papers/pending.json")
APPROVED_PATH = Path("data/external_papers/approved.json")

class ExternalPaper(BaseModel):
    title: str
    abstract: str
    doi: str
    year: int | None = None
    semantic_scholar_id: str


class PendingExternalPaper(BaseModel):
    """spec §3.5 step 1: external hits "自动进入本流水线存入库中,但携带
    external_unverified 标签". This queue IS that storage — a lightweight
    JSON index, not the full ingestion pipeline (no PDF/vectors; see module
    docstring)."""

    doi: str
    title: str
    abstract: str
    year: int | None
    hit_count: int
    first_seen: str
    last_seen: str


def _load_pending() -> dict[str, PendingExternalPaper]:
    if not PENDING_PATH.exists():
        return {}
    data = json.loads(PENDING_PATH.read_text(encoding="utf-8"))
    return {item["doi"]: PendingExternalPaper.model_validate(item) for item in data}


def _save_pending(pending: dict[str, PendingExternalPaper]) -> None:
    PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    PENDING_PATH.write_text(
        json.dumps([p.model_dump() for p in pending.values()], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def record_pending_papers(papers: list[ExternalPaper]) -> None:
    """Upserts `papers` into the pending queue, incrementing hit_count for
    ones already seen — spec §3.5's "被内部检索重复命中达到阈值" auto-promote
    rule (scripts/approve_external_papers.py) reads hit_count from here."""
    if not papers:
        return
    pending = _load_pending()
    now = datetime.now(UTC).isoformat()
    for paper in papers:
        existing = pending.get(paper.doi)
        if existing is not None:
            existing.hit_count += 1
            existing.last_seen = now
        else:
            pending[paper.doi] = PendingExternalPaper(
                doi=paper.doi,
                title=paper.title,
                abstract=paper.abstract,
                year=paper.year,
                hit_count=1,
                first_seen=now,
                last_seen=now,
            )
    _save_pending(pending)


def search_semantic_scholar(query: str, limit: int = DEFAULT_LIMIT) -> list[ExternalPaper]:
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,externalIds,year,paperId",
    }
    try:
        response = httpx.get(
            SEMANTIC_SCHOLAR_SEARCH_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning("Semantic Scholar search failed, skipping external augmentation: %s", e)
        return []

    data = response.json()
    papers = []
    for item in data.get("data", []):
        abstract = item.get("abstract")
        if not abstract:
            continue  # no usable evidence text without full-text PDF ingestion
        external_ids = item.get("externalIds") or {}
        doi = external_ids.get("DOI")
        if not doi:
            # sciencerag.priors.models.SourcePaper requires a real `doi:
            # str` — a paper we can't cite properly isn't usable evidence
            # here, not worth inventing a semantic_scholar_id-as-doi hack.
            continue
        papers.append(
            ExternalPaper(
                title=item.get("title") or "(untitled)",
                abstract=abstract,
                doi=doi,
                year=item.get("year"),
                semantic_scholar_id=item.get("paperId") or "",
            )
        )
    return papers
