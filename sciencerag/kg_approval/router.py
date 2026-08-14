"""FastAPI routes for sciencerag.kg_approval — the web-panel equivalent of
scripts/approve_kg_candidates.py. Same underlying logic either way
(sciencerag/validate/kg_approval.py + kg_candidate_store.py); this only
adds an HTTP surface, including the same archive-after-approving behavior
the CLI has (spec §6.3's gate is unchanged, just reachable from the
frontend now too)."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from sciencerag.common.validators import reject_path_unsafe_id
from sciencerag.kg_approval.models import (
    ApprovalResult,
    ApproveRequest,
    ApproveResponse,
    PendingBatchDetail,
    PendingBatchSummary,
)
from sciencerag.validate import kg_candidate_store
from sciencerag.validate.kg_approval import approve_candidate

router = APIRouter()


def _safe_stem(stem: str) -> str | JSONResponse:
    try:
        return reject_path_unsafe_id(stem)
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "error": {"category": "invalid_request", "message": str(e)}},
        )


@router.get("/sciencerag/kg_candidates/pending", response_model=list[PendingBatchSummary])
def list_pending() -> list[PendingBatchSummary]:
    return [PendingBatchSummary(**entry) for entry in kg_candidate_store.list_pending()]


@router.get("/sciencerag/kg_candidates/pending/{stem}", response_model=PendingBatchDetail)
def get_pending(stem: str) -> PendingBatchDetail | JSONResponse:
    safe = _safe_stem(stem)
    if isinstance(safe, JSONResponse):
        return safe
    try:
        candidates = kg_candidate_store.load_pending(stem)
    except FileNotFoundError:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "error": {"category": "not_found", "message": f"no pending batch {stem!r}"}},
        )
    return PendingBatchDetail(stem=stem, candidates=candidates)


@router.post("/sciencerag/kg_candidates/pending/{stem}/approve", response_model=ApproveResponse)
def approve_pending(stem: str, request: ApproveRequest) -> ApproveResponse | JSONResponse:
    safe = _safe_stem(stem)
    if isinstance(safe, JSONResponse):
        return safe
    try:
        candidates = kg_candidate_store.load_pending(stem)
    except FileNotFoundError:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "error": {"category": "not_found", "message": f"no pending batch {stem!r}"}},
        )

    if request.approve_all:
        indices = list(range(len(candidates)))
    elif request.indices:
        indices = request.indices
    else:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "error": {"category": "invalid_request", "message": "pass approve_all=true or a non-empty indices list"},
            },
        )

    results: list[ApprovalResult] = []
    for index in indices:
        if not 0 <= index < len(candidates):
            results.append(ApprovalResult(index=index, status="error", error="index out of range"))
            continue
        try:
            triple, status = approve_candidate(candidates[index], request.operator, request.reason)
            results.append(ApprovalResult(index=index, status=status, triple_id=triple.triple_id))
        except ValueError as e:
            # A single bad candidate (e.g. add_triple's non-finite-value
            # gate) shouldn't abort the rest of an otherwise-good batch —
            # same per-candidate isolation the CLI has.
            results.append(ApprovalResult(index=index, status="error", error=str(e)))

    # Archive regardless of partial vs full approval, same as the CLI
    # (kg_candidate_store.archive_pending's docstring: unapproved
    # candidates aren't lost, just no longer in the default queue).
    kg_candidate_store.archive_pending(stem)
    return ApproveResponse(stem=stem, results=results, archived=True)
