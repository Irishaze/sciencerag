"""FastAPI route for sciencerag.ask (spec §6, §8)."""

import logging
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from sciencerag.ask.models import AskRequest, AskResponse, Subgraph
from sciencerag.ask.pipeline import answer_question
from sciencerag.common.audit import log_audit_entry
from sciencerag.common.errors import ErrorDetail, ErrorResponse
from sciencerag.common.trace import new_trace_id
from sciencerag.priors.kg import full_graph

router = APIRouter()
logger = logging.getLogger(__name__)

# spec §8: "ask ≤ 10 秒(交互式)". Soft guard, same convention as the other
# endpoints' latency targets.
LATENCY_TARGET_SECONDS = 10


def _now() -> float:
    return time.monotonic()


@router.post("/sciencerag/ask", response_model=AskResponse)
def post_ask(request: AskRequest) -> AskResponse | JSONResponse:
    t0 = _now()
    trace_id = new_trace_id()
    try:
        result = answer_question(request.question, max_hits=request.max_hits)
        response = AskResponse(**result, trace_id=trace_id)
    except Exception as e:  # noqa: BLE001 - spec §8: always return typed, schema-valid JSON.
        elapsed_s = _now() - t0
        error_response = ErrorResponse(
            error=ErrorDetail(category="retrieval_timeout", message=str(e)),
            trace_id=trace_id,
        )
        log_audit_entry(
            trace_id=trace_id,
            endpoint="sciencerag.ask",
            request=request.model_dump(),
            evidence=[],
            output=error_response.model_dump(),
            elapsed_s=round(elapsed_s, 1),
        )
        return JSONResponse(status_code=502, content=error_response.model_dump())

    elapsed_s = _now() - t0
    if elapsed_s > LATENCY_TARGET_SECONDS:
        logger.warning(
            "sciencerag.ask exceeded %ss latency target: %.1fs (trace_id=%s)",
            LATENCY_TARGET_SECONDS,
            elapsed_s,
            trace_id,
        )

    log_audit_entry(
        trace_id=trace_id,
        endpoint="sciencerag.ask",
        request=request.model_dump(),
        evidence=[source.model_dump() for source in response.sources],
        output=response.model_dump(),
        elapsed_s=round(elapsed_s, 1),
    )
    return response


@router.get("/sciencerag/graph", response_model=Subgraph)
def get_graph() -> Subgraph:
    """Read-only listing for the frontend's standalone graph-browsing page
    — not itself a spec-numbered endpoint, just the HTTP surface over
    kg.full_graph() (same pattern as GET /sciencerag/reports over
    report/store.py). Returns every triple currently in the graph, not
    scoped to any one question — spec §6.3's write-path constraint is
    untouched, this only reads."""
    return Subgraph(**full_graph())
