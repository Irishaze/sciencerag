"""FastAPI route for sciencerag.priors (spec §3.3).

Backed by real PaperQA2 retrieval over corpus/papers/ (see
sciencerag/priors/retrieval.py) and LLM-based extraction (see
sciencerag/priors/extract.py).
"""

import logging
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from sciencerag.common.audit import log_audit_entry
from sciencerag.common.config import get_embedding_model, get_llm_model
from sciencerag.common.errors import ErrorDetail, ErrorResponse
from sciencerag.common.trace import new_trace_id
from sciencerag.priors import retrieval
from sciencerag.priors.models import PriorsRequest, PriorsResponse

router = APIRouter()
logger = logging.getLogger(__name__)

# M1-19 / spec §9: "延迟目标: priors ≤ 30 秒(阻塞规划)". Soft guard only — a
# slow response still returns normally; we just log a warning and record the
# real elapsed time in the audit log so latency regressions are visible
# after the fact, without ever hard-failing a request over timing alone.
LATENCY_TARGET_SECONDS = 30


def _now() -> float:
    """Thin seam over time.monotonic() — tests monkeypatch THIS, not the
    global time module, so faking elapsed time can't destabilize unrelated
    code (httpx/starlette internals also call time.monotonic())."""
    return time.monotonic()


@router.post("/sciencerag/priors", response_model=PriorsResponse)
def get_priors(request: PriorsRequest) -> PriorsResponse | JSONResponse:
    # Returning a JSONResponse directly (the error path below) bypasses
    # response_model validation entirely — FastAPI passes Response subclasses
    # through unchanged. response_model still applies to the normal
    # PriorsResponse return, keeping strict validation + accurate OpenAPI docs.
    t0 = _now()
    try:
        response = retrieval.build_priors_response(
            request.query, allow_external=request.allow_external
        )
    except Exception as e:  # noqa: BLE001 - last-resort gate, spec §8: always
        # return typed, schema-valid JSON on failure, never a bare 500.
        elapsed_s = _now() - t0
        error_response = ErrorResponse(
            error=ErrorDetail(category="retrieval_timeout", message=str(e)),
            trace_id=new_trace_id(),
        )
        log_audit_entry(
            trace_id=error_response.trace_id,
            endpoint="sciencerag.priors",
            request=request.model_dump(),
            evidence=[],
            output=error_response.model_dump(),
            model_config={"llm_model": get_llm_model(), "embedding_model": get_embedding_model()},
            elapsed_s=round(elapsed_s, 1),
        )
        return JSONResponse(status_code=502, content=error_response.model_dump())

    elapsed_s = _now() - t0
    if elapsed_s > LATENCY_TARGET_SECONDS:
        logger.warning(
            "sciencerag.priors exceeded %ss latency target: %.1fs (trace_id=%s)",
            LATENCY_TARGET_SECONDS,
            elapsed_s,
            response.trace_id,
        )

    log_audit_entry(
        trace_id=response.trace_id,
        endpoint="sciencerag.priors",
        request=request.model_dump(),
        evidence=[source.model_dump() for prior in response.priors for source in prior.sources],
        output=response.model_dump(),
        # spec §3.5: index/embedding model version must be traceable per response.
        model_config={"llm_model": get_llm_model(), "embedding_model": get_embedding_model()},
        elapsed_s=round(elapsed_s, 1),
    )

    return response


@router.post("/sciencerag/priors/_debug")
def get_priors_debug(request: PriorsRequest) -> dict:
    """Demo/debug only — NOT part of the spec-compliant API contract.

    Returns the real PriorsResponse plus a full PipelineTrace (evidence
    snippets, prompt, raw LLM output per attempt, confidence math) so the
    demo page can show the pipeline stages, not just the final answer.
    """
    response, trace = retrieval.build_priors_response_with_trace(
        request.query, allow_external=request.allow_external
    )
    return {"response": response.model_dump(), "trace": trace.model_dump()}
