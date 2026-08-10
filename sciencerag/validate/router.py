"""FastAPI route for sciencerag.validate (spec §4.5).

M2 scope only (spec §10): 4.1 anomaly checks + 4.2 evaluation. 4.3/4.4 are
M3 — update_package.surrogate_update/kg_candidates are always empty here.
"""

import logging
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from sciencerag.common.audit import log_audit_entry
from sciencerag.common.errors import ErrorDetail, ErrorResponse
from sciencerag.common.trace import new_trace_id
from sciencerag.validate.checks import run_anomaly_checks
from sciencerag.validate.evaluation import evaluate
from sciencerag.validate.finetune import suggest_surrogate_update
from sciencerag.validate.kg_candidate_store import store_pending_candidates
from sciencerag.validate.kg_candidates import extract_kg_candidates
from sciencerag.validate.models import UpdatePackage, ValidateRequest, ValidateResponse

router = APIRouter()
logger = logging.getLogger(__name__)

# spec §8: "validate ≤ 60 秒(阻塞闭环推进,其中守恒/残差检查为主要开销)". Soft
# guard only, same as priors/router.py's LATENCY_TARGET_SECONDS.
LATENCY_TARGET_SECONDS = 60


def _now() -> float:
    return time.monotonic()


@router.post("/sciencerag/validate", response_model=ValidateResponse)
def post_validate(request: ValidateRequest) -> ValidateResponse | JSONResponse:
    t0 = _now()
    trace_id = new_trace_id()
    try:
        anomalies = run_anomaly_checks(request)
        evaluation = evaluate(request)
        # spec §4.1: "任何 blocking 级异常都会短路后续所有环节" — a blocked
        # run never reaches 4.3/4.4 at all, not just an empty result from
        # them.
        blocked = any(anomaly.severity == "blocking" for anomaly in anomalies)
        if blocked:
            update_package = UpdatePackage(surrogate_update=None, kg_candidates=[], blocked=True)
        else:
            kg_candidates = extract_kg_candidates(request, evaluation)
            update_package = UpdatePackage(
                surrogate_update=suggest_surrogate_update(request, anomalies, evaluation),
                kg_candidates=kg_candidates,
                blocked=False,
            )
            # Queue for scripts/approve_kg_candidates.py --list-pending; the
            # approval step itself stays a deliberate human action (spec
            # §6.3), this only removes the "manually copy the response body
            # into a file" busywork in front of it.
            store_pending_candidates(request.run_id, kg_candidates)
        response = ValidateResponse(
            anomalies=anomalies,
            evaluation=evaluation,
            update_package=update_package,
            trace_id=trace_id,
        )
    except Exception as e:  # noqa: BLE001 - spec §8: always return typed, schema-valid JSON.
        elapsed_s = _now() - t0
        error_response = ErrorResponse(
            error=ErrorDetail(category="benchmark_unavailable", message=str(e)),
            trace_id=trace_id,
        )
        log_audit_entry(
            trace_id=trace_id,
            endpoint="sciencerag.validate",
            request=request.model_dump(),
            evidence=[],
            output=error_response.model_dump(),
            elapsed_s=round(elapsed_s, 1),
        )
        return JSONResponse(status_code=502, content=error_response.model_dump())

    elapsed_s = _now() - t0
    if elapsed_s > LATENCY_TARGET_SECONDS:
        logger.warning(
            "sciencerag.validate exceeded %ss latency target: %.1fs (trace_id=%s)",
            LATENCY_TARGET_SECONDS,
            elapsed_s,
            trace_id,
        )

    log_audit_entry(
        trace_id=trace_id,
        endpoint="sciencerag.validate",
        request=request.model_dump(),
        evidence=[anomaly.model_dump() for anomaly in response.anomalies],
        output=response.model_dump(),
        elapsed_s=round(elapsed_s, 1),
    )
    return response
