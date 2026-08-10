"""FastAPI route for sciencerag.report (spec §5)."""

import logging
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

from sciencerag.common.audit import log_audit_entry
from sciencerag.common.errors import ErrorDetail, ErrorResponse
from sciencerag.common.trace import new_trace_id
from sciencerag.report.models import ReportRequest, ReportResponse
from sciencerag.report.render import build_report, render_pdf
from sciencerag.report.store import list_reports, load_report, store_report

router = APIRouter()
logger = logging.getLogger(__name__)

# spec §8: "report 为尽力而为的批处理" — no hard latency target, unlike the
# other three endpoints; kept only as a soft observability log line.
LATENCY_LOG_THRESHOLD_SECONDS = 30


def _now() -> float:
    return time.monotonic()


@router.post("/sciencerag/report", response_model=ReportResponse)
def post_report(request: ReportRequest) -> ReportResponse | JSONResponse:
    t0 = _now()
    trace_id = new_trace_id()
    try:
        response = build_report(request, trace_id)
        store_report(response)
    except Exception as e:  # noqa: BLE001 - spec §8: always return typed, schema-valid JSON.
        elapsed_s = _now() - t0
        error_response = ErrorResponse(
            error=ErrorDetail(category="schema_validation_failed", message=str(e)),
            trace_id=trace_id,
        )
        log_audit_entry(
            trace_id=trace_id,
            endpoint="sciencerag.report",
            request=request.model_dump(),
            evidence=[],
            output=error_response.model_dump(),
            elapsed_s=round(elapsed_s, 1),
        )
        return JSONResponse(status_code=502, content=error_response.model_dump())

    elapsed_s = _now() - t0
    if elapsed_s > LATENCY_LOG_THRESHOLD_SECONDS:
        logger.info(
            "sciencerag.report took %.1fs (trace_id=%s, no hard target per spec §8)",
            elapsed_s,
            trace_id,
        )

    log_audit_entry(
        trace_id=trace_id,
        endpoint="sciencerag.report",
        request=request.model_dump(),
        evidence=[source.model_dump() for source in response.citations],
        output=response.model_dump(),
        elapsed_s=round(elapsed_s, 1),
    )
    return response


@router.get("/sciencerag/reports")
def get_reports() -> list[dict]:
    """Read-only listing for M5's report-browsing panel (spec §7) — not
    itself a spec-numbered endpoint, just the HTTP surface over
    report/store.py's on-disk index."""
    return list_reports()


@router.get("/sciencerag/reports/{stem}", response_model=ReportResponse)
def get_report(stem: str) -> ReportResponse | JSONResponse:
    try:
        return load_report(stem)
    except FileNotFoundError:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "error": {"category": "insufficient_coverage", "message": f"no report {stem!r}"}},
        )
    except ValueError as e:
        # load_report rejects a path-unsafe stem (see its docstring) —
        # a bad request, not a missing report, so 400 rather than 404.
        return JSONResponse(
            status_code=400,
            content={"status": "error", "error": {"category": "schema_validation_failed", "message": str(e)}},
        )


@router.get("/sciencerag/reports/{stem}/pdf", response_model=None)
def get_report_pdf(stem: str) -> Response | JSONResponse:
    """Renders on demand from the already-stored Markdown rather than
    storing a separate .pdf file per report — PDF rendering is a
    deterministic, cheap function of text already on disk, so there's
    nothing to gain from persisting it, and no backfill needed for
    reports generated before this route existed."""
    try:
        report = load_report(stem)
    except FileNotFoundError:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "error": {"category": "insufficient_coverage", "message": f"no report {stem!r}"}},
        )
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "error": {"category": "schema_validation_failed", "message": str(e)}},
        )

    try:
        pdf_bytes = render_pdf(report.markdown)
    except Exception as e:  # noqa: BLE001 - spec §8: always return typed, schema-valid JSON.
        return JSONResponse(
            status_code=502,
            content={"status": "error", "error": {"category": "schema_validation_failed", "message": str(e)}},
        )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{stem}.pdf"'},
    )
