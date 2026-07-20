"""FastAPI route for sciencerag.priors (spec §3.3).

M1 stub: returns a hardcoded, schema-valid response regardless of the
request content. Real PaperQA2-backed retrieval replaces this body in a
later M1 step.
"""

from fastapi import APIRouter

from sciencerag.common.audit import log_audit_entry
from sciencerag.common.trace import new_trace_id
from sciencerag.priors.models import (
    Coverage,
    Prior,
    PriorsRequest,
    PriorsResponse,
    SourcePaper,
)

router = APIRouter()


@router.post("/sciencerag/priors", response_model=PriorsResponse)
def get_priors(request: PriorsRequest) -> PriorsResponse:
    priors = [
        Prior(
            prior_id="pr_stub_0001",
            kind="parameter_range",
            field="leg_length_um",
            value={"min": 20, "max": 200, "typical": 60},
            confidence=0.5,
            sources=[SourcePaper(doi="10.0000/stub", span="stub")],
            notes="stub response from M1-6; not yet backed by real retrieval",
        )
    ]
    response = PriorsResponse(
        priors=priors,
        coverage=Coverage(
            internal_hits=0,
            external_hits=0,
            gaps=["stub: retrieval not yet implemented"],
        ),
        trace_id=new_trace_id(),
    )

    log_audit_entry(
        trace_id=response.trace_id,
        endpoint="sciencerag.priors",
        request=request.model_dump(),
        evidence=[source.model_dump() for prior in priors for source in prior.sources],
        output=response.model_dump(),
    )

    return response
