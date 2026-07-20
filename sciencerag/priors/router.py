"""FastAPI route for sciencerag.priors (spec §3.3).

Backed by real PaperQA2 retrieval over corpus/papers/ (see
sciencerag/priors/retrieval.py). Kind/field classification is still a
placeholder pending M1-13.
"""

from fastapi import APIRouter

from sciencerag.common.audit import log_audit_entry
from sciencerag.priors import retrieval
from sciencerag.priors.models import PriorsRequest, PriorsResponse

router = APIRouter()


@router.post("/sciencerag/priors", response_model=PriorsResponse)
def get_priors(request: PriorsRequest) -> PriorsResponse:
    response = retrieval.build_priors_response(request.query)

    log_audit_entry(
        trace_id=response.trace_id,
        endpoint="sciencerag.priors",
        request=request.model_dump(),
        evidence=[source.model_dump() for prior in response.priors for source in prior.sources],
        output=response.model_dump(),
    )

    return response
