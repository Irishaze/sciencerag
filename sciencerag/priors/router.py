"""FastAPI route for sciencerag.priors (spec §3.3).

Backed by real PaperQA2 retrieval over corpus/papers/ (see
sciencerag/priors/retrieval.py) and LLM-based extraction (see
sciencerag/priors/extract.py).
"""

from fastapi import APIRouter

from sciencerag.common.audit import log_audit_entry
from sciencerag.common.config import get_embedding_model, get_llm_model
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
        # spec §3.5: index/embedding model version must be traceable per response.
        model_config={"llm_model": get_llm_model(), "embedding_model": get_embedding_model()},
    )

    return response


@router.post("/sciencerag/priors/_debug")
def get_priors_debug(request: PriorsRequest) -> dict:
    """Demo/debug only — NOT part of the spec-compliant API contract.

    Returns the real PriorsResponse plus a full PipelineTrace (evidence
    snippets, prompt, raw LLM output per attempt, confidence math) so the
    demo page can show the pipeline stages, not just the final answer.
    """
    response, trace = retrieval.build_priors_response_with_trace(request.query)
    return {"response": response.model_dump(), "trace": trace.model_dump()}
