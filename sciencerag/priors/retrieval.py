"""PaperQA2 wiring for sciencerag.priors (spec §3.2).

Builds a paper-qa Settings object from our LLM/embedding config (§9 OQ#2)
pointed at the internal literature corpus (corpus/papers/), and exposes a
thin query function. The retrieval index is cached under .pqa_index/
(project-local, gitignored) rather than the shared ~/.pqa/ cache so this
project's index doesn't mix with unrelated projects.
"""

from pathlib import Path

from paperqa import Settings, ask
from paperqa.agents.main import AnswerResponse

from sciencerag.common.config import get_embedding_model, get_llm_model
from sciencerag.common.trace import new_trace_id
from sciencerag.priors.extract import (
    EvidenceItem,
    ExtractionError,
    PipelineTrace,
    extract_priors,
)
from sciencerag.priors.models import Coverage, Prior, PriorsResponse

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_DIR = REPO_ROOT / "corpus" / "papers"
INDEX_DIR = REPO_ROOT / ".pqa_index"


def build_settings() -> Settings:
    llm = get_llm_model()
    settings = Settings(
        llm=llm,
        summary_llm=llm,
        embedding=get_embedding_model(),
        paper_directory=str(CORPUS_DIR),
    )
    # paper-qa defaults summary_llm/agent_llm to OpenAI's gpt-4o independently
    # of the top-level `llm` field. summary_llm=DeepSeek works fine, but
    # agent_llm=DeepSeek gets stuck: it never emits a "complete" tool call and
    # loops on generate_answer indefinitely (confirmed — burned $0.25+ before
    # being killed). Leave agent_llm on its OpenAI default; only llm and
    # summary_llm are DeepSeek.
    settings.agent.index.index_directory = str(INDEX_DIR)
    return settings


def run_query(query: str) -> AnswerResponse:
    return ask(query, settings=build_settings())


# Below this, evidence never reaches the LLM at all — found via manual
# review that PaperQA2's summary_llm sometimes hallucinates a "finding" from
# a paper's reference-list titles rather than its body text, and those
# summaries can still score a moderate relevance (e.g. 0.60) that survives
# the post-extraction CONFIDENCE_THRESHOLD filter. Cutting evidence off
# earlier, before it can seed any prior, is more reliable than trying to
# catch the resulting hallucination after the fact.
MIN_EVIDENCE_RELEVANCE = 0.7


def _build_evidence_table(contexts) -> dict[str, EvidenceItem]:
    table = {}
    i = 0
    for context in contexts:
        relevance = max(0.0, min(1.0, context.score / 10))
        if relevance < MIN_EVIDENCE_RELEVANCE:
            continue
        doc = context.text.doc
        i += 1
        table[f"E{i}"] = EvidenceItem(
            text=context.context,
            doi=getattr(doc, "doi", None),
            span=context.text.name,
            notes=getattr(doc, "title", None),
            relevance=relevance,
        )
    return table


CONFIDENCE_THRESHOLD = 0.5


def _split_by_confidence(priors: list[Prior]) -> tuple[list[Prior], list[Prior]]:
    """Split into (strong, weak) by CONFIDENCE_THRESHOLD.

    NOTE: `priors` here are already LLM-merged/extracted Prior objects, each
    possibly backed by multiple evidence contexts (see extract.py's
    confidence formula) — confidence is a derived score, not a single
    evidence context's raw relevance. A weak prior means "this specific
    finding is weakly supported", not "this evidence snippet was irrelevant".
    """
    strong = [p for p in priors if p.confidence >= CONFIDENCE_THRESHOLD]
    weak = [p for p in priors if p.confidence < CONFIDENCE_THRESHOLD]
    return strong, weak


def _build_gaps(weak_priors: list[Prior], total_hits: int) -> list[str]:
    if total_hits == 0:
        return ["internal corpus returned no relevant evidence for this query"]
    if not weak_priors:
        return []
    papers = sorted({p.notes for p in weak_priors if p.notes})
    papers_str = "; ".join(papers) if papers else "unknown source"
    return [
        f"{len(weak_priors)} low-confidence prior(s) "
        f"(confidence < {CONFIDENCE_THRESHOLD}) were excluded from priors; "
        f"papers: {papers_str}"
    ]


def _build_priors_response(query: str, trace: PipelineTrace | None = None) -> PriorsResponse:
    """Run a real PaperQA2 query, then LLM-extract structured priors from
    the evidence contexts (see extract.py). On extraction failure, return
    an empty-but-valid response with the failure noted in gaps — never a
    half-broken result (spec principle)."""
    response = run_query(query)
    contexts = response.session.contexts

    if not contexts:
        return PriorsResponse(
            priors=[],
            coverage=Coverage(
                internal_hits=0,
                external_hits=0,
                gaps=["internal corpus returned no relevant evidence for this query"],
            ),
            trace_id=new_trace_id(),
        )

    evidence_table = _build_evidence_table(contexts)

    if not evidence_table:
        return PriorsResponse(
            priors=[],
            coverage=Coverage(
                internal_hits=len(contexts),
                external_hits=0,
                gaps=[
                    f"{len(contexts)} evidence context(s) retrieved, but none met the "
                    f"minimum relevance ({MIN_EVIDENCE_RELEVANCE}) required to extract from"
                ],
            ),
            trace_id=new_trace_id(),
        )

    try:
        all_priors = extract_priors(query, evidence_table, trace=trace)
    except ExtractionError as e:
        return PriorsResponse(
            priors=[],
            coverage=Coverage(
                internal_hits=len(contexts),
                external_hits=0,
                gaps=[f"LLM extraction failed schema validation after retries: {e}"],
            ),
            trace_id=new_trace_id(),
        )

    strong_priors, weak_priors = _split_by_confidence(all_priors)
    gaps = _build_gaps(weak_priors, total_hits=len(contexts))

    return PriorsResponse(
        priors=strong_priors,
        coverage=Coverage(internal_hits=len(contexts), external_hits=0, gaps=gaps),
        trace_id=new_trace_id(),
    )


def build_priors_response(query: str) -> PriorsResponse:
    return _build_priors_response(query)


def build_priors_response_with_trace(query: str) -> tuple[PriorsResponse, PipelineTrace]:
    """Same as build_priors_response, but also returns a PipelineTrace
    capturing every intermediate stage — powers the demo's pipeline view
    (GET/POST /sciencerag/priors/_debug). Not part of the spec-compliant
    API contract."""
    trace = PipelineTrace(query=query)
    response = _build_priors_response(query, trace=trace)
    return response, trace
