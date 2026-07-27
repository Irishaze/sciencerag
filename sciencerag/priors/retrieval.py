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
from sciencerag.priors.kg import query_kg
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


def _build_evidence_table(
    contexts, trace: PipelineTrace | None = None
) -> dict[str, EvidenceItem]:
    table = {}
    i = 0
    for context in contexts:
        relevance = max(0.0, min(1.0, context.score / 10))
        doc = context.text.doc
        item = EvidenceItem(
            text=context.context,
            doi=getattr(doc, "doi", None),
            span=context.text.name,
            notes=getattr(doc, "title", None),
            relevance=relevance,
        )
        if trace is not None:
            trace.all_evidence.append(item)
        if relevance < MIN_EVIDENCE_RELEVANCE:
            continue
        i += 1
        table[f"E{i}"] = item
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
    # NOTE: can't use p.notes here — it's the LLM's own clarifying note when
    # the LLM provided one, and only falls back to a paper title otherwise
    # (see extract.py's _to_prior). Using it for "which paper" would show
    # LLM commentary instead of a source half the time. DOI is always real
    # and never overwritten, so use that instead.
    dois = sorted({s.doi for p in weak_priors for s in p.sources if s.doi})
    dois_str = "; ".join(dois) if dois else "unknown source"
    return [
        f"{len(weak_priors)} low-confidence prior(s) "
        f"(confidence < {CONFIDENCE_THRESHOLD}) were excluded from priors; "
        f"source DOIs: {dois_str}"
    ]


# M1-15 / spec §9 OQ#1 ("外部检索:是否需要?用哪些 API?"): decided as an
# explicit no-op for M1 — external retrieval (Semantic Scholar/arXiv) is
# deferred to M6 ("外部检索回退 + 批量证据模式"). `allow_external` is a real,
# validated request field (not silently dropped), but in M1 it can only ever
# make the response note that it was requested-but-unavailable — it never
# changes retrieval behavior or `external_hits` (always 0 until M6).
def _add_external_note(response: PriorsResponse, allow_external: bool) -> PriorsResponse:
    if allow_external:
        response.coverage.gaps.append(
            "allow_external=true was requested, but external retrieval "
            "(Semantic Scholar/arXiv) is not implemented until M6 (spec §9 "
            "OQ#1) — this response is internal-corpus-only"
        )
    return response


def _build_priors_response(
    query: str, trace: PipelineTrace | None = None, allow_external: bool = False
) -> PriorsResponse:
    """Run a real PaperQA2 query, then LLM-extract structured priors from
    the evidence contexts (see extract.py). On extraction failure, return
    an empty-but-valid response with the failure noted in gaps — never a
    half-broken result (spec principle)."""
    # Query priority per spec §3.2: KG first, literature second. Through
    # M1-M4 the graph is an empty stub (see kg.py) — this always returns
    # [], so the fall-through to PaperQA2 below is the only real path for
    # now. Kept as an explicit call (not dead code) so M2+ only has to
    # replace query_kg's internals, not restructure this function.
    query_kg(query)

    response = run_query(query)
    contexts = response.session.contexts

    if not contexts:
        return _add_external_note(
            PriorsResponse(
                priors=[],
                coverage=Coverage(
                    internal_hits=0,
                    external_hits=0,
                    gaps=["internal corpus returned no relevant evidence for this query"],
                ),
                trace_id=new_trace_id(),
            ),
            allow_external,
        )

    evidence_table = _build_evidence_table(contexts, trace=trace)

    if not evidence_table:
        return _add_external_note(
            PriorsResponse(
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
            ),
            allow_external,
        )

    try:
        all_priors = extract_priors(query, evidence_table, trace=trace)
        if trace is not None:
            trace.all_priors = all_priors
    except ExtractionError as e:
        return _add_external_note(
            PriorsResponse(
                priors=[],
                coverage=Coverage(
                    internal_hits=len(contexts),
                    external_hits=0,
                    gaps=[f"LLM extraction failed schema validation after retries: {e}"],
                ),
                trace_id=new_trace_id(),
            ),
            allow_external,
        )

    strong_priors, weak_priors = _split_by_confidence(all_priors)
    gaps = _build_gaps(weak_priors, total_hits=len(contexts))

    return _add_external_note(
        PriorsResponse(
            priors=strong_priors,
            coverage=Coverage(internal_hits=len(contexts), external_hits=0, gaps=gaps),
            trace_id=new_trace_id(),
        ),
        allow_external,
    )


def build_priors_response(query: str, allow_external: bool = False) -> PriorsResponse:
    return _build_priors_response(query, allow_external=allow_external)


def build_priors_response_with_trace(
    query: str, allow_external: bool = False
) -> tuple[PriorsResponse, PipelineTrace]:
    """Same as build_priors_response, but also returns a PipelineTrace
    capturing every intermediate stage — powers the demo's pipeline view
    (GET/POST /sciencerag/priors/_debug). Not part of the spec-compliant
    API contract."""
    trace = PipelineTrace(query=query)
    response = _build_priors_response(query, trace=trace, allow_external=allow_external)
    return response, trace
