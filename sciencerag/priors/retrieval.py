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
from sciencerag.priors.contract import GEOMETRY_FREE_NAMES
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

# Retrieval top-k (spec §3.7): was silently paper-qa's own default (10),
# never tuned for "cover these 12 sim_params.json geometry parameters"
# specifically. scripts/k_sweep_probe.py measured extraction-reachable
# coverage of the 12 geometry_free params vs evidence_k on 10 targeted
# queries: k=10 -> 9/12, k=20 -> 11/12, k=30 -> 11/12 (flat — the extra
# evidence retrieved didn't turn into new coverage, pure wasted cost),
# k=50 -> 12/12 (closes the last param, sink_fin_n, but at ~3x the raw
# context volume of k=10). 20 is the elbow: real gains through 10->20,
# zero marginal gain 20->30. Not going to 50 — the last gap is plausibly
# just how rarely fin count specifically gets reported in the corpus, not
# something k alone reliably fixes, and it's not worth ~3x the per-query
# cost/latency to chase one parameter.
EVIDENCE_K = 20


def build_settings() -> Settings:
    llm = get_llm_model()
    settings = Settings(
        llm=llm,
        summary_llm=llm,
        embedding=get_embedding_model(),
        paper_directory=str(CORPUS_DIR),
    )
    settings.answer.evidence_k = EVIDENCE_K
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


# Below this, evidence never reaches the LLM at all. Originally set to 0.7
# from manual review that PaperQA2's summary_llm sometimes hallucinates a
# "finding" from a paper's reference-list titles rather than its body text,
# with those hallucinated summaries still scoring a moderate ~0.60.
#
# Recalibrated (spec §3.7) with a GPT-4o judge (temperature=0) on a fresh
# 100-query probe collected under the sim-contract-scoped extraction
# (scripts/collect_threshold_data.py + scripts/threshold_judge.py +
# scripts/threshold_curve.py):
#   [0.1, 0.4): 24/25 DROP — off-topic or genuine reference-list content.
#               Clearly still exclude.
#   [0.5, 0.6): 7/12 KEEP (58.3%) — but every DROP reason here was "on
#               topic, not specific enough" (e.g. "describes the
#               measurement technique but gives no number"), NOT
#               reference-list hallucination. That failure mode doesn't
#               reappear until well below 0.5 in this data. Content that's
#               merely "not specific enough" is exactly what extract.py's
#               own validators already reject downstream (parameter_range
#               requires a real numeric value, fields must be in the sim
#               contract, evidence must actually ground the claim) — so
#               admitting it here mostly costs a bit of wasted context,
#               not bad output.
#   [0.6, 0.7): 100% KEEP (n=12) — 0.7 was dropping this as a false gap.
# Lowered to 0.5: recovers the [0.5, 0.6) bucket's real coverage: the
# "noise" let in there is low-risk given extract.py's own downstream
# grounding checks, and [0.1, 0.4) still gets excluded regardless.
MIN_EVIDENCE_RELEVANCE = 0.5


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


def _cap_priors(strong_priors: list[Prior], max_priors: int) -> tuple[list[Prior], list[Prior]]:
    """Split confidence-surviving priors into (returned, truncated) by
    max_priors, keeping the highest-confidence ones — a cap always trims
    the weakest of the strong priors, never an arbitrary prefix of
    extraction order."""
    ranked = sorted(strong_priors, key=lambda p: p.confidence, reverse=True)
    return ranked[:max_priors], ranked[max_priors:]


def _prior_geometry_fields(prior: Prior) -> set[str]:
    fields = set(prior.related_fields)
    if prior.field:
        fields.add(prior.field)
    return fields


def _build_geometry_gaps(all_priors: list[Prior], strong_priors: list[Prior]) -> list[str]:
    """Use the sim contract's 12 geometry_free parameters as the yardstick
    for coverage (spec §3.6): whatever this run didn't end up with a
    confidence-surviving prior for goes into gaps, not silently dropped.

    Distinguishes "extracted but too low-confidence to publish" from
    "nothing extracted at all" where that's cheap to know (from priors we
    already have in hand) — true evidence-relevance attribution (spec §3.6's
    3-way split) needs per-parameter semantic matching over raw retrieval
    contexts, which the spec explicitly defers past v1 ("第一版可先简化为
    '未覆盖'").
    """
    covered = {f for p in strong_priors for f in _prior_geometry_fields(p)}
    drafted = {f for p in all_priors for f in _prior_geometry_fields(p)}
    gaps = []
    for name in sorted(GEOMETRY_FREE_NAMES):
        if name in covered:
            continue
        if name in drafted:
            gaps.append(f"{name} 提取到先验但置信度不足")
        else:
            gaps.append(f"文献中未检索到 {name} 相关证据")
    return gaps


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


def _build_max_priors_gap(truncated_priors: list[Prior], max_priors: int) -> list[str]:
    """Mirrors _build_gaps's disclosure pattern: a prior dropped by the
    max_priors cap is still "missing" from Hermes's point of view, even
    though it met the confidence bar — so it must show up in gaps rather
    than silently vanish (same spec principle as the weak-confidence cut)."""
    if not truncated_priors:
        return []
    dois = sorted({s.doi for p in truncated_priors for s in p.sources if s.doi})
    dois_str = "; ".join(dois) if dois else "unknown source"
    return [
        f"{len(truncated_priors)} additional prior(s) met the confidence threshold "
        f"but were excluded by max_priors={max_priors}; source DOIs: {dois_str}"
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
    query: str,
    trace: PipelineTrace | None = None,
    allow_external: bool = False,
    max_priors: int = 5,
) -> tuple[PriorsResponse, int]:
    """Run a real PaperQA2 query, then LLM-extract structured priors from
    the evidence contexts (see extract.py). On extraction failure, return
    an empty-but-valid response with the failure noted in gaps — never a
    half-broken result (spec principle).

    Returns (response, filtered_material_count) — the latter is audit-log
    metadata, not part of the API contract (spec §3.6: material_property
    drafts the LLM emits anyway are dropped before becoming a Prior; the
    count is logged to logs/audit.jsonl by the caller, not coverage.gaps,
    since it's not something missing from coverage — it's something
    correctly excluded).
    """
    # Query priority per spec §3.2: KG first, literature second. Through
    # M1-M4 the graph is an empty stub (see kg.py) — this always returns
    # [], so the fall-through to PaperQA2 below is the only real path for
    # now. Kept as an explicit call (not dead code) so M2+ only has to
    # replace query_kg's internals, not restructure this function.
    query_kg(query)

    response = run_query(query)
    contexts = response.session.contexts

    if not contexts:
        return (
            _add_external_note(
                PriorsResponse(
                    priors=[],
                    coverage=Coverage(
                        internal_hits=0,
                        external_hits=0,
                        gaps=["internal corpus returned no relevant evidence for this query"]
                        + _build_geometry_gaps([], []),
                    ),
                    trace_id=new_trace_id(),
                ),
                allow_external,
            ),
            0,
        )

    evidence_table = _build_evidence_table(contexts, trace=trace)

    if not evidence_table:
        return (
            _add_external_note(
                PriorsResponse(
                    priors=[],
                    coverage=Coverage(
                        internal_hits=len(contexts),
                        external_hits=0,
                        gaps=[
                            f"{len(contexts)} evidence context(s) retrieved, but none met the "
                            f"minimum relevance ({MIN_EVIDENCE_RELEVANCE}) required to extract from"
                        ]
                        + _build_geometry_gaps([], []),
                    ),
                    trace_id=new_trace_id(),
                ),
                allow_external,
            ),
            0,
        )

    try:
        all_priors, filtered_material_count = extract_priors(query, evidence_table, trace=trace)
        if trace is not None:
            trace.all_priors = all_priors
    except ExtractionError as e:
        return (
            _add_external_note(
                PriorsResponse(
                    priors=[],
                    coverage=Coverage(
                        internal_hits=len(contexts),
                        external_hits=0,
                        gaps=[f"LLM extraction failed schema validation after retries: {e}"]
                        + _build_geometry_gaps([], []),
                    ),
                    trace_id=new_trace_id(),
                ),
                allow_external,
            ),
            0,
        )

    strong_priors, weak_priors = _split_by_confidence(all_priors)
    returned_priors, truncated_priors = _cap_priors(strong_priors, max_priors)

    gaps = _build_gaps(weak_priors, total_hits=len(contexts))
    gaps += _build_max_priors_gap(truncated_priors, max_priors)
    gaps += _build_geometry_gaps(all_priors, returned_priors)

    return (
        _add_external_note(
            PriorsResponse(
                priors=returned_priors,
                coverage=Coverage(internal_hits=len(contexts), external_hits=0, gaps=gaps),
                trace_id=new_trace_id(),
            ),
            allow_external,
        ),
        filtered_material_count,
    )


def build_priors_response(
    query: str, allow_external: bool = False, max_priors: int = 5
) -> tuple[PriorsResponse, int]:
    """Returns (response, filtered_material_count) — see
    _build_priors_response's docstring for why the count travels alongside
    the response instead of inside it."""
    return _build_priors_response(query, allow_external=allow_external, max_priors=max_priors)


def build_priors_response_with_trace(
    query: str, allow_external: bool = False, max_priors: int = 5
) -> tuple[PriorsResponse, PipelineTrace]:
    """Same as build_priors_response, but also returns a PipelineTrace
    capturing every intermediate stage — powers the demo's pipeline view
    (GET/POST /sciencerag/priors/_debug). Not part of the spec-compliant
    API contract."""
    trace = PipelineTrace(query=query)
    response, _filtered_material_count = _build_priors_response(
        query, trace=trace, allow_external=allow_external, max_priors=max_priors
    )
    return response, trace
