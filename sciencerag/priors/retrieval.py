"""PaperQA2 wiring for sciencerag.priors (spec §3.2).

Builds a paper-qa Settings object from our LLM/embedding config (§9 OQ#2)
pointed at the internal literature corpus (corpus/papers/), and exposes a
thin query function. The retrieval index is cached under .pqa_index/
(project-local, gitignored) rather than the shared ~/.pqa/ cache so this
project's index doesn't mix with unrelated projects.
"""

import json
import logging
from pathlib import Path

import litellm
from paperqa import Settings, ask
from paperqa.agents.main import AnswerResponse

logger = logging.getLogger(__name__)

from sciencerag.common.config import get_embedding_model, get_llm_model
from sciencerag.common.trace import new_trace_id
from sciencerag.priors.contract import GEOMETRY_FREE_NAMES, GEOMETRY_FREE_PARAMS
from sciencerag.priors.extract import (
    EvidenceItem,
    ExtractionError,
    PipelineTrace,
    ReviewedPrior,
    _strip_code_fences,
    extract_priors,
)
from sciencerag.priors.arxiv_retrieval import search_arxiv
from sciencerag.priors.external_retrieval import (
    ExternalPaper,
    download_new_papers,
    search_semantic_scholar,
)
from sciencerag.priors.kg import query_kg
from sciencerag.priors.models import Coverage, Prior, PriorsResponse

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_DIR = REPO_ROOT / "corpus" / "papers"
INDEX_DIR = REPO_ROOT / ".pqa_index"

# Retrieval top-k (spec §3.7). Original calibration (see
# data/k_sweep_results_relevance_0.7_2026-07-31.json) was done at
# MIN_EVIDENCE_RELEVANCE=0.7: k=10 -> 9/12, k=20/30 -> 11/12, k=50 -> 12/12
# (20 picked as the elbow). Re-measured 2026-08-03 under the current
# relevance=0.5 (data/k_sweep_results.json, same 10 queries, full 40-run
# sweep, real API calls) since a looser relevance filter changes which
# evidence survives to be counted: coverage is now FLAT 12/12 at every
# k in {10,20,30,50} — the union-of-12-params coverage metric this script
# uses no longer distinguishes k values at all, so it can't justify keeping
# k=20 over k=10 by itself.
#
# But raw evidence/prior volume keeps scaling with k even after coverage
# saturates (k=10: 49 evidence/34 priors, k=20: 80/39, k=30: 81/46,
# k=50: 106/69, summed across all 10 queries) — higher k still buys more
# independent supporting evidence per parameter, it just stops unlocking
# NEW parameters. That volume is plausibly load-bearing for the confidence
# formula redesign (plan k-relevance-abstract-lark.md Part 2 candidate B's
# consistency check needs ≥2 distinct DOIs per prior to have anything to
# compare) even though it's now moot for coverage. Left at 20 pending that
# decision — not lowering to 10 preemptively since Part 2 hasn't run yet
# and losing source redundancy now could quietly starve it later; revisit
# once Part 2's real data shows whether the extra volume actually mattered.
EVIDENCE_K = 20


# Fixed seed for the agent_llm's tool-selection loop (query reformulation,
# when to stop searching) — found via real repeated-query testing (this
# session) that thin-coverage fixtures return substantially different
# evidence/priors run-to-run even under an unchanged corpus, traced to this
# loop's decisions, not our own (already temperature=0) extraction call.
# OpenAI's `seed` is "best-effort" determinism (tied to a stable
# system_fingerprint, not a hard guarantee across backend model updates) —
# expected to reduce, not eliminate, that variance.
AGENT_SEED = 20260802

# Models that reject a custom `temperature` outright (BadRequestError,
# "Only the default (1) value is supported") — OpenAI's o-series and the
# gpt-5.6 family so far, confirmed via real calls (scripts/judge_shootout.py
# hit this first). Prefix-matched since exact model strings get a date/tier
# suffix (e.g. "gpt-5.6-luna", "gpt-5.6-sol"). Extend this list if swapping
# SCIENCERAG_LLM_MODEL to a new model hits the same error.
_NO_CUSTOM_TEMPERATURE_PREFIXES = ("gpt-5.6", "o1", "o3")


def _model_list_config(model: str, *, seed: int | None = None) -> dict:
    """A LiteLLM Router model_list config (paper-qa's llm_config/
    summary_llm_config/agent.agent_llm_config shape) for `model`, omitting
    `temperature` for models known not to support overriding it rather than
    letting every summarization/extraction call fail. See
    _NO_CUSTOM_TEMPERATURE_PREFIXES."""
    litellm_params: dict = {"model": model}
    if not model.startswith(_NO_CUSTOM_TEMPERATURE_PREFIXES):
        litellm_params["temperature"] = 0
    if seed is not None:
        litellm_params["seed"] = seed
    return {"model_list": [{"model_name": model, "litellm_params": litellm_params}]}


def build_settings() -> Settings:
    llm = get_llm_model()
    settings = Settings(
        llm=llm,
        summary_llm=llm,
        embedding=get_embedding_model(),
        paper_directory=str(CORPUS_DIR),
    )
    settings.answer.evidence_k = EVIDENCE_K
    # Only sciencerag.ask's fallback path (run_query -> response.session.answer)
    # ever shows this narrative text to a person — sciencerag.priors only reads
    # response.session.contexts (see extract.py), never .answer, so this has no
    # effect on prior extraction. Default answer_length produces one dense
    # paragraph with citations stacked mid-sentence; ask for markdown structure
    # instead so the frontend (react-markdown, see AnswerCard.tsx) can actually
    # render short paragraphs and bullet points instead of a wall of text.
    settings.answer.answer_length = (
        "about 200-300 words, written entirely in Chinese (中文，非英文). "
        "Follow Zinsser's four principles of good writing from 'On Writing "
        "Well': 清晰 clarity, 简洁 simplicity, 简明 brevity (no clutter, no "
        "word wasted), and 人性化 humanity (warm and direct, like explaining "
        "to a friend, not a stiff academic register). A reader with no "
        "physics background should be able to read it start to finish and "
        "come away understanding the main point. Prefer describing what "
        "something DOES in plain words over naming what it's CALLED — you "
        "do not need to name or define a technical term (Seebeck "
        "coefficient, figure of merit, Joule heating, etc.) just because "
        "the source paper uses it; if the underlying idea can be said in "
        "everyday words instead, do that and skip the term entirely. Use "
        "at most one equation total, and only if a number genuinely adds "
        "something the words didn't already say — never open with an "
        "equation. Format as markdown: short paragraphs (2-4 sentences) "
        "separated by blank lines, and a bullet list ('- ...') when "
        "presenting multiple examples or findings side by side. If you do "
        "include an equation, the renderer supports KaTeX — wrap it in "
        "\\( \\) inline or \\[ \\] on its own line; never use a bare $ for "
        "anything (not currency, not math), it is misread as a math "
        "delimiter. This does not change the citation rules above: still "
        "cite sources exactly as instructed, each citation key on its own "
        "inside its own parentheses at the end of the sentence it supports, "
        "e.g. (pqac-d79ef6fa) — never drop the parentheses and never merge "
        "a citation with an equation's \\( \\)."
    )
    # paper-qa defaults summary_llm/agent_llm to OpenAI's gpt-4o independently
    # of the top-level `llm` field. summary_llm=DeepSeek works fine, but
    # agent_llm=DeepSeek gets stuck: it never emits a "complete" tool call and
    # loops on generate_answer indefinitely (confirmed — burned $0.25+ before
    # being killed). Leave agent_llm on its OpenAI default; only llm and
    # summary_llm follow SCIENCERAG_LLM_MODEL.
    settings.agent.index.index_directory = str(INDEX_DIR)
    # Explicit model_list configs (not paper-qa's own default builder) so
    # temperature can be conditionally omitted per _NO_CUSTOM_TEMPERATURE_PREFIXES
    # — needed the moment SCIENCERAG_LLM_MODEL points at a model like
    # gpt-5.6-luna instead of DeepSeek.
    settings.llm_config = _model_list_config(llm)
    settings.summary_llm_config = _model_list_config(llm)
    settings.agent.agent_llm_config = _model_list_config(
        settings.agent.agent_llm, seed=AGENT_SEED
    )
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
) -> tuple[dict[str, EvidenceItem], list[EvidenceItem]]:
    """Returns (survived_table, below_threshold_items). The second element
    exists independently of `trace` (unlike trace.all_evidence, which is
    only ever populated on the debug/probe path) because
    _build_geometry_gaps' relevance-vs-nothing-retrieved attribution needs
    it on every real call, not just traced ones."""
    table = {}
    below_threshold = []
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
            below_threshold.append(item)
            continue
        i += 1
        table[f"E{i}"] = item
    return table, below_threshold


def _cap_priors(priors: list[Prior], max_priors: int) -> tuple[list[Prior], list[Prior]]:
    """Split into (returned, truncated) by max_priors, keeping the
    highest-confidence ones — a cap always trims the lowest-ranked priors,
    never an arbitrary prefix of extraction order.

    `priors` here is everything that already passed extract.py's numeric +
    semantic checks (2026-08-05: confidence stopped gating what counts as a
    valid prior — see extract.py's module docstring for why. There is no
    more "weak, excluded" tier at this stage; every prior here is
    approved, confidence is purely the ranking key for this cap)."""
    ranked = sorted(priors, key=lambda p: p.confidence, reverse=True)
    return ranked[:max_priors], ranked[max_priors:]


def _prior_geometry_fields(prior: Prior) -> set[str]:
    fields = set(prior.related_fields)
    if prior.field:
        fields.add(prior.field)
    return fields


def _covered_params(strong_priors: list[Prior]) -> set[str]:
    return {f for p in strong_priors for f in _prior_geometry_fields(p)}


def _drafted_params(all_priors: list[Prior]) -> set[str]:
    return {f for p in all_priors for f in _prior_geometry_fields(p)}


# Fallback only (see _match_params_to_evidence's docstring for the primary,
# LLM-based path) — used if that real call errors out, so a transient
# failure degrades coverage.gaps' precision rather than the whole request.
# Pure word-overlap: "leg_length" almost never appears verbatim in a paper
# (authors call it "leg height" as often as "leg length"), and the reverse
# problem — a word matching for the wrong reason — is real too (e.g. "fin
# height" contains "height", which would wrongly also flag the unrelated
# `height` contract parameter). A keyword heuristic can't tell those apart;
# an LLM reading the actual sentence can.
_PARAM_KEYWORD_SYNONYMS: dict[str, list[str]] = {
    "leg_length": ["leg length", "leg height", "leg dimension"],
    "leg_width": ["leg width", "leg cross-section", "leg cross section"],
    "pitch": ["pitch", "leg spacing", "leg pitch"],
    "d_conductor": ["conductor thickness", "metallization thickness", "electrode thickness"],
    "d_ceramics": ["ceramic thickness", "ceramic substrate thickness"],
    "length": ["module length", "device length"],
    "width": ["module width", "device width"],
    "height": ["module height", "device height"],
    "sink_base_h": ["base thickness", "base plate thickness", "sink base"],
    "sink_fin_h": ["fin height"],
    "sink_fin_w": ["fin thickness", "fin width"],
    "sink_fin_n": ["fin count", "number of fins", "fin number"],
}


def _keyword_fallback_match(
    below_threshold_evidence: list[EvidenceItem], candidate_params: list[str]
) -> set[str]:
    def mentions(text: str, param_name: str) -> bool:
        text_lower = text.lower()
        keywords = [*_PARAM_KEYWORD_SYNONYMS.get(param_name, []), param_name.replace("_", " ")]
        return any(kw in text_lower for kw in keywords)

    return {
        name
        for name in candidate_params
        if any(mentions(item.text, name) for item in below_threshold_evidence)
    }


_PARAM_MATCH_PROMPT = """You are analyzing why a literature search found no usable evidence for
certain simulation parameters, for a thermoelectric cooler (TEC) research
assistant.

You will be given EVIDENCE SNIPPETS that were retrieved but scored too low
on relevance to use for extraction, and a list of CANDIDATE PARAMETERS.

For each candidate parameter, decide whether ANY of the evidence snippets
actually discusses that specific physical parameter — even in passing,
even without a specific number. Do NOT match on incidental word overlap:
"fin height" is about a heat sink fin, not a device's overall "height", even
though the word appears in both.

Output JSON only, no markdown fences:
{"mentioned_params": ["<name>", ...]}
(only names from CANDIDATE PARAMETERS; empty list if none are discussed)"""


def _match_params_to_evidence(
    below_threshold_evidence: list[EvidenceItem], candidate_params: list[str]
) -> set[str]:
    """One LLM call per query (not one per parameter/evidence pair): given
    the evidence that was retrieved but didn't clear MIN_EVIDENCE_RELEVANCE,
    which of `candidate_params` does at least one snippet actually discuss?
    Used only to label coverage.gaps ("检索到但相关性不足" vs "未检索到") —
    never gates what becomes a Prior, so a failure here degrades to the
    keyword heuristic (_keyword_fallback_match) rather than failing the
    whole request. No-ops (no API call) if there's nothing to ask about.
    """
    if not below_threshold_evidence or not candidate_params:
        return set()

    param_descriptions = {p["name"]: p["desc"] for p in GEOMETRY_FREE_PARAMS}
    params_block = "\n".join(
        f"- {name}: {param_descriptions.get(name, '')}" for name in candidate_params
    )
    evidence_block = "\n\n".join(
        f"[{i}] {item.text}" for i, item in enumerate(below_threshold_evidence, 1)
    )
    messages = [
        {"role": "system", "content": _PARAM_MATCH_PROMPT},
        {
            "role": "user",
            "content": f"CANDIDATE PARAMETERS:\n{params_block}\n\nEVIDENCE SNIPPETS:\n{evidence_block}",
        },
    ]
    try:
        model = get_llm_model()
        try:
            response = litellm.completion(model=model, messages=messages, temperature=0)
        except litellm.BadRequestError as e:
            if "temperature" not in str(e):
                raise
            response = litellm.completion(model=model, messages=messages)
        raw = response.choices[0].message.content
        parsed = json.loads(_strip_code_fences(raw))
        matched = set(parsed.get("mentioned_params", []))
        return matched & set(candidate_params)
    except Exception:  # noqa: BLE001 - explanatory metadata only, never fail the request over it
        return _keyword_fallback_match(below_threshold_evidence, candidate_params)


def _build_geometry_gaps(
    all_kept_priors: list[Prior],
    returned_priors: list[Prior],
    relevance_matched_params: set[str],
) -> list[str]:
    """Use the sim contract's 12 geometry_free parameters as the yardstick
    for coverage (spec §3.6): whatever this run didn't end up with a
    returned prior for goes into gaps, not silently dropped, with a 3-way
    attribution per parameter (spec §3.7):
      1. extracted (passed numeric + semantic checks) but not in the final
         returned list — either cut by max_priors or excluded as REVIEW
         (see extract.py's module docstring: confidence no longer gates
         what counts as "extracted", only what's returned/ranked).
      2. evidence retrieved but relevance-filtered — `relevance_matched_params`
         is the caller's already-resolved answer (see
         _match_params_to_evidence) for which uncovered/undrafted params at
         least one below-threshold evidence snippet actually discusses.
         Deliberately not computed in here — this stays a pure function so
         its tier-priority logic is unit-testable without a real LLM call.
      3. nothing retrieved at all, as a last resort.

    `all_kept_priors` must be the superset of `returned_priors` (everything
    that passed numeric + semantic checks, before the max_priors cap and
    including REVIEW-excluded ones) — the caller is responsible for that
    union, kept out of this function so it stays pure/unit-testable.
    """
    covered = _covered_params(returned_priors)
    drafted = _drafted_params(all_kept_priors)
    gaps = []
    for name in sorted(GEOMETRY_FREE_NAMES):
        if name in covered:
            continue
        if name in drafted:
            gaps.append(f"{name} 提取到先验但未进入最终结果")
        elif name in relevance_matched_params:
            gaps.append(f"{name} 检索到证据但相关性不足(未达到 {MIN_EVIDENCE_RELEVANCE})")
        else:
            gaps.append(f"文献中未检索到 {name} 相关证据")
    return gaps


def _build_gaps(review_priors: list[ReviewedPrior], total_hits: int) -> list[str]:
    """Reports priors the semantic support judge marked REVIEW (spec §3.7):
    plausible domain-standard inference, not literal evidence wording — not
    wrong the way a DROP is, but not confident enough to include either.
    Confidence no longer excludes priors on its own (see extract.py's
    module docstring) — REVIEW is the only remaining "extracted but not
    included" reason at this stage."""
    if total_hits == 0:
        return ["internal corpus returned no relevant evidence for this query"]
    if not review_priors:
        return []
    # NOTE: can't use p.notes here — it's the LLM's own clarifying note when
    # the LLM provided one, and only falls back to a paper title otherwise
    # (see extract.py's _to_prior). Using it for "which paper" would show
    # LLM commentary instead of a source half the time. DOI is always real
    # and never overwritten, so use that instead.
    dois = sorted({s.doi for rp in review_priors for s in rp.prior.sources if s.doi})
    dois_str = "; ".join(dois) if dois else "unknown source"
    reasons = "; ".join(f"{rp.prior.prior_id}: {rp.reason}" for rp in review_priors)
    return [
        f"{len(review_priors)} prior(s) had uncertain semantic support and were excluded "
        f"from priors; source DOIs: {dois_str}; reasons: {reasons}"
    ]


def _build_max_priors_gap(truncated_priors: list[Prior], max_priors: int) -> list[str]:
    """Mirrors _build_gaps's disclosure pattern: a prior dropped by the
    max_priors cap is still "missing" from Hermes's point of view, even
    though it passed every quality check — so it must show up in gaps
    rather than silently vanish (same spec principle as the REVIEW cut)."""
    if not truncated_priors:
        return []
    dois = sorted({s.doi for p in truncated_priors for s in p.sources if s.doi})
    dois_str = "; ".join(dois) if dois else "unknown source"
    return [
        f"{len(truncated_priors)} additional prior(s) passed all quality checks "
        f"but were excluded by max_priors={max_priors}; source DOIs: {dois_str}"
    ]


# spec §9 OQ#1 ("外部检索:是否需要?用哪些 API?"): M6 implements Semantic
# Scholar + arXiv. `allow_external` only changes behavior when internal
# coverage is thin (see the `coverage.gaps` check below).
def _augment_with_external(
    response: PriorsResponse, query: str, allow_external: bool
) -> PriorsResponse:
    """M6 (spec §3.2/§3.5): supplement with Semantic Scholar + arXiv when
    internal coverage is insufficient (any `coverage.gaps`) and the caller
    opted in.

    Hits with a real downloadable PDF (all arXiv hits; Semantic Scholar
    hits with an open-access PDF) get their full text pulled straight into
    corpus/papers/ (external_retrieval.download_new_papers) and re-indexed
    by PaperQA2 on a second query — from then on they're ordinary internal
    evidence, `provenance="internal"`, no approval step. Semantic Scholar
    hits with no open-access PDF fall back to abstract-only evidence,
    tagged `provenance="external_unverified"` — that tag reflects "this is
    a thin abstract snippet", not a trust judgment, since there's no way
    to obtain full text for a paywalled paper."""
    if not allow_external or not response.coverage.gaps:
        return response

    semantic_scholar_papers = search_semantic_scholar(query)
    arxiv_papers = search_arxiv(query)

    # The same paper can legitimately turn up in both searches (e.g. an
    # arXiv preprint that Semantic Scholar also indexes under the same
    # DOI) — dedup by DOI so external_hits reflects distinct papers found,
    # not search hits, and so it isn't downloaded/extracted twice. When a
    # DOI appears from both sources, prefer whichever copy has a pdf_url
    # so a paper doesn't lose its full-text eligibility just because the
    # abstract-only source happened to be deduped in second.
    all_papers_by_doi: dict[str, ExternalPaper] = {}
    for paper in semantic_scholar_papers + arxiv_papers:
        existing = all_papers_by_doi.get(paper.doi)
        if existing is None or (not existing.pdf_url and paper.pdf_url):
            all_papers_by_doi[paper.doi] = paper
    all_papers = list(all_papers_by_doi.values())

    if not all_papers:
        response.coverage.gaps.append(
            "allow_external=true and internal coverage was insufficient, but "
            "Semantic Scholar and arXiv search both returned no usable results"
        )
        return response

    full_text_candidates = [p for p in all_papers if p.pdf_url]
    abstract_only = [p for p in all_papers if not p.pdf_url]
    newly_downloaded = download_new_papers(full_text_candidates)

    external_priors: list[Prior] = []
    extraction_errors: list[str] = []

    if newly_downloaded:
        newly_downloaded_dois = {p.doi for p in newly_downloaded}
        try:
            # This re-query is the one place in the augmentation path that
            # wasn't previously guarded: run_query hits a real LLM/PaperQA2
            # pipeline a second time, and a transient failure here (a
            # timeout, a provider error, an unparseable file that just
            # landed on disk) must not take down a request whose *first*
            # pass may already have produced perfectly good internal
            # priors — external augmentation is explicitly best-effort
            # (module docstring), and the router's catch-all turns any
            # uncaught exception here into a 502 for the whole response,
            # discarding those results. Broad except is deliberate: this
            # boundary needs to swallow whatever a third-party retrieval
            # pipeline can throw, not just the extraction-specific error.
            full_text_contexts = run_query(query).session.contexts
            full_text_evidence, _below_threshold = _build_evidence_table(full_text_contexts)
            full_text_evidence = {
                label: item
                for label, item in full_text_evidence.items()
                if item.doi in newly_downloaded_dois
            }
            if full_text_evidence:
                full_text_priors, _filtered_material_count, _review_priors = extract_priors(
                    query, full_text_evidence, trace=None
                )
                external_priors += full_text_priors
        except ExtractionError as e:
            extraction_errors.append(str(e))
        except Exception as e:  # noqa: BLE001 - best-effort augmentation, see comment above
            logger.warning("Full-text external augmentation query failed: %s", e)
            extraction_errors.append(f"full-text re-query failed: {e}")

    if abstract_only:
        evidence_table = {
            f"EXT{i + 1}": EvidenceItem(
                text=paper.abstract, doi=paper.doi, span="abstract", notes=paper.title, relevance=1.0
            )
            for i, paper in enumerate(abstract_only)
        }
        try:
            abstract_priors, _filtered_material_count, _review_priors = extract_priors(
                query, evidence_table, trace=None
            )
            for prior in abstract_priors:
                prior.provenance = "external_unverified"
            external_priors += abstract_priors
        except ExtractionError as e:
            extraction_errors.append(str(e))

    if extraction_errors and not external_priors:
        response.coverage.gaps.append(
            f"external retrieval found {len(all_papers)} paper(s), but LLM "
            f"extraction failed schema validation after retries: "
            f"{'; '.join(extraction_errors)}"
        )

    response.priors = response.priors + external_priors
    response.coverage.external_hits = len(all_papers)
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
            _augment_with_external(
                PriorsResponse(
                    priors=[],
                    coverage=Coverage(
                        internal_hits=0,
                        external_hits=0,
                        gaps=["internal corpus returned no relevant evidence for this query"]
                        + _build_geometry_gaps([], [], set()),
                    ),
                    trace_id=new_trace_id(),
                ),
                query,
                allow_external,
            ),
            0,
        )

    evidence_table, below_threshold_evidence = _build_evidence_table(contexts, trace=trace)

    if not evidence_table:
        relevance_matched = _match_params_to_evidence(
            below_threshold_evidence, sorted(GEOMETRY_FREE_NAMES)
        )
        return (
            _augment_with_external(
                PriorsResponse(
                    priors=[],
                    coverage=Coverage(
                        internal_hits=len(contexts),
                        external_hits=0,
                        gaps=[
                            f"{len(contexts)} evidence context(s) retrieved, but none met the "
                            f"minimum relevance ({MIN_EVIDENCE_RELEVANCE}) required to extract from"
                        ]
                        + _build_geometry_gaps([], [], relevance_matched),
                    ),
                    trace_id=new_trace_id(),
                ),
                query,
                allow_external,
            ),
            0,
        )

    try:
        kept_priors, filtered_material_count, review_priors = extract_priors(
            query, evidence_table, trace=trace
        )
        if trace is not None:
            trace.all_priors = kept_priors
    except ExtractionError as e:
        relevance_matched = _match_params_to_evidence(
            below_threshold_evidence, sorted(GEOMETRY_FREE_NAMES)
        )
        return (
            _augment_with_external(
                PriorsResponse(
                    priors=[],
                    coverage=Coverage(
                        internal_hits=len(contexts),
                        external_hits=0,
                        gaps=[f"LLM extraction failed schema validation after retries: {e}"]
                        + _build_geometry_gaps([], [], relevance_matched),
                    ),
                    trace_id=new_trace_id(),
                ),
                query,
                allow_external,
            ),
            0,
        )

    returned_priors, truncated_priors = _cap_priors(kept_priors, max_priors)
    all_kept_priors = kept_priors + [rp.prior for rp in review_priors]

    unexplained = sorted(
        GEOMETRY_FREE_NAMES - _covered_params(returned_priors) - _drafted_params(all_kept_priors)
    )
    relevance_matched = _match_params_to_evidence(below_threshold_evidence, unexplained)

    gaps = _build_gaps(review_priors, total_hits=len(contexts))
    gaps += _build_max_priors_gap(truncated_priors, max_priors)
    gaps += _build_geometry_gaps(all_kept_priors, returned_priors, relevance_matched)

    return (
        _augment_with_external(
            PriorsResponse(
                priors=returned_priors,
                coverage=Coverage(internal_hits=len(contexts), external_hits=0, gaps=gaps),
                trace_id=new_trace_id(),
            ),
            query,
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
