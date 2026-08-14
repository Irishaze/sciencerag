"""LLM-based structured extraction for sciencerag.priors (M1-13 v2).

Replaces the keyword-heuristic classifier (classify.py) with a real
extraction pipeline:

  evidence contexts -> numbered evidence table -> prompt -> LLM JSON output
  -> Pydantic validation (retry, feeding the error back to the LLM on
  failure) -> evidence labels resolved back to real DOI/span by our own
  code -> numeric groundedness gate (deterministic, numeric_check.py) ->
  batched semantic support judge (KEEP/REVIEW/DROP, an independent model) ->
  confidence computed by a deterministic formula.

The LLM never outputs a DOI directly (a hallucination vector) — it only
picks which numbered evidence snippets support each prior; we look up the
real source ourselves.

Confidence is NOT the gate that decides whether a prior survives into the
response — that job now belongs to the numeric + semantic checks above.
Confidence is a purely deterministic, source-count/relevance-derived
ranking signal (never LLM-scored, per spec's "don't fake a calibrated
probability" principle — see README's priors section), used only to order
survivors and decide what gets cut when there are more than max_priors.
This split happened after real-data comparison (k/relevance/confidence plan
Part 2, 2026-08-04) found no confidence formula variant — including having
an LLM score it directly — actually separated KEEP from DROP; the signals
confidence is built from measure "how much evidence backs this," not "is
this specific claim actually correct," which is a different question a
formula over source-count/relevance structurally can't answer.
"""

import json
from typing import Any, Literal, NamedTuple

import litellm
from pydantic import BaseModel, Field, ValidationError, model_validator

from sciencerag.common.config import get_llm_model
from sciencerag.priors.contract import GEOMETRY_FREE_NAMES, GEOMETRY_FREE_PARAMS
from sciencerag.priors.models import Prior, SourcePaper
from sciencerag.priors.numeric_check import (
    extract_numbers,
    extract_numbers_from_text,
    find_unmatched_numbers,
)

MAX_RETRIES = 3
# litellm.completion has no default timeout — an unresponsive DeepSeek call
# hangs the whole request forever. Bound it so a stuck call becomes a
# retryable failure instead (spec principle: never hang, never half-fail).
# A real ~10-evidence-snippet prompt (~6k chars) measured at ~58s to
# DeepSeek; 45s was cutting off legitimate in-progress calls. 90s gives
# headroom without waiting forever.
REQUEST_TIMEOUT_SECONDS = 90

def _format_target_params() -> str:
    return "\n".join(
        f"- {p['name']} (unit: {p['unit']}) — {p['desc']}" for p in GEOMETRY_FREE_PARAMS
    )


# Target-oriented extraction (spec §3.6): the LLM no
# longer invents its own `field` slugs. It only ever names one of the sim
# contract's 12 free geometry parameters (sciencerag/priors/sim_params.json),
# so priors line up with the simulation side without a name/unit
# translation step. Enforced in code too, not just the prompt — see
# ExtractedPriorDraft's _fields_must_be_in_contract validator below.
SYSTEM_PROMPT = f"""You are a scientific information extractor for a thermoelectric cooler (TEC) research assistant.

You will be given a research question and a list of numbered evidence snippets, each already tagged with its source. Extract structured "priors" (facts/findings) from the evidence that help answer the question.

TARGET PARAMETERS — this is closed, goal-directed extraction, not open-ended. The ONLY parameters a prior's "field" or "related_fields" may name are these free geometry parameters of the simulation contract (use these exact names — do not invent, translate, or rename them):
{_format_target_params()}

Do NOT extract priors about material properties (Seebeck coefficient, resistivity, thermal conductivity, ZT, etc.) — the material is fixed (Bi2Te3) and its properties are already registered in the simulation contract; priors should never propose values for it. Skip evidence that is purely about material properties.
Do NOT extract priors about operating conditions (ambient temperature, current/voltage set points, air speed, etc.) or derived/numerical settings — these are out of scope for priors.

For each prior, output a JSON object with exactly these fields:
- "kind": one of "parameter_range", "scaling_relationship", "candidate_config", "caution"
  - parameter_range: a SPECIFIC NUMBER for exactly ONE target parameter — an optimal/typical value or a min/max range. If you cannot put a number in `value`, it is NOT parameter_range. Requires "field" set to that one target parameter.
  - scaling_relationship: a relationship BETWEEN target parameters (e.g. "the optimal leg_length depends on leg_width"), WITHOUT necessarily citing a specific number. Leave "field" null and list every target parameter the relationship involves in "related_fields". Only report a relationship the evidence actually states — never presume two parameters are related just because both are geometric.
  - candidate_config: a specific, already-reported combination of several target parameter values (a full design point). Leave "field" null, list every parameter it covers in "related_fields", and give each one's value inside "value" (keyed by the exact parameter name).
  - caution: a limitation, caveat, or warning about applicability of a target parameter — usually set "field" to that one parameter; if it spans several, use "related_fields" instead.
- "field": null, OR the exact name of exactly one target parameter listed above (never a material/operating/derived parameter, never an invented name).
- "related_fields": a list of exact target parameter names this prior relates to (default: empty list). Used for scaling_relationship (exactly two names) and candidate_config (two or more names); leave empty for single-parameter priors.
- "value": a JSON object whose shape is FIXED per kind, given below. There is no free-form "summary" key in ANY kind — if the evidence doesn't give you enough to fill a kind's required fields, don't force it; extract nothing for that finding instead.

  - parameter_range:
    - "field_name" (required, string) — the EXACT SAME name as "field" above.
    - at least one of "min", "max", "typical" (numbers) — in the target parameter's contract unit where the evidence allows.
    - "unit" (required, string) — the unit the number(s) are in; use "dimensionless" if there is none.
    - "conditions" (optional object) — operating conditions the value was reported under, e.g. {{"temperature_k": 300}}. Omit if the evidence doesn't state any.
    Example (plain): {{"field_name": "leg_length", "typical": 0.06, "unit": "mm"}}
    Example (with conditions): {{"field_name": "pitch", "min": 0.02, "max": 0.2, "unit": "mm", "conditions": {{"temperature_k": 300}}}}

  - scaling_relationship:
    - "x", "y" (required, strings) — the two target parameter names from "related_fields", exactly (x's effect on y).
    - "direction" (required) — one of "positive", "negative", "convex", "concave", "non_monotonic", "unknown".
    - "functional_form" (optional, string) — a short description of the shape, e.g. "COP peaks at intermediate leg length".
    - "validity_range" (optional, string) — the range this relationship was reported to hold over, if stated.
    Example: {{"x": "leg_length", "y": "cop", "direction": "convex", "functional_form": "COP peaks at intermediate leg length"}}

  - candidate_config:
    - "parameters" (required object, 2+ entries) — every parameter this design point covers, keyed by the EXACT SAME names as "related_fields", each mapped to its reported value.
    - "reported_performance" (optional object) — any performance metric reported alongside it, e.g. {{"cop": 1.5}}.
    - "context" (optional, string) — short context for the design point.
    Example: {{"parameters": {{"leg_length": 0.07, "leg_width": 0.12}}, "reported_performance": {{"cop": 1.5}}}}

  - caution:
    - "statement" (required, string) — the limitation/caveat itself, as stated by the evidence.
    - "applicability_scope" (optional, string) — the condition(s) under which it applies, if stated.
    Example: {{"statement": "This range assumes ideal contact resistance", "applicability_scope": "no delamination"}}

- "notes": optional short clarifying note, or null
- "evidence": a list of evidence labels (e.g. ["E1", "E3"]) that support this prior — list ALL evidence snippets that support it, not just one

Rules:
- Only extract priors that are actually and explicitly stated in the evidence. Do not use outside knowledge, and do not add your own inferred direction, magnitude, or recommendation beyond what the evidence text literally says — if the evidence says a factor merely "influences" or "is relevant to" something without saying how, report only that, don't guess "increases" or "should be minimized".
- A non-numeric "X affects/influences Y" statement is NEVER parameter_range, even about one parameter — use scaling_relationship instead (with "unknown" direction if the evidence doesn't say which way).
- The value object's own field name(s) — "field_name" for parameter_range, "x"/"y" for scaling_relationship, the keys of "parameters" for candidate_config — must exactly match "field"/"related_fields" above. Never let them disagree, and never include a "summary" key anywhere.
- Every number you write (in "min"/"max"/"typical"/"magnitude"/"parameters"/"reported_performance"/"conditions", and in "notes") must be copied EXACTLY as the evidence states it, in the evidence's OWN unit — do not convert units (e.g. don't turn evidence's "20 µm" into "0.02 mm") and do not recompute or round beyond what's needed to copy the digits. A converted or recomputed number will fail downstream verification even if the physical quantity is correct.
- Only reference evidence labels that appear in the input. Never invent a label.
- Do not split one finding into near-duplicate priors across different evidence — merge them and cite all supporting evidence together instead.
- Some evidence snippets may be a summary of a paper's reference list or acknowledgments rather than its own findings (phrases like "the references suggest...", "Reference N examines..."). Treat these as weak, secondary support only — never as the sole evidence for a prior.
- Do not presuppose which target parameters are grouped or related — only report a scaling_relationship or candidate_config grouping when the evidence itself actually ties those parameters together.
- If a target parameter has no supporting evidence, simply produce no prior for it — do not force or guess a value just to cover it.
- Output ONLY a JSON object of the form {{"priors": [...]}}. No explanation, no markdown fences.
"""


class ExtractedPriorDraft(BaseModel):
    kind: Literal[
        "parameter_range",
        "material_property",
        "scaling_relationship",
        "candidate_config",
        "caution",
    ]
    field: str | None = None
    # New in the sim-contract sync (spec §3.6): relationships/configs spanning
    # more than one parameter (scaling_relationship/candidate_config) can't
    # be expressed with a single `field`. Optional + defaults to [] so
    # single-parameter priors are unaffected — backward compatible.
    related_fields: list[str] = Field(default_factory=list)
    value: dict[str, Any]
    notes: str | None = None
    evidence: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _parameter_range_requires_a_number(self) -> "ExtractedPriorDraft":
        """Don't trust the prompt alone to keep parameter_range numeric —
        enforce it in code so a violation is a validation failure (retried),
        not a silently-accepted empty-hearted prior. Found via manual review:
        the LLM was dumping non-numeric "X affects Y" statements into
        parameter_range instead of scaling_relationship."""
        if self.kind == "parameter_range":
            if self.field is None:
                raise ValueError(
                    "kind='parameter_range' requires a non-null `field` naming exactly "
                    "one contract geometry parameter"
                )
            has_numeric = any(
                isinstance(v, int | float) and not isinstance(v, bool)
                for v in self.value.values()
            )
            if not has_numeric:
                raise ValueError(
                    f"kind='parameter_range' (field={self.field!r}) requires at least one "
                    "numeric value (e.g. min/max/typical) in `value`; if this is a "
                    "non-numeric 'X affects Y' statement, use kind='scaling_relationship' "
                    "instead, with a 'direction' key in value"
                )
        return self

    @model_validator(mode="after")
    def _fields_must_be_in_contract(self) -> "ExtractedPriorDraft":
        """Hard constraint from spec §3.6: field/related_fields must be exact
        sim_params.json geometry_free names — never an LLM-invented slug,
        never a material/operating/derived parameter. material_property
        drafts are exempt (and always silently dropped downstream in
        extract_priors, never becoming a Prior — see spec §3.6: material is
        fixed, prior_target=false, this kind is schema-only)."""
        if self.kind == "material_property":
            return self
        names = ([self.field] if self.field is not None else []) + self.related_fields
        unknown = [n for n in names if n not in GEOMETRY_FREE_NAMES]
        if unknown:
            raise ValueError(
                f"field/related_fields must be exact sim_params.json geometry_free "
                f"names; got unrecognized name(s) {unknown!r} "
                f"(allowed: {sorted(GEOMETRY_FREE_NAMES)})"
            )
        return self


class ExtractionOutput(BaseModel):
    priors: list[ExtractedPriorDraft]


class EvidenceItem(BaseModel):
    text: str
    doi: str | None
    span: str
    notes: str | None
    relevance: float  # 0-1, from PaperQA2's context.score / 10


class ExtractionError(Exception):
    """LLM failed to produce valid, grounded JSON after MAX_RETRIES attempts."""


class PipelineAttempt(BaseModel):
    attempt: int
    raw_output: str
    error: str | None = None


class ConfidenceBreakdown(BaseModel):
    prior_field: str | None
    evidence_labels: list[str]
    n_papers: int
    n_snippets: int
    base: float
    avg_relevance: float
    confidence: float


class PipelineTrace(BaseModel):
    """Captures every stage of the M1-13 pipeline for demo/debug display.

    Not part of the spec-compliant PriorsResponse — this is extra, only
    populated when a caller explicitly asks for it (see /sciencerag/priors/_debug).
    """

    query: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    evidence_labels: list[str] = Field(default_factory=list)
    system_prompt: str = ""
    user_prompt: str = ""
    attempts: list[PipelineAttempt] = Field(default_factory=list)
    confidence_breakdown: list[ConfidenceBreakdown] = Field(default_factory=list)
    # Every PaperQA2 context, converted to an EvidenceItem (text/doi/span/
    # relevance) BEFORE MIN_EVIDENCE_RELEVANCE filtering drops any of them
    # (see retrieval.py's _build_evidence_table) — unlike `evidence` above,
    # which only holds the survivors. Needed to see what the threshold
    # discards, and with what content (not just a bare score).
    all_evidence: list[EvidenceItem] = Field(default_factory=list)
    # Every prior that survived numeric + semantic checks (i.e. KEEP-verdict
    # only — REVIEW/DROP never reach here) — full kind/field/value/notes/
    # sources, not just the numbers in confidence_breakdown. Confidence no
    # longer gates what's in here (see module docstring); this is now
    # exactly what `priors` in the final response contains, before
    # max_priors capping.
    all_priors: list[Prior] = Field(default_factory=list)
    # Phase B3: every numeric-groundedness rejection (see _to_prior), one
    # entry per unmatched number, format "{field}: {number} not found in
    # {evidence labels}" — populated even for drafts that ultimately get
    # accepted on a later retry attempt, so a false-positive rejection is
    # visible in trace even though it didn't affect the final response.
    numeric_check_failures: list[str] = Field(default_factory=list)
    # Semantic support judge (see _judge_semantic_support): one entry per
    # DROP verdict, format "{field}: {reason}" — same "visible even if a
    # later retry attempt fixes it" semantics as numeric_check_failures.
    semantic_check_failures: list[str] = Field(default_factory=list)
    # One entry per REVIEW verdict (plausible-but-not-literal support, e.g.
    # a standard domain-knowledge translation the evidence doesn't spell
    # out verbatim) — these priors are excluded from the final response but
    # aren't "wrong" the way a DROP is, so tracked separately for visibility.
    semantic_reviews: list[str] = Field(default_factory=list)
    # Knowledge-graph lookup (spec §3.2: KG first, always additive to
    # literature, never a substitute) — added 2026-08-13 because this step
    # runs entirely before extract_priors() and so was previously invisible
    # in every trace-based demo/debug view, even though its output ends up
    # merged into the final response's `priors` (retrieval.py's
    # _build_priors_response). No LLM involved: deterministic keyword
    # overlap over the graph (jieba-segmented for CJK query text).
    kg_total_matching_entities: int = 0
    kg_entities_returned: int = 0
    kg_priors: list[Prior] = Field(default_factory=list)
    # The English query actually sent to PaperQA2/Semantic Scholar/arXiv
    # when `query` contains CJK text (see retrieval.py's
    # _translate_for_literature_search) — equal to `query` itself
    # otherwise. Surfaced so a demo/debug view can show *why* a Chinese
    # question got noticeably faster/better evidence after 2026-08-13,
    # instead of the translation happening invisibly.
    literature_query: str = ""


def _build_evidence_block(evidence_table: dict[str, EvidenceItem]) -> str:
    lines = [
        f'[{label}] (source: {item.doi or "unknown"}, {item.span}) "{item.text}"'
        for label, item in evidence_table.items()
    ]
    return "\n".join(lines)


def _strip_code_fences(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    return cleaned.strip()


def _parse_and_validate(
    raw: str, evidence_table: dict[str, EvidenceItem]
) -> ExtractionOutput:
    data = json.loads(_strip_code_fences(raw))
    output = ExtractionOutput.model_validate(data)

    unknown = {
        label
        for prior in output.priors
        for label in prior.evidence
        if label not in evidence_table
    }
    if unknown:
        raise ValueError(f"referenced unknown evidence labels: {sorted(unknown)}")

    return output


# Confidence formula constants (spec §3.7 — pending real-data recalibration
# under the sim-contract-scoped pipeline; current values are placeholders
# carried over from the pre-fix formula's base/cap, not yet re-derived).
# Two signals, weighted unevenly on purpose:
#   - n_papers (distinct DOIs): the dominant term. Independent papers
#     agreeing is real corroboration.
#   - extra_snippets (evidence items beyond one-per-paper, i.e. PaperQA2
#     chunking the SAME paper's passage into multiple overlapping
#     contexts): a much smaller bonus. More text from a paper you already
#     have is weaker signal than a second paper, but it's not zero signal
#     either — a paper repeating/elaborating a claim across a table and a
#     paragraph is mildly more trustworthy than a single passing mention.
CONFIDENCE_BASE = 0.5
CONFIDENCE_PER_PAPER = 0.1
CONFIDENCE_PAPER_CAP = 3
CONFIDENCE_PER_EXTRA_SNIPPET = 0.02
CONFIDENCE_EXTRA_SNIPPET_CAP = 3


def _to_prior(
    draft: ExtractedPriorDraft,
    evidence_table: dict[str, EvidenceItem],
    trace: PipelineTrace | None = None,
) -> Prior:
    items = [evidence_table[label] for label in draft.evidence]
    sources = [SourcePaper(doi=item.doi or "", span=item.span) for item in items]

    # Distinct papers vs. total snippets: PaperQA2 often chunks one paper's
    # one passage into several overlapping contexts, each becoming its own
    # evidence label (E1/E2/E3...) — citing 3 snippets from the SAME paper
    # is not 3 independent corroborating sources, it's 1 source cited 3
    # times, and shouldn't get the same confidence boost as 3 distinct
    # papers agreeing. A missing DOI never merges with another missing DOI
    # (each gets its own unique fallback key) so unknown-source items can't
    # silently absorb into someone else's citation count.
    n_snippets = len(items)
    distinct_papers = {item.doi if item.doi else f"_nodoi_{id(item)}" for item in items}
    n_papers = len(distinct_papers)
    extra_snippets = max(0, n_snippets - n_papers)

    base = (
        CONFIDENCE_BASE
        + CONFIDENCE_PER_PAPER * min(n_papers, CONFIDENCE_PAPER_CAP)
        + CONFIDENCE_PER_EXTRA_SNIPPET * min(extra_snippets, CONFIDENCE_EXTRA_SNIPPET_CAP)
    )
    avg_relevance = sum(item.relevance for item in items) / n_snippets
    confidence = round(min(1.0, base * avg_relevance), 2)

    if trace is not None:
        trace.confidence_breakdown.append(
            ConfidenceBreakdown(
                prior_field=draft.field,
                evidence_labels=draft.evidence,
                n_papers=n_papers,
                n_snippets=n_snippets,
                base=round(base, 2),
                avg_relevance=round(avg_relevance, 2),
                confidence=confidence,
            )
        )

    id_suffix = draft.field or "_".join(draft.related_fields) or draft.kind
    prior_id = f"pr_{'_'.join(draft.evidence)}_{id_suffix}"[:64]
    prior = Prior(
        prior_id=prior_id,
        kind=draft.kind,
        field=draft.field,
        related_fields=draft.related_fields,
        value=draft.value,
        confidence=confidence,
        sources=sources,
        notes=draft.notes or items[0].notes,
    )

    # Phase B3: numeric groundedness gate. Every number the prior asserts
    # (numeric_check.extract_numbers — value's numeric fields + notes) must
    # be found, exact or within numeric_check's rounding tolerance,
    # somewhere in the combined text of the evidence it cites. A number
    # with no match is treated as fabricated, not a rounding difference
    # (the matcher already tolerates those) — reject the whole prior and
    # let extract_priors' existing retry loop feed the failure back to the
    # LLM, the same path any other validation failure already takes.
    evidence_text = "\n\n".join(item.text for item in items)
    evidence_numbers = extract_numbers_from_text(evidence_text)
    unmatched = find_unmatched_numbers(extract_numbers(prior), evidence_numbers)
    if unmatched:
        label = draft.field or "/".join(draft.related_fields) or draft.kind
        evidence_labels = ",".join(draft.evidence)
        failures = [f"{label}: {n:g} not found in {evidence_labels}" for n in unmatched]
        if trace is not None:
            trace.numeric_check_failures.extend(failures)
        raise ValueError("numeric groundedness check failed: " + "; ".join(failures))

    return prior


# Deliberately NOT get_llm_model()/SCIENCERAG_LLM_MODEL: this judge exists to
# check the production model's own output, so it must be a different model
# than whatever's currently doing extraction — judging output with the same
# model that produced it is grading its own homework (see module docstring).
# Picked via Phase C's judge shootout (data/judge_shootout_results.json) and
# cross-validated against a real independent GPT-4o judge on 22 real
# production priors (2026-08-04, k/relevance/confidence plan Part 2):
# 81.8% agreement, caught 2 real hallucinations GPT-4o itself missed, with
# its disagreements concentrated in exactly the "standard domain-knowledge
# translation, not literal evidence wording" pattern the REVIEW tier below
# is meant to absorb instead of forcing a binary call.
SEMANTIC_JUDGE_MODEL = "gpt-5.6-luna"

SEMANTIC_SUPPORT_RUBRIC = """You are a strict quality judge for structured scientific priors extracted
from TEC (thermoelectric cooler) literature. You verify each prior against
its OWN cited evidence snippets only — other snippets in the evidence block
may be irrelevant to a given prior.

You will be given a research query, a set of labeled evidence snippets, and
a list of drafted priors — each with its own id, kind, field/related_fields,
value, notes, and which evidence labels it cites.

For EACH prior, judge SUPPORTED, then USEFUL:

SUPPORTED — three levels:
- KEEP-level: every claim/number the prior makes is directly and literally
  stated in its cited evidence text.
- REVIEW-level: the prior applies a standard, textbook domain translation
  from the evidence's own wording onto our fixed parameter names (example:
  evidence says "high aspect ratio", prior maps that to leg_length/
  leg_width — aspect ratio IS length/width, a definitional mapping, not an
  invented claim). The underlying finding is real and grounded, just not
  phrased with the exact parameter names in the source text.
- DROP-level: the prior asserts something its cited evidence does not
  state or contradicts, invents a causal relationship the evidence never
  makes, or cites evidence about a different topic/device than the prior's
  claim (e.g. citing a thermoelectric GENERATOR study to support a
  thermoelectric COOLER claim), or states a number absent from the
  evidence text.

USEFUL (only matters if SUPPORTED is KEEP-level or REVIEW-level): the prior
must carry actionable information (a value, range, direction, configuration,
or concrete caveat) — a vague restatement with no real content is not
useful regardless of how well "supported" it looks.

Final verdict per prior:
- "KEEP": SUPPORTED at KEEP-level AND USEFUL.
- "REVIEW": SUPPORTED at REVIEW-level (reasonable inference, not literal)
  AND USEFUL — genuinely uncertain, not wrong.
- "DROP": SUPPORTED is DROP-level, OR not USEFUL.

`notes` is the extracting LLM's own free clarifying remark — context only.
Do NOT treat it as a claim requiring its own verification, and do NOT let
information found only in `notes` (not in the prior's structured `value`)
count as grounding for anything.

Output JSON only, no markdown fences:
{"verdicts": [
  {"prior_id": "<id>", "verdict": "KEEP" | "REVIEW" | "DROP", "reason": "<one sentence>"}
]}
One entry per prior given, in any order."""


class SemanticVerdict(NamedTuple):
    verdict: Literal["KEEP", "REVIEW", "DROP"]
    reason: str


class ReviewedPrior(NamedTuple):
    prior: Prior
    reason: str


def _evidence_labels_for(prior: Prior, evidence_table: dict[str, EvidenceItem]) -> list[str]:
    """Prior only keeps resolved doi/span in `sources`, not the E-labels
    used during extraction — reconstruct them by matching back into
    evidence_table, same approach used (and validated against real data) in
    scripts/confidence_formula_probe.py."""
    return [
        label
        for label, item in evidence_table.items()
        if any((item.doi or "") == s.doi and item.span == s.span for s in prior.sources)
    ]


def _judge_semantic_support(
    query: str,
    priors: list[Prior],
    evidence_table: dict[str, EvidenceItem],
) -> dict[str, SemanticVerdict]:
    """One batched call (not one per prior — spec §8's per-query latency
    budget) that judges every drafted prior's SUPPORTED+USEFUL status at
    once, returning a KEEP/REVIEW/DROP verdict per prior_id. Any failure
    (API error, malformed output, a missing verdict) is raised as
    ValueError so it flows into extract_priors' existing retry path exactly
    like a numeric-groundedness or schema-validation failure — a transient
    judge hiccup shouldn't crash a request that already extracted validly.
    """
    prior_descs = []
    for p in priors:
        labels = _evidence_labels_for(p, evidence_table)
        prior_descs.append(
            f"- id: {p.prior_id}\n"
            f"  kind: {p.kind}\n"
            f"  field: {p.field}\n"
            f"  related_fields: {p.related_fields}\n"
            f"  value: {json.dumps(p.value.model_dump())}\n"
            f"  notes: {p.notes}\n"
            f"  cites: {', '.join(labels) if labels else '(no evidence label matched)'}"
        )
    user_prompt = (
        f"Query: {query}\n\n"
        f"Evidence:\n{_build_evidence_block(evidence_table)}\n\n"
        f"Priors to judge:\n" + "\n".join(prior_descs)
    )
    messages = [
        {"role": "system", "content": SEMANTIC_SUPPORT_RUBRIC},
        {"role": "user", "content": user_prompt},
    ]
    try:
        try:
            response = litellm.completion(
                model=SEMANTIC_JUDGE_MODEL,
                messages=messages,
                temperature=0,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except litellm.BadRequestError as e:
            if "temperature" not in str(e):
                raise
            response = litellm.completion(
                model=SEMANTIC_JUDGE_MODEL, messages=messages, timeout=REQUEST_TIMEOUT_SECONDS
            )
        raw = response.choices[0].message.content
        parsed = json.loads(_strip_code_fences(raw))
        verdicts = {
            v["prior_id"]: SemanticVerdict(verdict=v["verdict"], reason=v.get("reason", ""))
            for v in parsed["verdicts"]
        }
    except Exception as e:  # noqa: BLE001 - any failure here retries the whole attempt, same as any other validation failure, rather than crashing the request
        raise ValueError(f"semantic support judge failed: {type(e).__name__}: {e}") from e

    missing = [p.prior_id for p in priors if p.prior_id not in verdicts]
    if missing:
        raise ValueError(f"semantic support judge omitted verdicts for: {missing}")
    return verdicts


class ExtractionResult(NamedTuple):
    priors: list[Prior]
    # Count of material_property drafts the LLM emitted anyway (against the
    # prompt's instructions) and that were filtered out before becoming a
    # Prior — surfaced so the caller can note it in coverage.gaps rather
    # than have it vanish with no trace (spec §3.6).
    filtered_material_count: int
    # Priors the semantic judge marked REVIEW (plausible domain-standard
    # inference, not literal evidence wording) — excluded from `priors` but
    # not discarded silently; the caller surfaces these in coverage.gaps
    # rather than lumping them in with "confidence too low" or "not found".
    # No default — NamedTuple defaults are shared mutable state across every
    # instance that doesn't override them, and this always has a real value
    # to pass (see extract_priors' single construction site).
    review_priors: list[ReviewedPrior]


def extract_priors(
    query: str,
    evidence_table: dict[str, EvidenceItem],
    trace: PipelineTrace | None = None,
) -> ExtractionResult:
    """Run the extraction pipeline; raises ExtractionError if the LLM never
    produces valid, grounded JSON within MAX_RETRIES attempts.

    Pass a PipelineTrace() to capture every stage (evidence table, prompts,
    raw LLM output per attempt, confidence math) for demo/debug display —
    see sciencerag/priors/router.py's /_debug endpoint.
    """
    user_prompt = f"Question: {query}\n\nEvidence:\n{_build_evidence_block(evidence_table)}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    if trace is not None:
        trace.query = query
        trace.evidence = list(evidence_table.values())
        trace.evidence_labels = list(evidence_table.keys())
        trace.system_prompt = SYSTEM_PROMPT
        trace.user_prompt = user_prompt

    last_error: Exception | None = None
    for attempt_num in range(1, MAX_RETRIES + 1):
        try:
            model = get_llm_model()
            try:
                response = litellm.completion(
                    model=model,
                    messages=messages,
                    temperature=0,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
            except litellm.BadRequestError as e:
                # Some models (OpenAI's o-series, the gpt-5.6 family) reject
                # a custom temperature outright — "Only the default (1)
                # value is supported" — not a real extraction failure, so
                # fall back to the model's default rather than burning a
                # retry (and the JSON-consistency benefit of temperature=0)
                # on every single call to one of these models.
                if "temperature" not in str(e):
                    raise
                response = litellm.completion(
                    model=model, messages=messages, timeout=REQUEST_TIMEOUT_SECONDS
                )
            raw = response.choices[0].message.content
        except Exception as e:  # network error, timeout, rate limit, etc.
            last_error = e
            if trace is not None:
                trace.attempts.append(
                    PipelineAttempt(attempt=attempt_num, raw_output="", error=f"LLM call failed: {e}")
                )
            continue  # retry with the same messages, no assistant turn to append

        try:
            output = _parse_and_validate(raw, evidence_table)
            # Material is fixed (Bi2Te3, prior_target=false) — material_property
            # stays a valid `kind` for schema compatibility (spec §3.6) but the
            # pipeline never produces it: filter any the LLM emits anyway,
            # rather than retrying or erroring over them.
            kept = [d for d in output.priors if d.kind != "material_property"]
            filtered_material_count = len(output.priors) - len(kept)
            priors = [_to_prior(draft, evidence_table, trace) for draft in kept]

            # Semantic support judge: runs once per attempt, batched over
            # every draft that already passed schema + numeric groundedness.
            # A DROP verdict fails the WHOLE attempt (same as a numeric or
            # schema failure below) — retrying re-drafts everything, not
            # just the bad one, but that's the existing retry granularity
            # this whole loop already has (see numeric groundedness above),
            # not something new introduced here. A REVIEW verdict does NOT
            # fail the attempt: the evidence isn't going to get more literal
            # on another retry, so there's nothing a retry would fix — it's
            # excluded from `priors` directly and reported via review_priors.
            review_priors: list[ReviewedPrior] = []
            if priors:
                verdicts = _judge_semantic_support(query, priors, evidence_table)
                dropped = [p for p in priors if verdicts[p.prior_id].verdict == "DROP"]
                if dropped:
                    reasons = [
                        f"{p.field or '/'.join(p.related_fields) or p.kind}: "
                        f"{verdicts[p.prior_id].reason}"
                        for p in dropped
                    ]
                    if trace is not None:
                        trace.semantic_check_failures.extend(reasons)
                    raise ValueError("semantic support check failed: " + "; ".join(reasons))

                review_priors = [
                    ReviewedPrior(prior=p, reason=verdicts[p.prior_id].reason)
                    for p in priors
                    if verdicts[p.prior_id].verdict == "REVIEW"
                ]
                if trace is not None:
                    trace.semantic_reviews.extend(
                        f"{rp.prior.field or '/'.join(rp.prior.related_fields) or rp.prior.kind}: "
                        f"{rp.reason}"
                        for rp in review_priors
                    )
                review_ids = {rp.prior.prior_id for rp in review_priors}
                priors = [p for p in priors if p.prior_id not in review_ids]
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            # _to_prior (numeric groundedness gate included) and the semantic
            # judge both run inside this try too, on purpose — a single
            # attempt/trace entry per loop iteration whether the failure
            # came from parsing, per-kind schema validation, numeric
            # groundedness, or semantic support. Recording success before
            # calling these used to double up trace.attempts (one "success"
            # entry already appended, then this except block's own entry for
            # the same attempt_num) on any downstream failure — silent since
            # these checks essentially never failed before they existed.
            last_error = e
            if trace is not None:
                trace.attempts.append(
                    PipelineAttempt(attempt=attempt_num, raw_output=raw, error=str(e))
                )
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": f"Your last output failed validation: {e}. "
                    "Fix it and output ONLY the corrected JSON.",
                }
            )
            continue

        if trace is not None:
            trace.attempts.append(PipelineAttempt(attempt=attempt_num, raw_output=raw))
        return ExtractionResult(priors, filtered_material_count, review_priors)

    raise ExtractionError(f"failed after {MAX_RETRIES} attempts: {last_error}")
