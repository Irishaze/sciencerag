"""LLM-based structured extraction for sciencerag.priors (M1-13 v2).

Replaces the keyword-heuristic classifier (classify.py) with a real
extraction pipeline:

  evidence contexts -> numbered evidence table -> prompt -> LLM JSON output
  -> Pydantic validation (retry, feeding the error back to the LLM on
  failure) -> evidence labels resolved back to real DOI/span by our own
  code -> confidence computed by a deterministic formula.

The LLM never outputs a DOI directly (a hallucination vector) — it only
picks which numbered evidence snippets support each prior; we look up the
real source ourselves. Confidence is not LLM-scored; it's derived from
evidence count + PaperQA2's own relevance score, per spec's "don't fake a
calibrated probability" principle (see README's priors section).
"""

import json
from typing import Any, Literal, NamedTuple

import litellm
from pydantic import BaseModel, Field, ValidationError, model_validator

from sciencerag.common.config import get_llm_model
from sciencerag.priors.contract import GEOMETRY_FREE_NAMES, GEOMETRY_FREE_PARAMS
from sciencerag.priors.models import Prior, SourcePaper

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
- "related_fields": a list of exact target parameter names this prior relates to (default: empty list). Used for scaling_relationship and candidate_config; leave empty for single-parameter priors.
- "value": a JSON object holding the actual content.
  - For parameter_range: MUST include at least one numeric key, e.g. {{"min": 0.02, "max": 0.2, "unit": "mm"}} or {{"typical": 0.06, "unit": "mm"}}. Match the target parameter's contract unit where the evidence allows.
  - For scaling_relationship: include a "direction" key with one of "positive", "negative", "convex", "unknown", plus a "summary" explaining the relationship.
  - For other kinds: use structured keys where possible, else {{"summary": "..."}}.
- "notes": optional short clarifying note, or null
- "evidence": a list of evidence labels (e.g. ["E1", "E3"]) that support this prior — list ALL evidence snippets that support it, not just one

Rules:
- Only extract priors that are actually and explicitly stated in the evidence. Do not use outside knowledge, and do not add your own inferred direction, magnitude, or recommendation beyond what the evidence text literally says — if the evidence says a factor merely "influences" or "is relevant to" something without saying how, report only that, don't guess "increases" or "should be minimized".
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
    # Every extracted Prior BEFORE CONFIDENCE_THRESHOLD splits them into
    # strong/weak (see retrieval.py's _split_by_confidence) — full kind/
    # field/value/notes/sources, not just the numbers in confidence_breakdown.
    # Needed to judge whether a low-confidence prior was correctly discarded.
    all_priors: list[Prior] = Field(default_factory=list)


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


def _to_prior(
    draft: ExtractedPriorDraft,
    evidence_table: dict[str, EvidenceItem],
    trace: PipelineTrace | None = None,
) -> Prior:
    items = [evidence_table[label] for label in draft.evidence]
    sources = [SourcePaper(doi=item.doi or "", span=item.span) for item in items]

    base = 0.5 + 0.1 * min(len(items), 3)
    avg_relevance = sum(item.relevance for item in items) / len(items)
    confidence = round(min(1.0, base * avg_relevance), 2)

    if trace is not None:
        trace.confidence_breakdown.append(
            ConfidenceBreakdown(
                prior_field=draft.field,
                evidence_labels=draft.evidence,
                base=round(base, 2),
                avg_relevance=round(avg_relevance, 2),
                confidence=confidence,
            )
        )

    id_suffix = draft.field or "_".join(draft.related_fields) or draft.kind
    prior_id = f"pr_{'_'.join(draft.evidence)}_{id_suffix}"[:64]
    return Prior(
        prior_id=prior_id,
        kind=draft.kind,
        field=draft.field,
        related_fields=draft.related_fields,
        value=draft.value,
        confidence=confidence,
        sources=sources,
        notes=draft.notes or items[0].notes,
    )


class ExtractionResult(NamedTuple):
    priors: list[Prior]
    # Count of material_property drafts the LLM emitted anyway (against the
    # prompt's instructions) and that were filtered out before becoming a
    # Prior — surfaced so the caller can note it in coverage.gaps rather
    # than have it vanish with no trace (spec §3.6).
    filtered_material_count: int


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
            response = litellm.completion(
                model=get_llm_model(),
                messages=messages,
                temperature=0,
                timeout=REQUEST_TIMEOUT_SECONDS,
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
            if trace is not None:
                trace.attempts.append(PipelineAttempt(attempt=attempt_num, raw_output=raw))
            # Material is fixed (Bi2Te3, prior_target=false) — material_property
            # stays a valid `kind` for schema compatibility (spec §3.6) but the
            # pipeline never produces it: filter any the LLM emits anyway,
            # rather than retrying or erroring over them.
            kept = [d for d in output.priors if d.kind != "material_property"]
            filtered_material_count = len(output.priors) - len(kept)
            priors = [_to_prior(draft, evidence_table, trace) for draft in kept]
            return ExtractionResult(priors, filtered_material_count)
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
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

    raise ExtractionError(f"failed after {MAX_RETRIES} attempts: {last_error}")
