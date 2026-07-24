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
from typing import Any, Literal

import litellm
from pydantic import BaseModel, Field, ValidationError

from sciencerag.common.config import get_llm_model
from sciencerag.priors.models import Prior, SourcePaper

MAX_RETRIES = 3

SYSTEM_PROMPT = """You are a scientific information extractor for a thermoelectric cooler (TEC) research assistant.

You will be given a research question and a list of numbered evidence snippets, each already tagged with its source. Extract structured "priors" (facts/findings) from the evidence that help answer the question.

For each prior, output a JSON object with exactly these fields:
- "kind": one of "parameter_range", "material_property", "scaling_relationship", "candidate_config", "caution"
  - parameter_range: a specific numeric operating parameter, or its optimal/typical value or range
  - material_property: an intrinsic material property (Seebeck coefficient, resistivity, thermal conductivity, etc.)
  - scaling_relationship: how one quantity varies as a function of another (proportional, convex, correlated, etc.)
  - candidate_config: a specific design/configuration choice or method (geometry, driving method, structure)
  - caution: a limitation, caveat, or warning about applicability
- "field": a short snake_case slug naming what this prior is about (e.g. "seebeck_coefficient", "driving_voltage")
- "value": a JSON object holding the actual content — use structured keys where possible (e.g. {"min": 20, "max": 200, "unit": "um"}), or {"summary": "..."} if unstructured
- "notes": optional short clarifying note, or null
- "evidence": a list of evidence labels (e.g. ["E1", "E3"]) that support this prior — list ALL evidence snippets that support it, not just one

Rules:
- Only extract priors that are actually stated in the evidence. Do not use outside knowledge.
- Only reference evidence labels that appear in the input. Never invent a label.
- Do not split one finding into near-duplicate priors across different evidence — merge them and cite all supporting evidence together instead.
- Output ONLY a JSON object of the form {"priors": [...]}. No explanation, no markdown fences.
"""


class ExtractedPriorDraft(BaseModel):
    kind: Literal[
        "parameter_range",
        "material_property",
        "scaling_relationship",
        "candidate_config",
        "caution",
    ]
    field: str
    value: dict[str, Any]
    notes: str | None = None
    evidence: list[str] = Field(min_length=1)


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


def _to_prior(draft: ExtractedPriorDraft, evidence_table: dict[str, EvidenceItem]) -> Prior:
    items = [evidence_table[label] for label in draft.evidence]
    sources = [SourcePaper(doi=item.doi or "", span=item.span) for item in items]

    base = 0.5 + 0.1 * min(len(items), 3)
    avg_relevance = sum(item.relevance for item in items) / len(items)
    confidence = round(min(1.0, base * avg_relevance), 2)

    prior_id = f"pr_{'_'.join(draft.evidence)}_{draft.field}"[:64]
    return Prior(
        prior_id=prior_id,
        kind=draft.kind,
        field=draft.field,
        value=draft.value,
        confidence=confidence,
        sources=sources,
        notes=draft.notes or items[0].notes,
    )


def extract_priors(query: str, evidence_table: dict[str, EvidenceItem]) -> list[Prior]:
    """Run the extraction pipeline; raises ExtractionError if the LLM never
    produces valid, grounded JSON within MAX_RETRIES attempts."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Question: {query}\n\nEvidence:\n{_build_evidence_block(evidence_table)}",
        },
    ]

    last_error: Exception | None = None
    for _attempt in range(MAX_RETRIES):
        response = litellm.completion(model=get_llm_model(), messages=messages, temperature=0)
        raw = response.choices[0].message.content

        try:
            output = _parse_and_validate(raw, evidence_table)
            return [_to_prior(draft, evidence_table) for draft in output.priors]
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            last_error = e
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": f"Your last output failed validation: {e}. "
                    "Fix it and output ONLY the corrected JSON.",
                }
            )

    raise ExtractionError(f"failed after {MAX_RETRIES} attempts: {last_error}")
