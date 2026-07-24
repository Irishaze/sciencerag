"""Tests for the LLM extraction pipeline (sciencerag/priors/extract.py).

litellm.completion is monkeypatched — no real API calls, no cost.
"""

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from sciencerag.priors import extract as extract_mod
from sciencerag.priors.extract import (
    EvidenceItem,
    ExtractedPriorDraft,
    ExtractionError,
    _build_evidence_block,
    _parse_and_validate,
    _to_prior,
    extract_priors,
)


def _fake_llm_response(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _evidence_table():
    return {
        "E1": EvidenceItem(
            text="Seebeck coefficient of 200 uV/K was measured.",
            doi="10.1234/example",
            span="p.4, Fig.3",
            notes="Example Paper",
            relevance=0.9,
        ),
        "E2": EvidenceItem(
            text="COP peaks at 60um leg length.",
            doi="10.5678/other",
            span="p.7",
            notes="Other Paper",
            relevance=0.6,
        ),
    }


def test_build_evidence_block_includes_label_source_and_text():
    block = _build_evidence_block(_evidence_table())
    assert "[E1] (source: 10.1234/example, p.4, Fig.3)" in block
    assert "Seebeck coefficient" in block
    assert "[E2]" in block


def test_parse_and_validate_accepts_well_formed_json():
    raw = json.dumps(
        {
            "priors": [
                {
                    "kind": "material_property",
                    "field": "seebeck_coefficient",
                    "value": {"typical_uV_per_K": 200},
                    "notes": None,
                    "evidence": ["E1"],
                }
            ]
        }
    )
    output = _parse_and_validate(raw, _evidence_table())
    assert len(output.priors) == 1
    assert output.priors[0].kind == "material_property"


def test_parse_and_validate_strips_markdown_fences():
    raw = "```json\n" + json.dumps({"priors": []}) + "\n```"
    output = _parse_and_validate(raw, _evidence_table())
    assert output.priors == []


def test_parse_and_validate_rejects_unknown_evidence_label():
    raw = json.dumps(
        {
            "priors": [
                {
                    "kind": "parameter_range",
                    "field": "x",
                    "value": {"typical": 42},
                    "evidence": ["E99"],
                }
            ]
        }
    )
    with pytest.raises(ValueError, match="unknown evidence labels"):
        _parse_and_validate(raw, _evidence_table())


def test_parse_and_validate_rejects_malformed_json():
    with pytest.raises(json.JSONDecodeError):
        _parse_and_validate("not json at all", _evidence_table())


def test_parameter_range_without_numeric_value_is_rejected():
    """Found via manual review: the LLM was dumping non-numeric "X affects Y"
    statements into parameter_range instead of scaling_relationship, and
    the resulting priors had empty {"summary": "..."} value with nothing a
    downstream consumer could actually use as a range. Enforce it in the
    model, not just the prompt."""
    with pytest.raises(ValidationError, match="requires at least one numeric value"):
        ExtractedPriorDraft(
            kind="parameter_range", field="driving_voltage", value={"summary": "affects COP"}, evidence=["E1"]
        )


def test_parameter_range_with_numeric_value_is_accepted():
    draft = ExtractedPriorDraft(
        kind="parameter_range", field="driving_voltage", value={"typical": 2.3, "unit": "V"}, evidence=["E1"]
    )
    assert draft.value["typical"] == 2.3


def test_non_parameter_range_kinds_do_not_require_numeric_value():
    draft = ExtractedPriorDraft(
        kind="scaling_relationship",
        field="voltage_vs_cop",
        value={"summary": "COP increases with voltage", "direction": "positive"},
        evidence=["E1"],
    )
    assert draft.kind == "scaling_relationship"


def test_parse_and_validate_rejects_invalid_kind_value():
    """A misspelled/invented kind (e.g. "param-range" instead of
    "parameter_range") must be rejected, not silently accepted — kind is a
    closed set (Literal) precisely so downstream if/else branches on it
    can't silently miss a value."""
    raw = json.dumps(
        {
            "priors": [
                {
                    "kind": "param-range",  # not a valid kind
                    "field": "x",
                    "value": {},
                    "evidence": ["E1"],
                }
            ]
        }
    )
    with pytest.raises(ValidationError):
        _parse_and_validate(raw, _evidence_table())


def test_to_prior_maps_evidence_to_real_sources_and_computes_confidence():
    table = _evidence_table()
    raw = json.dumps(
        {
            "kind": "material_property",
            "field": "seebeck_coefficient",
            "value": {"typical_uV_per_K": 200},
            "notes": None,
            "evidence": ["E1", "E2"],
        }
    )
    draft = ExtractedPriorDraft.model_validate_json(raw)
    prior = _to_prior(draft, table)

    assert [s.doi for s in prior.sources] == ["10.1234/example", "10.5678/other"]
    # base = 0.5 + 0.1*2 = 0.7; avg_relevance = (0.9+0.6)/2 = 0.75 -> 0.525
    # Python's round() uses round-half-to-even on the actual float, giving 0.52.
    assert prior.confidence == 0.52


def test_confidence_increases_with_more_supporting_evidence():
    """Holding relevance constant, more supporting evidence should yield a
    strictly higher confidence — the point of the base term scaling with
    min(n_sources, 3) in the formula."""
    table = {
        label: EvidenceItem(text="x", doi="10.0000/x", span="p.1", notes=None, relevance=0.8)
        for label in ["E1", "E2", "E3"]
    }

    def _confidence_for(evidence: list[str]) -> float:
        # kind="caution" here since this test is about the confidence
        # formula, not the parameter_range-must-be-numeric rule.
        draft = ExtractedPriorDraft(kind="caution", field="x", value={}, evidence=evidence)
        return _to_prior(draft, table).confidence

    conf_1 = _confidence_for(["E1"])
    conf_2 = _confidence_for(["E1", "E2"])
    conf_3 = _confidence_for(["E1", "E2", "E3"])

    assert conf_1 < conf_2 < conf_3


def test_extract_priors_retries_on_invalid_output_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_completion(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _fake_llm_response("not valid json")
        return _fake_llm_response(
            json.dumps(
                {
                    "priors": [
                        {
                            "kind": "caution",
                            "field": "note",
                            "value": {"summary": "ok"},
                            "evidence": ["E1"],
                        }
                    ]
                }
            )
        )

    monkeypatch.setattr(extract_mod.litellm, "completion", fake_completion)

    priors = extract_priors("test query", _evidence_table())
    assert calls["n"] == 2
    assert len(priors) == 1
    assert priors[0].kind == "caution"


def test_extract_priors_raises_after_max_retries(monkeypatch):
    def always_broken(**kwargs):
        return _fake_llm_response("still not json")

    monkeypatch.setattr(extract_mod.litellm, "completion", always_broken)

    with pytest.raises(ExtractionError):
        extract_priors("test query", _evidence_table())
