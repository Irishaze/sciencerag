"""Tests for the semantic support judge (sciencerag/priors/extract.py).

litellm.completion is monkeypatched — no real API calls, no cost. Real
model behavior (does luna's KEEP/REVIEW/DROP actually track a real
independent judge) was validated separately against production data
(scripts/luna_binary_judge_v3.py, 2026-08-04) — these tests only check
this module's own plumbing: request/response shape, retry wiring, error
handling.
"""

import json
from types import SimpleNamespace

import pytest

from sciencerag.priors import extract as extract_mod
from sciencerag.priors.extract import (
    EvidenceItem,
    Prior,
    SourcePaper,
    _evidence_labels_for,
    _judge_semantic_support,
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


def _prior(prior_id: str, doi: str, span: str) -> Prior:
    return Prior(
        prior_id=prior_id,
        kind="parameter_range",
        field="leg_length",
        value={"field_name": "leg_length", "typical": 60, "unit": "um"},
        confidence=0.6,
        sources=[SourcePaper(doi=doi, span=span)],
    )


# -- _evidence_labels_for -----------------------------------------------


def test_evidence_labels_for_matches_by_doi_and_span():
    table = _evidence_table()
    prior = _prior("pr_1", doi="10.5678/other", span="p.7")
    assert _evidence_labels_for(prior, table) == ["E2"]


def test_evidence_labels_for_returns_empty_when_no_match():
    table = _evidence_table()
    prior = _prior("pr_1", doi="10.9999/nomatch", span="p.99")
    assert _evidence_labels_for(prior, table) == []


# -- _judge_semantic_support ----------------------------------------------


def test_judge_semantic_support_parses_verdicts_by_prior_id(monkeypatch):
    table = _evidence_table()
    priors = [_prior("pr_1", doi="10.5678/other", span="p.7")]

    def fake_completion(**kwargs):
        return _fake_llm_response(
            json.dumps({"verdicts": [{"prior_id": "pr_1", "verdict": "KEEP", "reason": "matches"}]})
        )

    monkeypatch.setattr(extract_mod.litellm, "completion", fake_completion)
    verdicts = _judge_semantic_support("test query", priors, table)
    assert verdicts["pr_1"].verdict == "KEEP"
    assert verdicts["pr_1"].reason == "matches"


def test_judge_semantic_support_falls_back_on_temperature_rejection(monkeypatch):
    """Some models (gpt-5.6 family) reject a custom temperature outright —
    same fallback pattern as extract_priors' own LLM call."""
    table = _evidence_table()
    priors = [_prior("pr_1", doi="10.5678/other", span="p.7")]
    calls = {"n": 0}

    def fake_completion(**kwargs):
        calls["n"] += 1
        if "temperature" in kwargs:
            raise extract_mod.litellm.BadRequestError(
                "Unsupported value: 'temperature' does not support 0 with this model. "
                "Only the default (1) value is supported.",
                model="x",
                llm_provider="x",
            )
        return _fake_llm_response(
            json.dumps({"verdicts": [{"prior_id": "pr_1", "verdict": "KEEP", "reason": "ok"}]})
        )

    monkeypatch.setattr(extract_mod.litellm, "completion", fake_completion)
    verdicts = _judge_semantic_support("test query", priors, table)
    assert verdicts["pr_1"].verdict == "KEEP"
    assert calls["n"] == 2


def test_judge_semantic_support_raises_on_malformed_json(monkeypatch):
    table = _evidence_table()
    priors = [_prior("pr_1", doi="10.5678/other", span="p.7")]

    def fake_completion(**kwargs):
        return _fake_llm_response("not json")

    monkeypatch.setattr(extract_mod.litellm, "completion", fake_completion)
    with pytest.raises(ValueError, match="semantic support judge failed"):
        _judge_semantic_support("test query", priors, table)


def test_judge_semantic_support_raises_when_a_prior_id_is_omitted(monkeypatch):
    """The judge dropping a prior from its response entirely (not even a
    DROP verdict) must fail loud, not silently treat it as approved."""
    table = _evidence_table()
    priors = [
        _prior("pr_1", doi="10.5678/other", span="p.7"),
        _prior("pr_2", doi="10.1234/example", span="p.4, Fig.3"),
    ]

    def fake_completion(**kwargs):
        return _fake_llm_response(
            json.dumps({"verdicts": [{"prior_id": "pr_1", "verdict": "KEEP", "reason": "ok"}]})
        )

    monkeypatch.setattr(extract_mod.litellm, "completion", fake_completion)
    with pytest.raises(ValueError, match="omitted verdicts"):
        _judge_semantic_support("test query", priors, table)


# -- extract_priors integration: DROP retries, REVIEW excludes without retry --


def _draft_json(typical: int = 60) -> str:
    return json.dumps(
        {
            "priors": [
                {
                    "kind": "parameter_range",
                    "field": "leg_length",
                    "value": {"field_name": "leg_length", "typical": typical, "unit": "um"},
                    "evidence": ["E2"],
                }
            ]
        }
    )


def test_extract_priors_retries_on_semantic_drop_then_succeeds(monkeypatch):
    """A DROP verdict must fail the whole attempt and retry — same
    mechanism as a numeric-groundedness failure — not silently pass through
    or crash."""
    extraction_calls = {"n": 0}
    judge_calls = {"n": 0}

    def fake_completion(**kwargs):
        if kwargs["messages"][0]["content"] == extract_mod.SEMANTIC_SUPPORT_RUBRIC:
            judge_calls["n"] += 1
            user_prompt = kwargs["messages"][1]["content"]
            import re

            prior_id = re.search(r"id: (\S+)", user_prompt).group(1)
            verdict = "DROP" if judge_calls["n"] == 1 else "KEEP"
            return _fake_llm_response(
                json.dumps(
                    {"verdicts": [{"prior_id": prior_id, "verdict": verdict, "reason": "reason"}]}
                )
            )
        extraction_calls["n"] += 1
        return _fake_llm_response(_draft_json())

    monkeypatch.setattr(extract_mod.litellm, "completion", fake_completion)
    priors, _filtered, review_priors = extract_priors("test query", _evidence_table())

    assert extraction_calls["n"] == 2  # first attempt's draft got DROP'd, retried
    assert judge_calls["n"] == 2
    assert len(priors) == 1
    assert review_priors == []


def test_extract_priors_excludes_review_verdict_without_retrying(monkeypatch):
    """A REVIEW verdict must NOT trigger a retry (the evidence won't get
    more literal on another attempt) — the prior is excluded from `priors`
    and surfaced via `review_priors` instead."""
    extraction_calls = {"n": 0}

    def fake_completion(**kwargs):
        if kwargs["messages"][0]["content"] == extract_mod.SEMANTIC_SUPPORT_RUBRIC:
            user_prompt = kwargs["messages"][1]["content"]
            import re

            prior_id = re.search(r"id: (\S+)", user_prompt).group(1)
            return _fake_llm_response(
                json.dumps(
                    {
                        "verdicts": [
                            {"prior_id": prior_id, "verdict": "REVIEW", "reason": "inferred mapping"}
                        ]
                    }
                )
            )
        extraction_calls["n"] += 1
        return _fake_llm_response(_draft_json())

    monkeypatch.setattr(extract_mod.litellm, "completion", fake_completion)
    priors, _filtered, review_priors = extract_priors("test query", _evidence_table())

    assert extraction_calls["n"] == 1  # no retry triggered
    assert priors == []
    assert len(review_priors) == 1
    assert review_priors[0].prior.field == "leg_length"
    assert review_priors[0].reason == "inferred mapping"
