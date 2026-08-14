"""Tests for the LLM-based relevance-tier matcher (sciencerag/priors/retrieval.py).

Only the API-free paths are testable here: the empty-input short-circuit
(no litellm call at all) and the keyword-heuristic fallback used when the
real call errors out. The real LLM path itself is exercised by manual
real-query smoke testing (spec principle: unit tests don't call real APIs).
"""

from types import SimpleNamespace

from sciencerag.priors import retrieval
from sciencerag.priors.extract import EvidenceItem
from sciencerag.priors.retrieval import (
    _keyword_fallback_match,
    _match_params_to_evidence,
    _translate_for_literature_search,
)


def _evidence(text: str, relevance: float = 0.3, doi: str = "10.0000/x") -> EvidenceItem:
    return EvidenceItem(text=text, doi=doi, span="p.1", notes=None, relevance=relevance)


def test_match_params_to_evidence_short_circuits_on_empty_evidence():
    assert _match_params_to_evidence([], ["leg_length"]) == set()


def test_match_params_to_evidence_short_circuits_on_empty_candidates():
    below_threshold = [_evidence("A study of leg length in TEC modules.")]
    assert _match_params_to_evidence(below_threshold, []) == set()


def test_keyword_fallback_matches_leg_length_synonym_leg_height():
    """Real evidence text calls this dimension "leg height" as often as
    "leg length" (the leg stands between hot/cold plates)."""
    below_threshold = [_evidence("Effect of leg height on cooling performance.")]
    matched = _keyword_fallback_match(below_threshold, ["leg_length", "leg_width"])
    assert matched == {"leg_length"}


def test_keyword_fallback_no_match_for_unrelated_evidence():
    below_threshold = [_evidence("Discussion of Seebeck coefficient temperature dependence.")]
    matched = _keyword_fallback_match(below_threshold, ["leg_length"])
    assert matched == set()


def _fake_llm_response(text: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def test_translate_short_circuits_on_ascii_query_no_llm_call(monkeypatch):
    called = []
    monkeypatch.setattr(
        retrieval.litellm, "completion", lambda **kwargs: called.append(kwargs) or _fake_llm_response("x")
    )
    result = _translate_for_literature_search("Bi2Te3 leg length effect on COP")
    assert result == "Bi2Te3 leg length effect on COP"
    assert called == []


def test_translate_calls_llm_for_cjk_query(monkeypatch):
    # Regression for the 2026-08-13 finding: PaperQA2's own agent burns a
    # full paper_search+gather_evidence round finding little when the
    # first query is Chinese against an all-English corpus, then
    # reformulates into English itself — several extra minutes, confirmed
    # via real container logs. Translating up front lets the first round
    # already succeed.
    captured = {}

    def _fake_completion(**kwargs):
        captured.update(kwargs)
        return _fake_llm_response("Bi2Te3 leg length effect on maximum temperature difference")

    monkeypatch.setattr(retrieval.litellm, "completion", _fake_completion)
    result = _translate_for_literature_search("Bi2Te3热电制冷器的腿长对最大温差有什么影响")
    assert result == "Bi2Te3 leg length effect on maximum temperature difference"
    assert captured["messages"][1]["content"] == "Bi2Te3热电制冷器的腿长对最大温差有什么影响"


def test_translate_falls_back_to_original_on_llm_failure(monkeypatch):
    def _boom(**kwargs):
        raise RuntimeError("DeepSeek API timed out")

    monkeypatch.setattr(retrieval.litellm, "completion", _boom)
    result = _translate_for_literature_search("Bi2Te3热电制冷器的腿长对最大温差有什么影响")
    assert result == "Bi2Te3热电制冷器的腿长对最大温差有什么影响"


def test_translate_falls_back_to_original_on_blank_llm_output(monkeypatch):
    monkeypatch.setattr(retrieval.litellm, "completion", lambda **kwargs: _fake_llm_response("   "))
    result = _translate_for_literature_search("Bi2Te3热电制冷器的腿长对最大温差有什么影响")
    assert result == "Bi2Te3热电制冷器的腿长对最大温差有什么影响"
