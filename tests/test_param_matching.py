"""Tests for the LLM-based relevance-tier matcher (sciencerag/priors/retrieval.py).

Only the API-free paths are testable here: the empty-input short-circuit
(no litellm call at all) and the keyword-heuristic fallback used when the
real call errors out. The real LLM path itself is exercised by manual
real-query smoke testing (spec principle: unit tests don't call real APIs).
"""

from sciencerag.priors.extract import EvidenceItem
from sciencerag.priors.retrieval import _keyword_fallback_match, _match_params_to_evidence


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
