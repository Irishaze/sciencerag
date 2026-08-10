"""M6 (spec §3.2/§3.5): allow_external triggers real Semantic Scholar
augmentation when internal coverage is insufficient. Mocks
search_semantic_scholar and extract_priors (no real HTTP/LLM calls) so this
stays fast/free — same convention as test_priors_route.py.
"""

import pytest

from sciencerag.priors import external_retrieval, retrieval
from sciencerag.priors.external_retrieval import ExternalPaper
from sciencerag.priors.extract import ExtractionError
from sciencerag.priors.models import Coverage, Prior, PriorsResponse, SourcePaper


@pytest.fixture(autouse=True)
def _tmp_pending_path(tmp_path, monkeypatch):
    monkeypatch.setattr(external_retrieval, "PENDING_PATH", tmp_path / "pending.json")


def _response(gaps: list[str] | None = None) -> PriorsResponse:
    return PriorsResponse(
        priors=[],
        coverage=Coverage(internal_hits=0, external_hits=0, gaps=gaps or []),
        trace_id="tr_test",
    )


def _fake_paper(doi: str = "10.1/ext") -> ExternalPaper:
    return ExternalPaper(
        title="An external paper",
        abstract="Leg length of 60um yields peak COP in Bi2Te3 coolers.",
        doi=doi,
        year=2024,
        semantic_scholar_id="ss1",
    )


def _fake_prior(confidence: float = 0.8) -> Prior:
    return Prior(
        prior_id="pr_ext_1",
        kind="parameter_range",
        field="leg_length",
        value={"field_name": "leg_length", "typical": 0.06, "unit": "mm"},
        confidence=confidence,
        sources=[SourcePaper(doi="10.1/ext")],
    )


def test_allow_external_false_is_a_no_op(monkeypatch):
    called = []
    monkeypatch.setattr(retrieval, "search_semantic_scholar", lambda query: called.append(query) or [])
    response = retrieval._augment_with_external(_response(gaps=["some gap"]), "query", allow_external=False)
    assert response.coverage.gaps == ["some gap"]
    assert response.coverage.external_hits == 0
    assert called == []


def test_allow_external_true_but_no_gaps_is_a_no_op(monkeypatch):
    called = []
    monkeypatch.setattr(retrieval, "search_semantic_scholar", lambda query: called.append(query) or [])
    response = retrieval._augment_with_external(_response(gaps=[]), "query", allow_external=True)
    assert response.coverage.external_hits == 0
    assert called == []


def test_augments_priors_with_external_provenance_and_unweighted_confidence(monkeypatch):
    monkeypatch.setattr(retrieval, "search_semantic_scholar", lambda query: [_fake_paper()])
    monkeypatch.setattr(
        retrieval, "extract_priors", lambda query, evidence_table, trace=None: ([_fake_prior(0.8)], 0)
    )
    response = retrieval._augment_with_external(_response(gaps=["thin coverage"]), "leg length COP", allow_external=True)
    assert response.coverage.external_hits == 1
    assert len(response.priors) == 1
    prior = response.priors[0]
    assert prior.provenance == "external_unverified"
    # provenance alone marks the source as unverified — trust is upgraded
    # only through scripts/approve_external_papers.py, not a confidence
    # multiplier, so this should be the same 0.8 extract_priors produced.
    assert prior.confidence == 0.8

    pending = external_retrieval._load_pending()
    assert "10.1/ext" in pending
    assert pending["10.1/ext"].hit_count == 1


def test_no_search_results_appends_gap_not_priors(monkeypatch):
    monkeypatch.setattr(retrieval, "search_semantic_scholar", lambda query: [])
    response = retrieval._augment_with_external(_response(gaps=["thin coverage"]), "query", allow_external=True)
    assert response.priors == []
    assert response.coverage.external_hits == 0
    assert any("Semantic Scholar" in gap for gap in response.coverage.gaps)


def test_extraction_failure_records_hit_count_without_priors(monkeypatch):
    def _raise(query, evidence_table, trace=None):
        raise ExtractionError("schema validation failed")

    monkeypatch.setattr(retrieval, "search_semantic_scholar", lambda query: [_fake_paper(), _fake_paper("10.1/ext2")])
    monkeypatch.setattr(retrieval, "extract_priors", _raise)
    response = retrieval._augment_with_external(_response(gaps=["thin coverage"]), "query", allow_external=True)
    assert response.priors == []
    assert response.coverage.external_hits == 2
    assert any("LLM extraction failed" in gap for gap in response.coverage.gaps)
