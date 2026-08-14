"""M6 (spec §3.2/§3.5): allow_external triggers real Semantic Scholar +
arXiv augmentation when internal coverage is insufficient. Mocks every
network/LLM boundary (search_semantic_scholar, search_arxiv,
download_new_papers, run_query, _build_evidence_table, extract_priors) so
this stays fast/free — same convention as test_priors_route.py.

A paper with a downloadable PDF (arXiv always; Semantic Scholar when
open-access) is treated as trusted the moment it's downloaded — no
approval queue, no confidence penalty, `provenance` stays the model
default ("internal"). Only the abstract-only fallback (no PDF available)
still gets tagged `provenance="external_unverified"`.
"""

import pytest

from sciencerag.priors import retrieval
from sciencerag.priors.extract import EvidenceItem, ExtractionError
from sciencerag.priors.external_retrieval import ExternalPaper
from sciencerag.priors.models import Coverage, Prior, PriorsResponse, SourcePaper


def _response(gaps: list[str] | None = None) -> PriorsResponse:
    return PriorsResponse(
        priors=[],
        coverage=Coverage(internal_hits=0, external_hits=0, gaps=gaps or []),
        trace_id="tr_test",
    )


def _ss_paper_no_pdf(doi: str = "10.1/ext") -> ExternalPaper:
    return ExternalPaper(
        title="A paywalled paper",
        abstract="Leg length of 60um yields peak COP in Bi2Te3 coolers.",
        doi=doi,
        year=2024,
        semantic_scholar_id="ss1",
        source="semantic_scholar",
        pdf_url=None,
    )


def _ss_paper_open_access(doi: str = "10.1/oa") -> ExternalPaper:
    return ExternalPaper(
        title="An open-access paper",
        abstract="short abstract",
        doi=doi,
        year=2024,
        semantic_scholar_id="ss2",
        source="semantic_scholar",
        pdf_url="https://example.org/oa.pdf",
    )


def _arxiv_paper(doi: str = "10.48550/arXiv.2011.02585") -> ExternalPaper:
    return ExternalPaper(
        title="An arXiv preprint",
        abstract="short abstract",
        doi=doi,
        year=2020,
        source="arxiv",
        pdf_url="https://arxiv.org/pdf/2011.02585",
    )


def _fake_prior(doi: str = "10.1/ext", confidence: float = 0.8) -> Prior:
    return Prior(
        prior_id="pr_ext_1",
        kind="parameter_range",
        field="leg_length",
        value={"field_name": "leg_length", "typical": 0.06, "unit": "mm"},
        confidence=confidence,
        sources=[SourcePaper(doi=doi)],
    )


class _FakeSession:
    def __init__(self, contexts):
        self.contexts = contexts


class _FakeQueryResponse:
    def __init__(self, contexts):
        self.session = _FakeSession(contexts)


def test_allow_external_false_is_a_no_op(monkeypatch):
    called = []
    monkeypatch.setattr(retrieval, "search_semantic_scholar", lambda query: called.append(query) or [])
    monkeypatch.setattr(retrieval, "search_arxiv", lambda query: [])
    response = retrieval._augment_with_external(_response(gaps=["some gap"]), "query", "query", allow_external=False)
    assert response.coverage.gaps == ["some gap"]
    assert response.coverage.external_hits == 0
    assert called == []


def test_allow_external_true_but_no_gaps_is_a_no_op(monkeypatch):
    called = []
    monkeypatch.setattr(retrieval, "search_semantic_scholar", lambda query: called.append(query) or [])
    monkeypatch.setattr(retrieval, "search_arxiv", lambda query: [])
    response = retrieval._augment_with_external(_response(gaps=[]), "query", "query", allow_external=True)
    assert response.coverage.external_hits == 0
    assert called == []


def test_no_search_results_appends_gap_not_priors(monkeypatch):
    monkeypatch.setattr(retrieval, "search_semantic_scholar", lambda query: [])
    monkeypatch.setattr(retrieval, "search_arxiv", lambda query: [])
    response = retrieval._augment_with_external(_response(gaps=["thin coverage"]), "query", "query", allow_external=True)
    assert response.priors == []
    assert response.coverage.external_hits == 0
    assert any("Semantic Scholar and arXiv" in gap for gap in response.coverage.gaps)


def test_full_text_hit_is_downloaded_requeried_and_tagged_internal(monkeypatch):
    arxiv_paper = _arxiv_paper()
    evidence_item = EvidenceItem(
        text="real full-text chunk about leg length",
        doi=arxiv_paper.doi,
        span="p3-4",
        notes="An arXiv preprint",
        relevance=0.9,
    )

    monkeypatch.setattr(retrieval, "search_semantic_scholar", lambda query: [])
    monkeypatch.setattr(retrieval, "search_arxiv", lambda query: [arxiv_paper])
    monkeypatch.setattr(retrieval, "download_new_papers", lambda papers: [arxiv_paper])
    monkeypatch.setattr(retrieval, "run_query", lambda query: _FakeQueryResponse(["ctx"]))
    monkeypatch.setattr(
        retrieval, "_build_evidence_table", lambda contexts: ({"E1": evidence_item}, [])
    )
    monkeypatch.setattr(
        retrieval,
        "extract_priors",
        lambda query, evidence_table, trace=None: ([_fake_prior(arxiv_paper.doi)], 0, []),
    )

    response = retrieval._augment_with_external(_response(gaps=["thin coverage"]), "leg length COP", "leg length COP", allow_external=True)

    assert response.coverage.external_hits == 1
    assert len(response.priors) == 1
    assert response.priors[0].provenance == "internal"


def test_paywalled_hit_falls_back_to_abstract_and_stays_external_unverified(monkeypatch):
    no_pdf_paper = _ss_paper_no_pdf()

    monkeypatch.setattr(retrieval, "search_semantic_scholar", lambda query: [no_pdf_paper])
    monkeypatch.setattr(retrieval, "search_arxiv", lambda query: [])
    monkeypatch.setattr(retrieval, "download_new_papers", lambda papers: [])

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("run_query should not be called when nothing new was downloaded")

    monkeypatch.setattr(retrieval, "run_query", _fail_if_called)
    monkeypatch.setattr(
        retrieval,
        "extract_priors",
        lambda query, evidence_table, trace=None: ([_fake_prior(no_pdf_paper.doi, 0.8)], 0, []),
    )

    response = retrieval._augment_with_external(_response(gaps=["thin coverage"]), "leg length COP", "leg length COP", allow_external=True)

    assert response.coverage.external_hits == 1
    assert len(response.priors) == 1
    prior = response.priors[0]
    assert prior.provenance == "external_unverified"
    # no confidence penalty — same score extract_priors produced
    assert prior.confidence == 0.8


def test_mixed_full_text_and_abstract_only_both_contribute(monkeypatch):
    open_access = _ss_paper_open_access()
    paywalled = _ss_paper_no_pdf(doi="10.1/paywalled")
    evidence_item = EvidenceItem(
        text="full text chunk", doi=open_access.doi, span="p1", notes="An open-access paper", relevance=0.9
    )

    monkeypatch.setattr(retrieval, "search_semantic_scholar", lambda query: [open_access, paywalled])
    monkeypatch.setattr(retrieval, "search_arxiv", lambda query: [])
    monkeypatch.setattr(retrieval, "download_new_papers", lambda papers: [open_access])
    monkeypatch.setattr(retrieval, "run_query", lambda query: _FakeQueryResponse(["ctx"]))
    monkeypatch.setattr(
        retrieval, "_build_evidence_table", lambda contexts: ({"E1": evidence_item}, [])
    )

    calls = []

    def _fake_extract(query, evidence_table, trace=None):
        calls.append(evidence_table)
        doi = next(iter(evidence_table.values())).doi
        return ([_fake_prior(doi)], 0, [])

    monkeypatch.setattr(retrieval, "extract_priors", _fake_extract)

    response = retrieval._augment_with_external(_response(gaps=["thin coverage"]), "query", "query", allow_external=True)

    assert response.coverage.external_hits == 2
    assert len(response.priors) == 2
    provenances = {p.provenance for p in response.priors}
    assert provenances == {"internal", "external_unverified"}
    assert len(calls) == 2  # one call for the full-text table, one for the abstract table


def test_extraction_failure_on_abstract_path_appends_gap(monkeypatch):
    def _raise(query, evidence_table, trace=None):
        raise ExtractionError("schema validation failed")

    monkeypatch.setattr(retrieval, "search_semantic_scholar", lambda query: [_ss_paper_no_pdf(), _ss_paper_no_pdf("10.1/ext2")])
    monkeypatch.setattr(retrieval, "search_arxiv", lambda query: [])
    monkeypatch.setattr(retrieval, "download_new_papers", lambda papers: [])
    monkeypatch.setattr(retrieval, "extract_priors", _raise)

    response = retrieval._augment_with_external(_response(gaps=["thin coverage"]), "query", "query", allow_external=True)

    assert response.priors == []
    assert response.coverage.external_hits == 2
    assert any("LLM extraction failed" in gap or "extraction failed" in gap for gap in response.coverage.gaps)


def test_same_doi_from_both_sources_is_deduped_and_prefers_full_text(monkeypatch):
    """The same paper can legitimately turn up in both Semantic Scholar and
    arXiv search results under the same DOI — external_hits must reflect
    distinct papers found, not raw search hits, and the merge must not
    downgrade a full-text-eligible hit to abstract-only just because the
    no-pdf_url copy happened to be deduped in second."""
    same_doi = "10.48550/arXiv.2011.02585"
    ss_copy = ExternalPaper(
        title="SS version", abstract="a", doi=same_doi, source="semantic_scholar", pdf_url=None
    )
    arxiv_copy = ExternalPaper(
        title="arXiv version",
        abstract="a",
        doi=same_doi,
        source="arxiv",
        pdf_url="https://arxiv.org/pdf/2011.02585",
    )

    monkeypatch.setattr(retrieval, "search_semantic_scholar", lambda query: [ss_copy])
    monkeypatch.setattr(retrieval, "search_arxiv", lambda query: [arxiv_copy])

    captured = {}

    def _fake_download(papers):
        captured["papers"] = list(papers)
        return list(papers)

    monkeypatch.setattr(retrieval, "download_new_papers", _fake_download)
    monkeypatch.setattr(retrieval, "run_query", lambda query: _FakeQueryResponse([]))

    response = retrieval._augment_with_external(_response(gaps=["thin coverage"]), "query", "query", allow_external=True)

    assert response.coverage.external_hits == 1
    assert len(captured["papers"]) == 1
    assert captured["papers"][0].pdf_url is not None


def test_run_query_failure_during_requery_degrades_gracefully(monkeypatch):
    """The re-query for newly-downloaded full text hits a real LLM/PaperQA2
    pipeline a second time — a transient failure there (timeout, provider
    error, an unparseable file that just landed on disk) must not take
    down the whole response. Before this was guarded, any exception here
    propagated out of _augment_with_external uncaught, and the router's
    catch-all (sciencerag/priors/router.py) turned that into a 502 for the
    *entire* request — discarding whatever good internal priors the first
    pass already found, just because an optional enhancement hiccuped."""
    arxiv_paper = _arxiv_paper()

    monkeypatch.setattr(retrieval, "search_semantic_scholar", lambda query: [])
    monkeypatch.setattr(retrieval, "search_arxiv", lambda query: [arxiv_paper])
    monkeypatch.setattr(retrieval, "download_new_papers", lambda papers: [arxiv_paper])

    def _boom(query):
        raise RuntimeError("DeepSeek API timed out mid-summary")

    monkeypatch.setattr(retrieval, "run_query", _boom)

    response = retrieval._augment_with_external(_response(gaps=["thin coverage"]), "query", "query", allow_external=True)

    assert response.priors == []
    assert response.coverage.external_hits == 1
    assert any("re-query failed" in gap for gap in response.coverage.gaps)


def test_already_downloaded_paper_does_not_requery(monkeypatch):
    """download_new_papers returns [] when nothing new landed on disk (already
    present from a prior request) — no reason to pay for a second query;
    that paper's evidence already flows through the ordinary internal pass
    on its own."""
    open_access = _ss_paper_open_access()

    monkeypatch.setattr(retrieval, "search_semantic_scholar", lambda query: [open_access])
    monkeypatch.setattr(retrieval, "search_arxiv", lambda query: [])
    monkeypatch.setattr(retrieval, "download_new_papers", lambda papers: [])

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("run_query should not be called when nothing new was downloaded")

    monkeypatch.setattr(retrieval, "run_query", _fail_if_called)

    response = retrieval._augment_with_external(_response(gaps=["thin coverage"]), "query", "query", allow_external=True)

    assert response.priors == []
    assert response.coverage.external_hits == 1
