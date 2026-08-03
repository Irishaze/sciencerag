"""Unit tests for sciencerag/priors/external_retrieval.py (M6). Mocks
httpx.get so this stays fast/free and independent of Semantic Scholar's
live rate limits.
"""

from types import SimpleNamespace

from sciencerag.priors import external_retrieval


def _fake_response(payload: dict, status_code: int = 200):
    def raise_for_status():
        if status_code >= 400:
            raise external_retrieval.httpx.HTTPStatusError(
                "error", request=None, response=SimpleNamespace(status_code=status_code)
            )

    return SimpleNamespace(json=lambda: payload, raise_for_status=raise_for_status)


def test_search_returns_papers_with_abstract_and_doi(monkeypatch):
    payload = {
        "data": [
            {
                "title": "Paper A",
                "abstract": "Some findings about leg length.",
                "externalIds": {"DOI": "10.1/a"},
                "year": 2023,
                "paperId": "ss_a",
            },
        ]
    }
    monkeypatch.setattr(external_retrieval.httpx, "get", lambda *a, **k: _fake_response(payload))
    papers = external_retrieval.search_semantic_scholar("query")
    assert len(papers) == 1
    assert papers[0].doi == "10.1/a"
    assert papers[0].abstract == "Some findings about leg length."


def test_papers_without_abstract_are_skipped(monkeypatch):
    payload = {"data": [{"title": "No abstract", "externalIds": {"DOI": "10.1/b"}, "paperId": "ss_b"}]}
    monkeypatch.setattr(external_retrieval.httpx, "get", lambda *a, **k: _fake_response(payload))
    assert external_retrieval.search_semantic_scholar("query") == []


def test_papers_without_doi_are_skipped(monkeypatch):
    payload = {"data": [{"title": "No DOI", "abstract": "text", "externalIds": {}, "paperId": "ss_c"}]}
    monkeypatch.setattr(external_retrieval.httpx, "get", lambda *a, **k: _fake_response(payload))
    assert external_retrieval.search_semantic_scholar("query") == []


def test_http_error_degrades_to_empty_list(monkeypatch):
    def _raise(*args, **kwargs):
        raise external_retrieval.httpx.ConnectTimeout("timed out")

    monkeypatch.setattr(external_retrieval.httpx, "get", _raise)
    assert external_retrieval.search_semantic_scholar("query") == []


def test_rate_limited_response_degrades_to_empty_list(monkeypatch):
    monkeypatch.setattr(
        external_retrieval.httpx, "get", lambda *a, **k: _fake_response({}, status_code=429)
    )
    assert external_retrieval.search_semantic_scholar("query") == []
