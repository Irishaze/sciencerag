"""Unit tests for sciencerag/priors/external_retrieval.py (M6). Mocks
httpx so this stays fast/free and independent of Semantic Scholar's live
rate limits and real network access.
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
    assert papers[0].source == "semantic_scholar"
    assert papers[0].pdf_url is None


def test_search_populates_pdf_url_when_open_access(monkeypatch):
    payload = {
        "data": [
            {
                "title": "Open access paper",
                "abstract": "text",
                "externalIds": {"DOI": "10.1/oa"},
                "year": 2023,
                "paperId": "ss_oa",
                "openAccessPdf": {"url": "https://example.org/oa.pdf", "status": "GOLD"},
            },
        ]
    }
    monkeypatch.setattr(external_retrieval.httpx, "get", lambda *a, **k: _fake_response(payload))
    papers = external_retrieval.search_semantic_scholar("query")
    assert papers[0].pdf_url == "https://example.org/oa.pdf"


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


def _paper(doi: str = "10.1/x", pdf_url: str | None = "https://example.org/x.pdf"):
    return external_retrieval.ExternalPaper(
        title="T", abstract="A", doi=doi, year=2024, source="arxiv", pdf_url=pdf_url
    )


class _FakeStreamCtx:
    """Mimics httpx.stream(...)'s context-manager return value."""

    def __init__(
        self,
        chunks: list[bytes],
        status_code: int = 200,
        is_redirect: bool = False,
        location: str | None = None,
        url: str = "https://example.org/x.pdf",
    ):
        self._chunks = chunks
        self._status_code = status_code
        self._is_redirect = is_redirect
        self._location = location
        self._url = url

    def __enter__(self):
        def raise_for_status():
            if self._status_code >= 400:
                raise external_retrieval.httpx.HTTPStatusError(
                    "error", request=None, response=SimpleNamespace(status_code=self._status_code)
                )

        return SimpleNamespace(
            raise_for_status=raise_for_status,
            iter_bytes=lambda: iter(self._chunks),
            is_redirect=self._is_redirect,
            headers={"location": self._location} if self._location else {},
            url=external_retrieval.httpx.URL(self._url),
        )

    def __exit__(self, *exc_info):
        return False


def _stub_stream(content: bytes | None = None, chunks: list[bytes] | None = None, status_code: int = 200):
    payload = chunks if chunks is not None else [content if content is not None else b""]
    return lambda *a, **k: _FakeStreamCtx(payload, status_code)


def test_paper_pdf_path_percent_encodes_doi_no_collisions(monkeypatch, tmp_path):
    monkeypatch.setattr(external_retrieval, "CORPUS_DIR", tmp_path)
    path = external_retrieval.paper_pdf_path("10.48550/arXiv.2011.02585")
    assert path == tmp_path / "10.48550%2FarXiv.2011.02585.pdf"
    # a naive "/" -> "_" replacement would collide these two distinct DOIs
    # onto the same filename, silently shadowing one paper's evidence with
    # another's; percent-encoding must not.
    assert external_retrieval.paper_pdf_path("10.1000/182") != external_retrieval.paper_pdf_path(
        "10.1000_182"
    )


def test_download_new_papers_skips_papers_without_pdf_url(monkeypatch, tmp_path):
    monkeypatch.setattr(external_retrieval, "CORPUS_DIR", tmp_path)
    called = []
    monkeypatch.setattr(external_retrieval.httpx, "stream", lambda *a, **k: called.append(1))
    downloaded = external_retrieval.download_new_papers([_paper(pdf_url=None)])
    assert downloaded == []
    assert called == []


def test_download_new_papers_writes_file_and_returns_it(monkeypatch, tmp_path):
    monkeypatch.setattr(external_retrieval, "CORPUS_DIR", tmp_path)
    monkeypatch.setattr(external_retrieval.httpx, "stream", _stub_stream(b"%PDF-fake"))
    paper = _paper()
    downloaded = external_retrieval.download_new_papers([paper])
    assert downloaded == [paper]
    assert external_retrieval.paper_pdf_path(paper.doi).read_bytes() == b"%PDF-fake"
    # no leftover .tmp file from the write-then-rename
    assert list(tmp_path.glob("*.tmp")) == []


def test_download_new_papers_skips_already_on_disk(monkeypatch, tmp_path):
    monkeypatch.setattr(external_retrieval, "CORPUS_DIR", tmp_path)
    external_retrieval.paper_pdf_path("10.1/x").write_bytes(b"already here")

    called = []
    monkeypatch.setattr(external_retrieval.httpx, "stream", lambda *a, **k: called.append(1))

    downloaded = external_retrieval.download_new_papers([_paper()])
    assert downloaded == []
    assert called == []


def test_download_new_papers_degrades_on_http_error(monkeypatch, tmp_path):
    monkeypatch.setattr(external_retrieval, "CORPUS_DIR", tmp_path)

    def _raise(*a, **k):
        raise external_retrieval.httpx.ConnectTimeout("timed out")

    monkeypatch.setattr(external_retrieval.httpx, "stream", _raise)
    downloaded = external_retrieval.download_new_papers([_paper()])
    assert downloaded == []
    assert not external_retrieval.paper_pdf_path("10.1/x").exists()


def test_download_new_papers_rejects_non_public_pdf_url(monkeypatch, tmp_path):
    """SSRF guard: pdf_url is attacker-influenceable (comes straight from a
    public API response) and must not be used to reach internal/loopback
    services."""
    monkeypatch.setattr(external_retrieval, "CORPUS_DIR", tmp_path)
    called = []
    monkeypatch.setattr(external_retrieval.httpx, "stream", lambda *a, **k: called.append(1))

    for evil_url in [
        "http://127.0.0.1:8000/internal",
        "http://169.254.169.254/latest/meta-data/",
        "http://localhost/admin",
        "file:///etc/passwd",
        "ftp://example.org/x.pdf",
    ]:
        downloaded = external_retrieval.download_new_papers([_paper(pdf_url=evil_url)])
        assert downloaded == [], f"should have rejected {evil_url}"
    assert called == []


def test_download_new_papers_rejects_redirect_to_internal_address(monkeypatch, tmp_path):
    """SSRF-via-redirect (2026-08-15 full-system adversarial review): the
    initial pdf_url is validated against _is_fetchable_url, but the old code
    passed follow_redirects=True straight to httpx, which transparently
    follows any redirect chain with zero re-validation of the destination —
    a URL that itself resolves to a public IP (passing the initial check)
    could 302 to http://169.254.169.254/... or any internal/loopback
    address and httpx would fetch it anyway. Each hop must be re-checked."""
    monkeypatch.setattr(external_retrieval, "CORPUS_DIR", tmp_path)

    calls = []

    def _stream(method, url, **kwargs):
        calls.append(url)
        if url == "https://example.org/x.pdf":
            return _FakeStreamCtx([], status_code=302, is_redirect=True, location="http://169.254.169.254/secret")
        raise AssertionError(f"must not have followed the redirect to {url}")

    monkeypatch.setattr(external_retrieval.httpx, "stream", _stream)
    downloaded = external_retrieval.download_new_papers([_paper()])
    assert downloaded == []
    assert not external_retrieval.paper_pdf_path("10.1/x").exists()
    assert calls == ["https://example.org/x.pdf"]  # never actually requested the redirect target


def test_download_new_papers_follows_redirect_to_public_address(monkeypatch, tmp_path):
    """The other half of the same fix: a redirect to a genuinely public
    address (the normal case — e.g. a DOI resolver bouncing to the
    publisher's real PDF host) must still work, not be broken by closing
    the SSRF hole."""
    monkeypatch.setattr(external_retrieval, "CORPUS_DIR", tmp_path)

    calls = []

    def _stream(method, url, **kwargs):
        calls.append(url)
        if url == "https://example.org/x.pdf":
            return _FakeStreamCtx([], status_code=302, is_redirect=True, location="https://example.com/real.pdf")
        return _FakeStreamCtx([b"%PDF-real content"], status_code=200)

    monkeypatch.setattr(external_retrieval.httpx, "stream", _stream)
    downloaded = external_retrieval.download_new_papers([_paper()])
    assert len(downloaded) == 1
    assert calls == ["https://example.org/x.pdf", "https://example.com/real.pdf"]
    assert external_retrieval.paper_pdf_path("10.1/x").read_bytes() == b"%PDF-real content"


def test_download_new_papers_rejects_non_pdf_content(monkeypatch, tmp_path):
    """A stale/broken openAccessPdf link commonly redirects to an HTML
    paywall page instead of 404ing — that must not be saved and trusted as
    the paper's full text."""
    monkeypatch.setattr(external_retrieval, "CORPUS_DIR", tmp_path)
    monkeypatch.setattr(
        external_retrieval.httpx, "stream", _stub_stream(b"<html><body>Please log in</body></html>")
    )
    downloaded = external_retrieval.download_new_papers([_paper()])
    assert downloaded == []
    assert not external_retrieval.paper_pdf_path("10.1/x").exists()


def test_download_new_papers_enforces_size_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(external_retrieval, "CORPUS_DIR", tmp_path)
    monkeypatch.setattr(external_retrieval, "MAX_PDF_BYTES", 10)
    monkeypatch.setattr(
        external_retrieval.httpx, "stream", _stub_stream(chunks=[b"%PDF-", b"0123456789extra"])
    )
    downloaded = external_retrieval.download_new_papers([_paper()])
    assert downloaded == []
    assert not external_retrieval.paper_pdf_path("10.1/x").exists()


def test_download_new_papers_degrades_on_disk_write_failure(monkeypatch, tmp_path):
    # A path that cannot be created (parent is a file, not a directory)
    # forces mkdir to raise OSError.
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"not a directory")
    monkeypatch.setattr(external_retrieval, "CORPUS_DIR", blocker / "papers")
    monkeypatch.setattr(external_retrieval.httpx, "stream", _stub_stream(b"%PDF-fake"))

    downloaded = external_retrieval.download_new_papers([_paper()])
    assert downloaded == []
