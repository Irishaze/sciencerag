"""Unit tests for sciencerag/priors/arxiv_retrieval.py (M6). Mocks httpx
so this stays fast/free and independent of arXiv's live API.
"""

from types import SimpleNamespace

from sciencerag.priors import arxiv_retrieval

_ATOM_ENTRY = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/" xmlns:arxiv="http://arxiv.org/schemas/atom" xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2011.02585v2</id>
    <title>
  A Study of Thermoelectric Leg Geometry
</title>
    <updated>2020-11-05T00:00:00Z</updated>
    <link href="https://arxiv.org/abs/2011.02585v2" rel="alternate" type="text/html"/>
    <link href="https://arxiv.org/pdf/2011.02585v2" rel="related" type="application/pdf" title="pdf"/>
    <summary>
  Leg length strongly affects COP in Bi2Te3 coolers.
</summary>
    <published>2020-11-05T00:00:00Z</published>
    <author><name>A. Author</name></author>
  </entry>
</feed>
"""

_EMPTY_FEED = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>
"""


def _fake_response(text: str, status_code: int = 200):
    def raise_for_status():
        if status_code >= 400:
            raise arxiv_retrieval.httpx.HTTPStatusError(
                "error", request=None, response=SimpleNamespace(status_code=status_code)
            )

    return SimpleNamespace(text=text, raise_for_status=raise_for_status)


def test_search_parses_entry_into_external_paper(monkeypatch):
    monkeypatch.setattr(arxiv_retrieval.httpx, "get", lambda *a, **k: _fake_response(_ATOM_ENTRY))
    papers = arxiv_retrieval.search_arxiv("leg length COP")
    assert len(papers) == 1
    paper = papers[0]
    assert paper.title == "A Study of Thermoelectric Leg Geometry"
    assert paper.abstract == "Leg length strongly affects COP in Bi2Te3 coolers."
    assert paper.doi == "10.48550/arXiv.2011.02585"
    assert paper.pdf_url == "https://arxiv.org/pdf/2011.02585v2"
    assert paper.source == "arxiv"
    assert paper.year == 2020


def test_empty_feed_returns_empty_list(monkeypatch):
    monkeypatch.setattr(arxiv_retrieval.httpx, "get", lambda *a, **k: _fake_response(_EMPTY_FEED))
    assert arxiv_retrieval.search_arxiv("query") == []


def test_http_error_degrades_to_empty_list(monkeypatch):
    def _raise(*args, **kwargs):
        raise arxiv_retrieval.httpx.ConnectTimeout("timed out")

    monkeypatch.setattr(arxiv_retrieval.httpx, "get", _raise)
    assert arxiv_retrieval.search_arxiv("query") == []


def test_unparseable_xml_degrades_to_empty_list(monkeypatch):
    monkeypatch.setattr(arxiv_retrieval.httpx, "get", lambda *a, **k: _fake_response("not xml"))
    assert arxiv_retrieval.search_arxiv("query") == []


def test_doi_strips_version_suffix():
    assert arxiv_retrieval._doi_for_arxiv_id("2011.02585v3") == "10.48550/arXiv.2011.02585"
    assert arxiv_retrieval._doi_for_arxiv_id("2011.02585") == "10.48550/arXiv.2011.02585"


_ENTITY_BOMB = """<?xml version="1.0"?>
<!DOCTYPE feed [
 <!ENTITY a "1234567890">
 <!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">
 <!ENTITY c "&b;&b;&b;&b;&b;&b;&b;&b;&b;&b;">
 <!ENTITY d "&c;&c;&c;&c;&c;&c;&c;&c;&c;&c;">
 <!ENTITY e "&d;&d;&d;&d;&d;&d;&d;&d;&d;&d;">
]>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><id>x</id><title>&e;</title><summary>boom</summary><published>2020</published></entry>
</feed>
"""


def test_entity_expansion_bomb_is_refused_not_parsed(monkeypatch):
    """ElementTree/expat expands internal DTD entities by default — a
    handful of nested <!ENTITY> definitions can blow up to a huge string
    from a few hundred bytes on the wire. Real arXiv responses never
    contain a DOCTYPE, so refusing to parse one entirely is a safe,
    zero-false-positive guard."""
    monkeypatch.setattr(arxiv_retrieval.httpx, "get", lambda *a, **k: _fake_response(_ENTITY_BOMB))
    assert arxiv_retrieval.search_arxiv("query") == []
