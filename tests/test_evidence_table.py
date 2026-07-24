"""Tests for _build_evidence_table's low-relevance filter (sciencerag/priors/retrieval.py).

Uses lightweight fake context objects (matching the .context/.score/.text.doc/
.text.name shape PaperQA2's Context provides) — no real PaperQA2/API calls.
"""

from types import SimpleNamespace

from sciencerag.priors.retrieval import MIN_EVIDENCE_RELEVANCE, _build_evidence_table


def _fake_context(text: str, score: int, doi: str = "10.0000/x", title: str = "Paper"):
    doc = SimpleNamespace(doi=doi, title=title)
    text_obj = SimpleNamespace(doc=doc, name="pages 1-2")
    return SimpleNamespace(context=text, score=score, text=text_obj)


def test_low_relevance_contexts_are_excluded_from_evidence_table():
    contexts = [
        _fake_context("strong evidence", score=9),  # relevance 0.9
        _fake_context("weak/hallucinated evidence", score=6),  # relevance 0.6 < 0.7
        _fake_context("another strong one", score=8),  # relevance 0.8
    ]
    table = _build_evidence_table(contexts)
    assert len(table) == 2
    assert all(item.relevance >= MIN_EVIDENCE_RELEVANCE for item in table.values())
    assert "weak/hallucinated evidence" not in [item.text for item in table.values()]


def test_evidence_labels_are_sequential_without_gaps_after_filtering():
    contexts = [
        _fake_context("a", score=9),
        _fake_context("b", score=6),  # filtered out
        _fake_context("c", score=8),
    ]
    table = _build_evidence_table(contexts)
    assert list(table.keys()) == ["E1", "E2"]
    assert table["E1"].text == "a"
    assert table["E2"].text == "c"


def test_all_low_relevance_yields_empty_table():
    contexts = [_fake_context("weak", score=3), _fake_context("also weak", score=5)]
    table = _build_evidence_table(contexts)
    assert table == {}
