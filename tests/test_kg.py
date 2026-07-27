"""M1-14: KG query stub always returns no hits until M2+ wires up real
graph storage (spec §3.2 cold-start behavior)."""

from sciencerag.priors.kg import query_kg


def test_query_kg_always_returns_no_hits():
    assert query_kg("any query at all") == []
