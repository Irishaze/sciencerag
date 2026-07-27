"""Knowledge-graph retrieval stub for sciencerag.priors (spec §3.2, §6.1).

Spec §3.2: the KG has the HIGHEST query priority — it holds findings from
the system's own past simulation runs, which no external literature corpus
has. "Priority" means query ORDER, not a hard dependency: during cold
start (M1-M4, before any simulation runs have been approved into the
graph), a KG query returns zero hits and retrieval falls back to the
literature corpus (retrieval.py's PaperQA2 path) — the endpoint still
returns normally, with priors sourced entirely from literature (spec:
"图谱查询命中 0 条,检索自动落到论文库...端点照常返回,此时先验全部来自文献").

Real graph storage/traversal (NetworkX + JSON stub through M1-M4, a real
graph DB once `ask` needs actual subgraph traversal in M5) is out of scope
for M1 — this module only exists so the query-priority-order structure is
already in place for M2+ to wire a real implementation into, without
reshaping the retrieval pipeline.
"""

from pydantic import BaseModel


class KGHit(BaseModel):
    triple_id: str
    text: str
    relevance: float


def query_kg(query: str) -> list[KGHit]:
    """Always returns no hits until M2+ wires up real graph storage."""
    return []
