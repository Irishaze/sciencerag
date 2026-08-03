"""Knowledge-graph storage + retrieval for sciencerag.priors (spec §3.2) and
sciencerag.ask (spec §6).

Real graph storage (NetworkX + JSON stub through M1-M4, upgraded here for
M5 since `ask` needs actual subgraph traversal — see the M1-era docstring
this replaces). Persisted as a flat JSON triple list under data/kg/ rather
than a real graph database: the spec explicitly scopes this project's
graph to that until `ask` needs more (§6.1: "v1 无需修改 schema"), and a
flat file keeps the M1-M4 cold-start behavior (query_kg returns hits from
whatever is actually stored, [] when nothing is) unchanged in shape.

Writes only happen through add_triple(), and only sciencerag.validate's §4.4
KG candidates (via an approval step — scripts/approve_kg_candidates.py,
spec §7: "v1 可用命令行脚本替代页面") ever call it — this module itself does
not gate who is allowed to write; that's the caller's job (spec §6.3: the
only path into the graph is candidate -> approval -> registration).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
import uuid

from pydantic import BaseModel, Field

GRAPH_PATH = Path("data/kg/graph.json")

# spec §3.8's numeric-groundedness tolerance (±2%, "容忍 LLM 抄证据时的四舍
# 五入") reused here for the same reason: two triples for the same
# subject/relation/conditions should be treated as the same finding if
# they're within ordinary rounding/reporting noise of each other, not
# flagged as a spurious conflict.
DUPLICATE_VALUE_RELATIVE_TOLERANCE = 0.02


class KGHit(BaseModel):
    triple_id: str
    text: str
    relevance: float


class KGSource(BaseModel):
    type: Literal["paper", "kg_triple", "run"] = "run"
    doi: str | None = None
    triple_id: str | None = None
    run_id: str | None = None


class KGTriple(BaseModel):
    triple_id: str
    subject: str
    relation: str
    object_value: float
    object_unit: str | None = None
    conditions: dict[str, float] = Field(default_factory=dict)
    confidence: float
    run_ids: list[str] = Field(default_factory=list)
    sources: list[KGSource] = Field(default_factory=list)
    # Set only when this triple was stored because it *disagreed* with an
    # existing one for the same subject/relation/conditions (spec §4.4:
    # "存在但数值冲突 → 标记冲突,双方来源并列呈现,不自动覆盖") — both
    # triples stay in the graph, neither is silently overwritten.
    conflicts_with: str | None = None
    created_at: str


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _load_triples() -> list[KGTriple]:
    if not GRAPH_PATH.exists():
        return []
    data = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    return [KGTriple.model_validate(item) for item in data]


def _save_triples(triples: list[KGTriple]) -> None:
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    GRAPH_PATH.write_text(
        json.dumps([t.model_dump() for t in triples], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _find_matching_condition_triple(
    triples: list[KGTriple], subject: str, relation: str, conditions: dict[str, float]
) -> KGTriple | None:
    for triple in triples:
        if (
            triple.subject == subject
            and triple.relation == relation
            and triple.conflicts_with is None
            and triple.conditions == conditions
        ):
            return triple
    return None


def add_triple(
    *,
    subject: str,
    relation: str,
    object_value: float,
    object_unit: str | None,
    conditions: dict[str, float],
    confidence: float,
    run_id: str,
    sources: list[KGSource],
) -> tuple[KGTriple, Literal["added", "merged", "conflict"]]:
    """The only write path into the graph (spec §6.3). Callers are expected
    to have already run this through human approval."""
    triples = _load_triples()
    existing = _find_matching_condition_triple(triples, subject, relation, conditions)

    if existing is not None:
        tolerance = DUPLICATE_VALUE_RELATIVE_TOLERANCE * max(abs(existing.object_value), 1e-9)
        if abs(existing.object_value - object_value) <= tolerance:
            if run_id not in existing.run_ids:
                existing.run_ids.append(run_id)
            existing.sources.extend(sources)
            _save_triples(triples)
            return existing, "merged"

        conflict = KGTriple(
            triple_id=f"kg_{uuid.uuid4().hex[:12]}",
            subject=subject,
            relation=relation,
            object_value=object_value,
            object_unit=object_unit,
            conditions=conditions,
            confidence=confidence,
            run_ids=[run_id],
            sources=sources,
            conflicts_with=existing.triple_id,
            created_at=_now_iso(),
        )
        triples.append(conflict)
        _save_triples(triples)
        return conflict, "conflict"

    new_triple = KGTriple(
        triple_id=f"kg_{uuid.uuid4().hex[:12]}",
        subject=subject,
        relation=relation,
        object_value=object_value,
        object_unit=object_unit,
        conditions=conditions,
        confidence=confidence,
        run_ids=[run_id],
        sources=sources,
        created_at=_now_iso(),
    )
    triples.append(new_triple)
    _save_triples(triples)
    return new_triple, "added"


def _render_text(triple: KGTriple) -> str:
    unit = triple.object_unit or ""
    conditions = ", ".join(f"{k}={v}" for k, v in sorted(triple.conditions.items()))
    return f"{triple.subject} {triple.relation} {triple.object_value}{unit} ({conditions})"


def query_kg(query: str, max_hits: int = 10) -> list[KGHit]:
    """Keyword-overlap search over stored triples — deliberately simple
    (no embeddings/vector index for the graph, spec §6.1 keeps v1's schema
    minimal). Empty graph or no overlapping terms both correctly return []
    (spec §3.2 cold-start: "图谱查询命中 0 条,检索自动落到论文库")."""
    triples = _load_triples()
    if not triples:
        return []
    query_terms = {term for term in query.lower().split() if term}
    if not query_terms:
        return []

    hits = []
    for triple in triples:
        text = _render_text(triple)
        text_terms = {term for term in text.lower().replace("(", " ").replace(")", " ").split() if term}
        overlap = query_terms & text_terms
        if not overlap:
            continue
        relevance = len(overlap) / len(query_terms)
        hits.append(KGHit(triple_id=triple.triple_id, text=text, relevance=relevance))

    hits.sort(key=lambda hit: hit.relevance, reverse=True)
    return hits[:max_hits]


def get_triples_by_ids(triple_ids: list[str]) -> list[KGTriple]:
    id_set = set(triple_ids)
    return [t for t in _load_triples() if t.triple_id in id_set]


def subgraph_from_triples(triples: list[KGTriple]) -> dict:
    """Nodes/edges for a given triple set — the response shape
    sciencerag.ask returns alongside its answer (spec §6.2: "返回答案并附带
    子图")."""
    nodes = {}
    edges = []
    for triple in triples:
        nodes.setdefault(triple.subject, {"id": triple.subject, "kind": "entity"})
        object_id = f"{triple.relation}={triple.object_value}{triple.object_unit or ''}"
        nodes.setdefault(object_id, {"id": object_id, "kind": "value"})
        edges.append(
            {
                "source": triple.subject,
                "target": object_id,
                "relation": triple.relation,
                "triple_id": triple.triple_id,
                "confidence": triple.confidence,
            }
        )
    return {"nodes": list(nodes.values()), "edges": edges}


def get_subgraph(entities: list[str]) -> dict:
    """subgraph_from_triples(), scoped to triples touching any of
    `entities` by subject match."""
    triples = _load_triples()
    matched = [t for t in triples if t.subject in entities or any(e in t.relation for e in entities)]
    return subgraph_from_triples(matched)
