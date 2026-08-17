"""Pydantic models for the sciencerag.ask request/response contract
(spec §6.2/§6.3)."""

from typing import Literal

from pydantic import BaseModel, Field

from sciencerag.priors.models import Source


class AskRequest(BaseModel):
    question: str
    max_hits: int = Field(default=10, ge=1, le=50)


class SubgraphNode(BaseModel):
    id: str
    kind: Literal["entity", "value"]
    label: str
    entity_type: str | None = None


class SubgraphEdge(BaseModel):
    source: str
    target: str
    relation: str
    # AI-generated plain-Chinese phrase for what `relation` means (see
    # sciencerag/priors/ontology_generator.py:describe_relation) — null for
    # older triples written before this field existed.
    description: str | None = None
    triple_id: str
    confidence: float
    # triple_id of the triple this one contradicts on the same
    # entity_id+relation (kg.py's KGTriple.conflicts_with) — null when this
    # triple doesn't conflict with anything currently in the graph.
    conflicts_with: str | None = None
    # What this specific reading was measured/reported under (e.g.
    # n_pairs for a literature-range claim, full geometry for a
    # simulation-derived achieves_* fact) — see kg.py's KGTriple.conditions.
    # Empty for structural link triples (SIMULATION_USES_*), which carry no
    # conditions of their own.
    conditions: dict[str, float] = Field(default_factory=dict)
    # Free-form backing detail — for simulation-derived facts this is the
    # benchmark-comparison verdict/deviation (kg.py:117-125); for
    # literature-derived facts this is the extraction method/notes
    # (sciencerag/validate/kg_approval.py's DOI-less fallback shape). None
    # when no such detail was ever recorded.
    evidence_detail: dict | None = None


class Subgraph(BaseModel):
    nodes: list[SubgraphNode] = Field(default_factory=list)
    edges: list[SubgraphEdge] = Field(default_factory=list)


class AskResponse(BaseModel):
    status: Literal["ok"] = "ok"
    answer: str
    subgraph: Subgraph
    sources: list[Source]
    # spec §6.2: "若图谱覆盖不足...可选择回退到组件 A 的文档检索,回退情况在
    # 响应中明确标注" — both fields exist so a caller never has to guess
    # which answer path produced this response.
    fallback_used: bool
    coverage_note: str | None = None
    trace_id: str
