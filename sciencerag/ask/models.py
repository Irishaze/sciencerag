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
    triple_id: str
    confidence: float


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
