"""Pydantic models for the sciencerag.priors request/response contract (spec §3.3)."""

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field


class TaskContext(BaseModel):
    objective: str | None = None
    constraints: dict[str, float] = Field(default_factory=dict)
    materials_hint: list[str] = Field(default_factory=list)


class PriorsRequest(BaseModel):
    query: str
    task_context: TaskContext = Field(default_factory=TaskContext)
    max_priors: int = 5
    allow_external: bool = False


class SourcePaper(BaseModel):
    type: Literal["paper"] = "paper"
    doi: str
    span: str | None = None


class SourceKGTriple(BaseModel):
    type: Literal["kg_triple"] = "kg_triple"
    triple_id: str


Source = Annotated[Union[SourcePaper, SourceKGTriple], Field(discriminator="type")]


class Prior(BaseModel):
    prior_id: str
    kind: Literal[
        "parameter_range",
        "material_property",
        "scaling_relationship",
        "candidate_config",
        "caution",
    ]
    field: str
    value: dict[str, Any]
    confidence: float = Field(ge=0, le=1)
    sources: list[Source] = Field(min_length=1)
    notes: str | None = None
    provenance: Literal["internal", "external_unverified"] = "internal"


class Coverage(BaseModel):
    internal_hits: int
    external_hits: int
    gaps: list[str] = Field(default_factory=list)


class PriorsResponse(BaseModel):
    status: Literal["ok"] = "ok"
    priors: list[Prior]
    coverage: Coverage
    trace_id: str
