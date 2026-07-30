"""Pydantic models for the sciencerag.priors request/response contract (spec §3.3)."""

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field


class TaskContext(BaseModel):
    objective: str | None = None
    constraints: dict[str, float] = Field(default_factory=dict)


class PriorsRequest(BaseModel):
    query: str
    task_context: TaskContext = Field(default_factory=TaskContext)
    max_priors: int = Field(default=5, ge=1)
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
    # Single-parameter priors (parameter_range/caution) use `field`;
    # relationships spanning multiple contract parameters
    # (scaling_relationship/candidate_config) use `related_fields` instead
    # and leave `field` null (spec §3.6). Both are
    # validated against sim_params.json's geometry_free names at extraction
    # time (see extract.py), not here — this schema stays permissive so
    # non-pipeline callers (tests, fixtures) aren't coupled to the contract.
    field: str | None = None
    related_fields: list[str] = Field(default_factory=list)
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
