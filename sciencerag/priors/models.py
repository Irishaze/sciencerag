"""Pydantic models for the sciencerag.priors request/response contract (spec §3.3)."""

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, model_validator


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


class ParameterRangeValue(BaseModel):
    field_name: str
    min: float | None = None
    max: float | None = None
    typical: float | None = None
    unit: str
    conditions: dict[str, str | float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _at_least_one_number(self) -> "ParameterRangeValue":
        if self.min is None and self.max is None and self.typical is None:
            raise ValueError("parameter_range requires at least one of min/max/typical")
        return self


class MaterialPropertyValue(BaseModel):
    material: str
    form: str | None = None
    property_name: str
    magnitude: float | None = None
    unit: str | None = None
    conditions: dict[str, str | float] = Field(default_factory=dict)
    method: Literal["measured", "computed", "cited", "unknown"] = "unknown"

    @model_validator(mode="after")
    def _magnitude_requires_unit(self) -> "MaterialPropertyValue":
        if self.magnitude is not None and not self.unit:
            raise ValueError("magnitude requires unit")
        return self


class ScalingRelationshipValue(BaseModel):
    x: str
    y: str
    direction: Literal[
        "positive", "negative", "convex", "concave", "non_monotonic", "unknown"
    ]
    functional_form: str | None = None
    validity_range: str | None = None


class CandidateConfigValue(BaseModel):
    parameters: dict[str, str | float]
    reported_performance: dict[str, str | float] = Field(default_factory=dict)
    context: str | None = None

    @model_validator(mode="after")
    def _min_two_params(self) -> "CandidateConfigValue":
        if len(self.parameters) < 2:
            raise ValueError("candidate_config needs >= 2 parameters")
        return self


class CautionValue(BaseModel):
    statement: str
    applicability_scope: str | None = None


_VALUE_MODEL_BY_KIND: dict[str, type[BaseModel]] = {
    "parameter_range": ParameterRangeValue,
    "material_property": MaterialPropertyValue,
    "scaling_relationship": ScalingRelationshipValue,
    "candidate_config": CandidateConfigValue,
    "caution": CautionValue,
}


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
    value: (
        ParameterRangeValue
        | MaterialPropertyValue
        | ScalingRelationshipValue
        | CandidateConfigValue
        | CautionValue
    )
    confidence: float = Field(ge=0, le=1)
    sources: list[Source] = Field(min_length=1)
    notes: str | None = None
    provenance: Literal["internal", "external_unverified"] = "internal"

    @model_validator(mode="before")
    @classmethod
    def _dispatch_value_by_kind(cls, data: Any) -> Any:
        """`value`'s shape depends on the sibling `kind` field, which sits
        outside the union itself — Pydantic's built-in discriminated-union
        tag has to live inside each member, so it can't key off a sibling.
        Resolve the right member by hand before the union ever sees it."""
        if not isinstance(data, dict):
            return data
        raw_value = data.get("value")
        model_cls = _VALUE_MODEL_BY_KIND.get(data.get("kind"))
        if model_cls is not None and isinstance(raw_value, dict):
            data = {**data, "value": model_cls.model_validate(raw_value)}
        return data


class Coverage(BaseModel):
    internal_hits: int
    external_hits: int
    gaps: list[str] = Field(default_factory=list)


class PriorsResponse(BaseModel):
    status: Literal["ok"] = "ok"
    priors: list[Prior]
    coverage: Coverage
    trace_id: str
