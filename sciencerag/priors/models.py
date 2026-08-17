"""Pydantic models for the sciencerag.priors request/response contract (spec §3.3)."""

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from sciencerag.common.validators import reject_non_finite_optional


class TaskContext(BaseModel):
    objective: str | None = None
    constraints: dict[str, float] = Field(default_factory=dict)


class PriorsRequest(BaseModel):
    query: str
    task_context: TaskContext = Field(default_factory=TaskContext)
    # Raised from 5 (2026-08-17): real audit-log data (975 real requests)
    # showed 99.7% of callers never override the default, so the default is
    # the only lever that actually changes behavior — and 12 matches the
    # true ceiling (contract.GEOMETRY_FREE_NAMES has 12 fields; recurring
    # real broad queries topped out at 7-8 before being truncated). Costs
    # nothing for the ~96% of narrow (0-1 field) queries that never
    # approach the cap either way — max_priors only trims the already-
    # extracted result, it never changes what extract_priors drafts or how
    # many LLM calls that costs (see retrieval.py's _cap_priors).
    max_priors: int = Field(default=12, ge=1)
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

    _validate_min = field_validator("min")(reject_non_finite_optional)
    _validate_max = field_validator("max")(reject_non_finite_optional)
    _validate_typical = field_validator("typical")(reject_non_finite_optional)

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


class RankedCandidateEntry(BaseModel):
    # Absolute rank within the full KG population (kg.KGRankingResult.
    # ranked), not within this list — can have gaps when some ranked
    # entities were filtered out of `candidates` (e.g. fewer than 2 real
    # contract parameters, see _kg_priors_from_group), which is meant to be
    # visible rather than silently renumbered away.
    rank: int = Field(ge=1)
    parameters: dict[str, str | float]
    reported_performance: dict[str, str | float] = Field(default_factory=dict)
    triple_ids: list[str] = Field(min_length=1)


class RankedCandidateSetValue(BaseModel):
    """One superlative-ranked KG query's full result, e.g. "哪个设计最优电流最大"
    against 5 simulated designs — kept as ONE prior instead of 5 separate
    candidate_config priors that would otherwise all compete individually
    for the same max_priors budget, even though they answer a single
    question (spec discussion 2026-08-17: 5 candidates answering the same
    question are one finding, not five)."""

    relation: str
    relation_description: str | None = None
    direction: Literal["max", "min"]
    # The true population size from kg.KGRankingResult.total_candidates —
    # can exceed len(candidates) when some ranked entities didn't produce a
    # usable Prior (see RankedCandidateEntry.rank's docstring); reported
    # as-is rather than silently narrowed to match, so a reader can tell
    # candidates were dropped.
    total_candidates: int = Field(ge=1)
    candidates: list[RankedCandidateEntry]

    @model_validator(mode="after")
    def _min_two_candidates(self) -> "RankedCandidateSetValue":
        if len(self.candidates) < 2:
            raise ValueError("ranked_candidate_set needs >= 2 candidates (else it's just a candidate_config)")
        return self


_VALUE_MODEL_BY_KIND: dict[str, type[BaseModel]] = {
    "parameter_range": ParameterRangeValue,
    "material_property": MaterialPropertyValue,
    "scaling_relationship": ScalingRelationshipValue,
    "candidate_config": CandidateConfigValue,
    "caution": CautionValue,
    "ranked_candidate_set": RankedCandidateSetValue,
}


class Prior(BaseModel):
    prior_id: str
    kind: Literal[
        "parameter_range",
        "material_property",
        "scaling_relationship",
        "candidate_config",
        "caution",
        "ranked_candidate_set",
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
        | RankedCandidateSetValue
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

    @model_validator(mode="after")
    def _value_field_names_match_prior_fields(self) -> "Prior":
        """The parameter name(s) a prior is about are the single source of
        truth on `field`/`related_fields` (that's what extract.py's contract
        check validates against sim_params.json) — every value kind that
        names its own parameter(s) internally must name the SAME ones, or a
        prior could silently point `field`/`related_fields` at one parameter
        while its value content is actually about another.

        material_property and caution are deliberately exempt: caution's
        `value` is prose with no parameter name to cross-check (the
        statement's applicability is exactly what field/related_fields
        already says), and material_property isn't a contract-field kind at
        all (materials are fixed, exempt from the contract check, filtered
        out downstream — see extract.py)."""
        if isinstance(self.value, ParameterRangeValue):
            if self.value.field_name != self.field:
                raise ValueError(
                    f"parameter_range value.field_name={self.value.field_name!r} must "
                    f"match prior.field={self.field!r}"
                )
        elif isinstance(self.value, ScalingRelationshipValue):
            value_fields = {self.value.x, self.value.y}
            if value_fields != set(self.related_fields):
                raise ValueError(
                    f"scaling_relationship value.x/y={sorted(value_fields)} must match "
                    f"prior.related_fields={sorted(self.related_fields)}"
                )
        elif isinstance(self.value, CandidateConfigValue):
            value_fields = set(self.value.parameters)
            if value_fields != set(self.related_fields):
                raise ValueError(
                    f"candidate_config value.parameters keys={sorted(value_fields)} must "
                    f"match prior.related_fields={sorted(self.related_fields)}"
                )
        elif isinstance(self.value, RankedCandidateSetValue):
            value_fields = {name for entry in self.value.candidates for name in entry.parameters}
            if value_fields != set(self.related_fields):
                raise ValueError(
                    f"ranked_candidate_set candidates' parameters keys={sorted(value_fields)} must "
                    f"match prior.related_fields={sorted(self.related_fields)}"
                )
        return self


class Coverage(BaseModel):
    internal_hits: int
    external_hits: int
    gaps: list[str] = Field(default_factory=list)


class PriorsResponse(BaseModel):
    status: Literal["ok"] = "ok"
    priors: list[Prior]
    coverage: Coverage
    trace_id: str
