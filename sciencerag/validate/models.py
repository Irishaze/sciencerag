"""Pydantic models for the sciencerag.validate request/response contract
(spec §4.5).

M2 scope (spec §10): only the 4.1 anomaly-check and 4.2 evaluation fields are
populated by real logic. `update_package.surrogate_update`/`kg_candidates`
are 4.3/4.4 (M3) — always empty here, with `blocked` enforced as the only
thing that can flip them from "empty because not built yet" to "empty
because this run must not feed downstream learning".
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from sciencerag.common.validators import (
    reject_non_finite_list,
    reject_non_finite_values,
    reject_path_unsafe_id,
)
from sciencerag.priors.models import Prior, Source


class ValidateRequest(BaseModel):
    run_id: str
    # Actual geometry_free parameter values used by this run (spec §3.6
    # contract; subset because the heatsink-geometry surrogate coverage gap
    # means not every contract name has a trained encoder behind it yet —
    # see tec_bridge.SUPPORTED_GEOMETRY_NAMES).
    design_parameters: dict[str, float] = Field(default_factory=dict)
    # Upper bound is a sanity guard against typos/unit mistakes, not a
    # physics limit — commercial multi-pair modules (e.g. the "127" in a
    # part number like TEC1-12706) commonly run into the low hundreds of
    # pairs, and this field only ever feeds benchmark/prior comparison and
    # KG candidate extraction (neither is n_pairs-sensitive beyond carrying
    # it in `conditions`), not a physics model with a trained coverage
    # range to stay inside.
    n_pairs: int = Field(default=1, ge=1, le=500)
    # This run's measured scalar performance (05 standardized result). Keys
    # should be a subset of tec_bridge's SCALAR_NAMES; used for 4.2.1
    # benchmark comparison. Empty => benchmark check reports
    # insufficient_benchmark rather than guessing.
    scalar_results: dict[str, float] = Field(default_factory=dict)

    _validate_run_id = field_validator("run_id")(reject_path_unsafe_id)
    _validate_design_parameters = field_validator("design_parameters")(reject_non_finite_values)
    _validate_scalar_results = field_validator("scalar_results")(reject_non_finite_values)
    # 06's output — produced upstream of validate (spec §4: "调用时机在...
    # 06 潜在状态编码...之后"). None skips the OOD check with an info anomaly.
    latent_state: list[float] | None = None
    prior_ids: list[str] = Field(default_factory=list)
    # The actual Prior objects used in planning this run, passed in-band
    # since validate has no priors store to resolve prior_ids against.
    priors: list[Prior] = Field(default_factory=list)

    _validate_latent_state = field_validator("latent_state")(reject_non_finite_list)


class Anomaly(BaseModel):
    # energy_balance/pde_residual (formerly the other two 4.1 checks) were
    # removed: both depended on tec_surrogate's compositional multi-pair
    # model, which its own training script documents as "not a substitute
    # for multi-pair COMSOL calibration data" — there was no real physics
    # validation being done, just an approximation with a narrow trained
    # range (1-20 pairs) that silently produced misleading confidence for
    # anything outside it. ood needs none of that: it's a straightforward
    # statistical distance check against the training distribution.
    check: Literal["ood"]
    severity: Literal["info", "warning", "blocking"]
    evidence: dict[str, Any] = Field(default_factory=dict)


class Deviation(BaseModel):
    field: str
    # 4.2 has two independent comparison lines (spec §4.2): benchmark vs.
    # known solved cases, and prior vs. the literature ranges from
    # component A. Both feed this one list, tagged by which line produced
    # them, rather than forcing benchmark comparisons into prior-shaped
    # fields (prior_id/typical) that don't apply to them.
    source: Literal["prior_comparison", "benchmark_comparison"]
    reference_id: str
    actual: float
    reference_min: float | None = None
    reference_max: float | None = None
    verdict: Literal["within_range", "deviation"]


class Evaluation(BaseModel):
    verdict: Literal["consistent", "deviation_found", "insufficient_benchmark"]
    deviations: list[Deviation] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)


class RecommendedSample(BaseModel):
    """spec §4.3: "推荐纳入训练集的样本清单(以运行 ID + 区域标识表示)"."""

    run_id: str
    region: str
    reason: str


class SurrogateUpdateSuggestion(BaseModel):
    """spec §4.3 output — a proposal only; nothing here executes training."""

    recommended_training_samples: list[RecommendedSample] = Field(default_factory=list)
    hyperparameter_direction: str


class KGCandidate(BaseModel):
    """spec §4.4: one 材料-结构-工况-性能 candidate triple, with full
    provenance (运行 ID、置信度、支撑数据切片) attached per spec."""

    subject: str
    relation: str
    # Exactly one of object_value or object_entity_id — a measured number
    # ("achieves 71.7K") or a link to another entity ("uses this Material"),
    # mirrors sciencerag/priors/kg.py:KGTriple's own split (see that class's
    # docstring) and the same validator, one layer up.
    object_value: float | None = None
    object_unit: str | None = None
    object_entity_id: str | None = None
    object_entity_label: str | None = None
    object_entity_type: str | None = None
    # AI-generated plain-Chinese phrase for what `relation` means to a
    # non-specialist (sciencerag/priors/ontology_generator.py:
    # describe_relation) — powers the graph UI's node/edge inspector.
    relation_description: str | None = None
    conditions: dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(ge=0, le=1)
    run_id: str
    # AI-classified against data/kg/ontology.json (sciencerag/priors/
    # ontology_generator.py) — which entity type `subject` belongs to.
    # Orthogonal to subject/conditions (which individuate *which* design
    # this is, kg.py's entity_id) — this classifies *what kind* of thing
    # it is.
    entity_type: str

    @model_validator(mode="after")
    def _exactly_one_object_kind(self) -> "KGCandidate":
        has_value = self.object_value is not None
        has_link = self.object_entity_id is not None
        if has_value == has_link:
            raise ValueError("exactly one of object_value or object_entity_id must be set")
        return self
    # spec §4.4: "与图谱中现有三元组做去重与冲突检测" — kg.py's graph storage
    # is still a stub returning zero hits (spec §3.2 cold-start note), so
    # every candidate resolves "new" in practice today; the field exists so
    # a later real-graph implementation can populate it without a schema
    # change (same "wire structure now, real impl later" pattern kg.py's
    # own docstring uses).
    dedup_status: Literal["new", "duplicate_confirmed", "conflict"]
    supporting_evidence: dict[str, Any] = Field(default_factory=dict)


class UpdatePackage(BaseModel):
    surrogate_update: SurrogateUpdateSuggestion | None = None
    kg_candidates: list[KGCandidate] = Field(default_factory=list)
    blocked: bool

    @model_validator(mode="after")
    def _blocked_implies_empty(self) -> "UpdatePackage":
        if self.blocked and (self.surrogate_update is not None or self.kg_candidates):
            raise ValueError("blocked=true requires surrogate_update=null and kg_candidates=[]")
        return self


class ValidateResponse(BaseModel):
    status: Literal["ok"] = "ok"
    anomalies: list[Anomaly]
    evaluation: Evaluation
    update_package: UpdatePackage
    trace_id: str
