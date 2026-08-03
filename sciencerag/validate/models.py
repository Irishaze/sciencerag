"""Pydantic models for the sciencerag.validate request/response contract
(spec §4.5).

M2 scope (spec §10): only the 4.1 anomaly-check and 4.2 evaluation fields are
populated by real logic. `update_package.surrogate_update`/`kg_candidates`
are 4.3/4.4 (M3) — always empty here, with `blocked` enforced as the only
thing that can flip them from "empty because not built yet" to "empty
because this run must not feed downstream learning".
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from sciencerag.priors.models import Prior, Source


class ValidateRequest(BaseModel):
    run_id: str
    # Actual geometry_free parameter values used by this run (spec §3.6
    # contract; subset because the heatsink-geometry surrogate coverage gap
    # means not every contract name has a trained encoder behind it yet —
    # see tec_bridge.SUPPORTED_GEOMETRY_NAMES).
    design_parameters: dict[str, float] = Field(default_factory=dict)
    n_pairs: int = Field(default=1, ge=1, le=20)
    # This run's measured scalar performance (05 standardized result). Keys
    # should be a subset of tec_bridge's SCALAR_NAMES; used for 4.2.1
    # benchmark comparison. Empty => benchmark check reports
    # insufficient_benchmark rather than guessing.
    scalar_results: dict[str, float] = Field(default_factory=dict)
    # Which of the 11 solved one-pair COMSOL operating points (index into
    # tec_1pair_dset3.npz) this run's geometry/current corresponds to. Only
    # real solved field data we have — None skips 4.1.1/4.1.2 with an info
    # anomaly instead of fabricating a residual against data we don't have.
    field_case_index: int | None = Field(default=None, ge=0, le=10)
    # 06's output — produced upstream of validate (spec §4: "调用时机在...
    # 06 潜在状态编码...之后"). None skips 4.1.3 with an info anomaly.
    latent_state: list[float] | None = None
    prior_ids: list[str] = Field(default_factory=list)
    # The actual Prior objects used in planning this run, passed in-band
    # since validate has no priors store to resolve prior_ids against.
    priors: list[Prior] = Field(default_factory=list)


class Anomaly(BaseModel):
    check: Literal["energy_balance", "pde_residual", "ood"]
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
    loss_reweighting: dict[str, float] = Field(default_factory=dict)
    hyperparameter_direction: str


class KGCandidate(BaseModel):
    """spec §4.4: one 材料-结构-工况-性能 candidate triple, with full
    provenance (运行 ID、置信度、支撑数据切片) attached per spec."""

    subject: str
    relation: str
    object_value: float
    object_unit: str | None = None
    conditions: dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(ge=0, le=1)
    run_id: str
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
