"""Pydantic models for the sciencerag.report request/response contract
(spec §5).

Reuses validate's Anomaly/Evaluation/UpdatePackage types directly for the
"异常与注意事项" / "与文献先验的对比" / "更新提案摘要" sections rather than
re-deriving equivalent shapes — spec §5 says these sections directly cite
component B's output, so the report *is* that output, framed for reading,
not a re-computation of it.
"""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from sciencerag.common.validators import reject_non_finite_values, reject_path_unsafe_id
from sciencerag.priors.models import Prior, Source, TaskContext
from sciencerag.validate.models import Anomaly, Evaluation, UpdatePackage


class ReportRequest(BaseModel):
    run_id: str
    task_context: TaskContext = Field(default_factory=TaskContext)
    design_parameters: dict[str, float] = Field(default_factory=dict)
    # Kept in sync with ValidateRequest.n_pairs's bound (validate/models.py)
    # — see that field's docstring for why 500, not 20.
    n_pairs: int = Field(default=1, ge=1, le=500)
    scalar_results: dict[str, float] = Field(default_factory=dict)
    # Capped to match ValidateRequest.priors (validate/models.py:57) — same
    # field, same upstream shape, same reasoning.
    priors: list[Prior] = Field(default_factory=list, max_length=64)
    # 2026-08-17 adversarial-review finding: this had no cap at all, unlike
    # every sibling list field in this API family (ValidateRequest.priors
    # above, batch_evidence's candidates cap from the same review round).
    # Only one check type exists today (Anomaly.check: Literal["ood"]), so
    # a real caller submits at most a handful of these — but nothing
    # server-side computes this list, it's raw client input straight into
    # render_pdf()'s Markdown->HTML->PDF pipeline. Confirmed live: 8,000
    # anomalies (a few-MB JSON body, trivial to construct) produced a 2MB
    # Markdown document and took >10s of CPU in xhtml2pdf per render — and
    # GET /sciencerag/reports/{stem}/pdf re-renders from scratch on every
    # call (see router.py's docstring), so one such report, generated
    # once, lets any caller repeatedly trigger that cost for free. 200
    # keeps real multi-anomaly runs unconstrained in practice while
    # bounding the render cost of a single request.
    anomalies: list[Anomaly] = Field(default_factory=list, max_length=200)
    evaluation: Evaluation
    update_package: UpdatePackage

    _validate_run_id = field_validator("run_id")(reject_path_unsafe_id)
    _validate_design_parameters = field_validator("design_parameters")(reject_non_finite_values)
    _validate_scalar_results = field_validator("scalar_results")(reject_non_finite_values)


class KeyResult(BaseModel):
    field: str
    value: float
    unit: str | None = None
    # Qualitative, not a numeric uncertainty band — tec_surrogate has no
    # calibrated error model to draw a real confidence interval from (see
    # sciencerag/validate/checks.py's baseline-relative severity note).
    # "check_flagged" means >=1 anomaly at warning/blocking severity;
    # fabricating a number here would misrepresent what's actually known.
    confidence_label: Literal["high", "check_flagged", "no_anomaly_data"]


class ReportResponse(BaseModel):
    status: Literal["ok"] = "ok"
    run_id: str
    generated_at: str
    objective_and_constraints: TaskContext
    spec_summary: dict[str, float]
    key_results: list[KeyResult]
    literature_comparison: Evaluation
    anomalies_and_cautions: list[Anomaly]
    update_proposal_summary: UpdatePackage
    citations: list[Source]
    markdown: str
    trace_id: str

    @staticmethod
    def now_iso() -> str:
        return datetime.now(UTC).isoformat()
