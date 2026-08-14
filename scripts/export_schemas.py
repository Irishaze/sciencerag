"""Regenerate the frozen JSON Schema artifacts under sciencerag/schemas/.

Run after any change to the Pydantic models in sciencerag/*/models.py:

    uv run python scripts/export_schemas.py
"""

import json
from pathlib import Path

from sciencerag.ask.models import AskRequest, AskResponse
from sciencerag.common.errors import ErrorResponse
from sciencerag.kg_approval.models import ApproveRequest, ApproveResponse, PendingBatchDetail, PendingBatchSummary
from sciencerag.priors.batch_evidence import BatchEvidenceRequest, BatchEvidenceResponse
from sciencerag.priors.models import PriorsRequest, PriorsResponse
from sciencerag.report.models import ReportRequest, ReportResponse
from sciencerag.validate.models import ValidateRequest, ValidateResponse

SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "sciencerag" / "schemas"

# spec §8: every endpoint's error envelope is part of the published
# contract too, not just its success shape — ErrorResponse goes in all of
# them, not just priors'.
_ENDPOINTS: dict[str, dict[str, type]] = {
    "priors": {
        "PriorsRequest": PriorsRequest,
        "PriorsResponse": PriorsResponse,
        # spec §3.4 (M6) — same router/package as priors, not a separate
        # spec-numbered endpoint, so it shares priors.schema.json.
        "BatchEvidenceRequest": BatchEvidenceRequest,
        "BatchEvidenceResponse": BatchEvidenceResponse,
    },
    "validate": {"ValidateRequest": ValidateRequest, "ValidateResponse": ValidateResponse},
    "report": {"ReportRequest": ReportRequest, "ReportResponse": ReportResponse},
    "ask": {"AskRequest": AskRequest, "AskResponse": AskResponse},
    "kg_approval": {
        "PendingBatchSummary": PendingBatchSummary,
        "PendingBatchDetail": PendingBatchDetail,
        "ApproveRequest": ApproveRequest,
        "ApproveResponse": ApproveResponse,
    },
}


def main() -> None:
    for name, models in _ENDPOINTS.items():
        schema = {key: model.model_json_schema() for key, model in models.items()}
        schema["ErrorResponse"] = ErrorResponse.model_json_schema()
        out_path = SCHEMAS_DIR / f"{name}.schema.json"
        out_path.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
