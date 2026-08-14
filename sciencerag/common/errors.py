"""Shared error envelope and error categories for all sciencerag endpoints (spec §8)."""

from typing import Any, Literal

from pydantic import BaseModel, Field

ErrorCategory = Literal[
    "retrieval_timeout",
    "schema_validation_failed",
    "insufficient_coverage",
    "benchmark_unavailable",
    "not_found",
    "invalid_request",
]


class ErrorDetail(BaseModel):
    category: ErrorCategory
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    error: ErrorDetail
    trace_id: str
