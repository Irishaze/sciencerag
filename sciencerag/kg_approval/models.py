"""Pydantic models for the sciencerag.kg_approval web panel — HTTP
equivalent of scripts/approve_kg_candidates.py's --list-pending / --list /
--approve flags."""

from typing import Literal

from pydantic import BaseModel, Field

from sciencerag.validate.models import KGCandidate


class PendingBatchSummary(BaseModel):
    stem: str
    count: int


class PendingBatchDetail(BaseModel):
    stem: str
    candidates: list[KGCandidate]


class ApproveRequest(BaseModel):
    # Either approve_all, or an explicit list of 0-based indices into the
    # batch — mirrors the CLI's mutually-exclusive --approve-all/--approve
    # flags. Indices not listed are neither approved nor lost: the whole
    # batch still gets archived afterward (matching the CLI's own
    # behavior — see kg_candidate_store.archive_pending's docstring), so an
    # unapproved candidate just stops showing up in the pending queue; the
    # archived file still has it.
    approve_all: bool = False
    # max_length is a DoS guard, not a real business limit — a real pending
    # batch (one validate run, or one literature-seeding query) never comes
    # close to this. Confirmed via a real adversarial test: an unbounded
    # list let a single request with 5,000,000 out-of-range indices burn
    # ~9s of server CPU and produce a 409MB response body, from a request
    # body a few hundred KB — attacker cost negligible, server cost large.
    indices: list[int] = Field(default_factory=list, max_length=1000)
    operator: str = "web"
    reason: str = ""


class ApprovalResult(BaseModel):
    index: int
    status: Literal["added", "merged", "conflict", "error"]
    triple_id: str | None = None
    error: str | None = None


class ApproveResponse(BaseModel):
    stem: str
    results: list[ApprovalResult]
    archived: bool
