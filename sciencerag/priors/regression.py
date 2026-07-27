"""Regression fixture format for sciencerag.priors (M1-17/18, spec §8).

Fixtures assert PROPERTIES of a real response, not exact text — a real
LLM extraction pipeline's wording is never byte-stable, but "at least N
priors, including a parameter_range, every source a real DOI" is a stable
contract to hold the pipeline to. Shared by:

  - tests/fixtures/priors_regression.json: the actual fixture data
  - scripts/run_regression.py (M1-18): runs each fixture against the real
    pipeline and reports pass/fail
  - tests/test_regression_fixtures.py: a cheap, offline sanity check that
    the fixture FILE itself is well-formed (not a real pipeline run)
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field

from sciencerag.priors.models import PriorsResponse


class RegressionFixture(BaseModel):
    id: str
    query: str
    category: str
    notes: str = ""

    # -- expectations, all property-based (spec §8: "NOT exact-match") --
    min_priors: int = Field(ge=0)
    must_have_kinds: list[str] = Field(default_factory=list)
    must_cite_real_doi: bool = True
    # Explicit escape hatch for queries with genuinely no corpus coverage
    # (verified by hand, not just "the pipeline happened to return 0 this
    # run") — asserts min_priors==0 AND that coverage.gaps explains why,
    # instead of silently treating a zero-hit result as a failure.
    allow_zero_priors: bool = False


def load_fixtures(path: Path) -> list[RegressionFixture]:
    data = json.loads(path.read_text())
    return [RegressionFixture.model_validate(item) for item in data]


def check_fixture(fixture: RegressionFixture, response: PriorsResponse) -> list[str]:
    """Check `response` against `fixture`'s properties. Returns a list of
    human-readable violation messages; empty list means it passed."""
    violations = []

    if fixture.allow_zero_priors:
        if len(response.priors) != 0:
            violations.append(
                f"expected 0 priors (allow_zero_priors=true) but got {len(response.priors)}"
            )
        if not response.coverage.gaps:
            violations.append("allow_zero_priors=true but coverage.gaps is empty")
        return violations

    if len(response.priors) < fixture.min_priors:
        violations.append(
            f"expected >= {fixture.min_priors} priors, got {len(response.priors)}"
        )

    got_kinds = {p.kind for p in response.priors}
    missing_kinds = set(fixture.must_have_kinds) - got_kinds
    if missing_kinds:
        violations.append(f"missing expected kind(s): {sorted(missing_kinds)}")

    if fixture.must_cite_real_doi:
        for prior in response.priors:
            paper_sources = [s for s in prior.sources if s.type == "paper"]
            if not any(s.doi.strip() for s in paper_sources):
                violations.append(
                    f"prior {prior.prior_id!r} (kind={prior.kind}) has no non-empty paper DOI"
                )

    return violations
