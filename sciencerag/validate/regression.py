"""Regression fixture format for sciencerag.validate (spec §8).

spec §8: "验证侧为(已知有缺陷的历史运行 → 预期异常判定,含应当 blocking 与不应
误伤的两类案例)". Unlike priors' regression set (sciencerag/priors/regression.py),
validate has no LLM call in its path — every check here is deterministic
local numpy/torch computation over fixed bundled data, so there's no
cheap-offline-check vs. expensive-real-run split to make: these fixtures run
the real pipeline directly and are still fast/free (see
tests/test_validate_regression.py).
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from sciencerag.validate.models import ValidateRequest, ValidateResponse

_SEVERITY_RANK = {"info": 0, "warning": 1, "blocking": 2}


class ValidateRegressionFixture(BaseModel):
    id: str
    category: Literal["should_block", "should_not_block", "should_flag_deviation", "should_be_consistent"]
    notes: str = ""
    request: dict
    expected_blocked: bool
    expected_verdict: (
        Literal["consistent", "deviation_found", "insufficient_benchmark"] | None
    ) = None
    # Minimum expected severity per check (e.g. {"ood": "blocking"}) — a
    # floor, not an exact match, since evidence values (residual magnitudes,
    # percentiles) can drift slightly as bundled data/checkpoints change.
    expected_min_severity_by_check: dict[str, Literal["info", "warning", "blocking"]] = Field(
        default_factory=dict
    )
    # Ceiling, not floor — for "known-good input must not be falsely
    # flagged" fixtures, which expected_min_severity_by_check alone can't
    # express (a minimum of "info" is trivially satisfied by any severity).
    expected_max_severity_by_check: dict[str, Literal["info", "warning", "blocking"]] = Field(
        default_factory=dict
    )


def load_fixtures(path: Path) -> list[ValidateRegressionFixture]:
    data = json.loads(path.read_text())
    return [ValidateRegressionFixture.model_validate(item) for item in data]


def build_request(fixture: ValidateRegressionFixture) -> ValidateRequest:
    return ValidateRequest.model_validate(fixture.request)


def check_fixture(fixture: ValidateRegressionFixture, response: ValidateResponse) -> list[str]:
    """Check `response` against `fixture`'s properties. Returns a list of
    human-readable violation messages; empty list means it passed."""
    violations = []

    if response.update_package.blocked != fixture.expected_blocked:
        violations.append(
            f"expected blocked={fixture.expected_blocked}, got {response.update_package.blocked}"
        )

    if fixture.expected_verdict is not None and response.evaluation.verdict != fixture.expected_verdict:
        violations.append(
            f"expected evaluation.verdict={fixture.expected_verdict!r}, "
            f"got {response.evaluation.verdict!r}"
        )

    severity_by_check = {a.check: a.severity for a in response.anomalies}
    for check, min_severity in fixture.expected_min_severity_by_check.items():
        actual = severity_by_check.get(check)
        if actual is None:
            violations.append(f"expected an anomaly entry for check={check!r}, found none")
            continue
        if _SEVERITY_RANK[actual] < _SEVERITY_RANK[min_severity]:
            violations.append(
                f"check={check!r}: expected severity >= {min_severity!r}, got {actual!r}"
            )

    for check, max_severity in fixture.expected_max_severity_by_check.items():
        actual = severity_by_check.get(check)
        if actual is None:
            violations.append(f"expected an anomaly entry for check={check!r}, found none")
            continue
        if _SEVERITY_RANK[actual] > _SEVERITY_RANK[max_severity]:
            violations.append(
                f"check={check!r}: expected severity <= {max_severity!r}, got {actual!r}"
            )

    return violations
