"""Regression fixture run for sciencerag.validate (spec §8).

Unlike priors' regression fixtures, these run the REAL validate pipeline
directly (see sciencerag/validate/regression.py's docstring for why: no LLM
call in this path, so there's no cheap/expensive split to make — every
fixture here doubles as both the format check and the real pipeline run).
"""

from pathlib import Path

import pytest

from sciencerag.validate.models import ValidateResponse
from sciencerag.validate.regression import build_request, check_fixture, load_fixtures
from sciencerag.validate.checks import run_anomaly_checks
from sciencerag.validate.evaluation import evaluate

FIXTURES_PATH = Path(__file__).resolve().parent / "fixtures" / "validate_regression.json"
FIXTURES = load_fixtures(FIXTURES_PATH)


def test_fixture_file_has_both_block_and_no_block_categories():
    categories = {fixture.category for fixture in FIXTURES}
    assert "should_block" in categories
    assert "should_not_block" in categories


@pytest.mark.parametrize("fixture", FIXTURES, ids=[f.id for f in FIXTURES])
def test_regression_fixture(fixture):
    request = build_request(fixture)
    anomalies = run_anomaly_checks(request)
    evaluation = evaluate(request)
    blocked = any(a.severity == "blocking" for a in anomalies)
    response = ValidateResponse(
        anomalies=anomalies,
        evaluation=evaluation,
        update_package={"surrogate_update": None, "kg_candidates": [], "blocked": blocked},
        trace_id="tr_regression_test",
    )
    violations = check_fixture(fixture, response)
    assert not violations, f"{fixture.id}: {violations}"
