"""M1-17: offline checks for the regression fixture format and file.

These are cheap/free — they validate the FIXTURE FILE is well-formed and
that check_fixture()'s pass/fail logic itself is correct, using constructed
PriorsResponse objects. They do NOT call the real pipeline; that's M1-18's
scripts/run_regression.py, a separate real-API-cost regression run.
"""

from pathlib import Path

from sciencerag.priors.models import Coverage, Prior, PriorsResponse, SourcePaper
from sciencerag.priors.regression import check_fixture, load_fixtures

FIXTURES_PATH = Path(__file__).resolve().parent / "fixtures" / "priors_regression.json"


def _priors_response(
    kinds: list[str], dois: list[str] | None = None, gaps: list[str] | None = None
) -> PriorsResponse:
    dois = dois or ["10.0000/x"] * len(kinds)
    return PriorsResponse(
        priors=[
            Prior(
                prior_id=f"pr_{i}",
                kind=kind,
                field="x",
                value={"summary": "x"},
                confidence=0.8,
                sources=[SourcePaper(doi=doi, span="p.1")],
            )
            for i, (kind, doi) in enumerate(zip(kinds, dois, strict=True))
        ],
        coverage=Coverage(internal_hits=len(kinds), external_hits=0, gaps=gaps or []),
        trace_id="tr_test",
    )


def test_fixture_file_loads_and_has_5_to_10_entries():
    fixtures = load_fixtures(FIXTURES_PATH)
    assert 5 <= len(fixtures) <= 10
    assert len({f.id for f in fixtures}) == len(fixtures), "fixture ids must be unique"


def test_fixture_file_covers_all_5_kinds_across_its_must_have_kinds():
    fixtures = load_fixtures(FIXTURES_PATH)
    covered = {k for f in fixtures for k in f.must_have_kinds}
    assert covered == {
        "parameter_range",
        "material_property",
        "scaling_relationship",
        "candidate_config",
        "caution",
    }


def test_check_fixture_passes_when_properties_are_met():
    fixtures = load_fixtures(FIXTURES_PATH)
    fixture = next(f for f in fixtures if f.id == "parameter_range_leg_length")
    response = _priors_response(["parameter_range", "parameter_range", "candidate_config"])
    assert check_fixture(fixture, response) == []


def test_check_fixture_flags_too_few_priors():
    fixtures = load_fixtures(FIXTURES_PATH)
    fixture = next(f for f in fixtures if f.id == "material_property_zt")
    response = _priors_response(["material_property"])  # far below min_priors=8
    violations = check_fixture(fixture, response)
    assert any("priors" in v for v in violations)


def test_check_fixture_flags_missing_required_kind():
    fixtures = load_fixtures(FIXTURES_PATH)
    fixture = next(f for f in fixtures if f.id == "caution_max_cop_limits")
    response = _priors_response(["parameter_range"] * 10)  # no "caution" at all
    violations = check_fixture(fixture, response)
    assert any("caution" in v for v in violations)


def test_check_fixture_flags_missing_doi():
    fixtures = load_fixtures(FIXTURES_PATH)
    fixture = next(f for f in fixtures if f.id == "scaling_relationship_fan_speed")
    response = _priors_response(
        ["scaling_relationship"] * 3, dois=["10.1/a", "", "10.2/b"]
    )
    violations = check_fixture(fixture, response)
    assert any("DOI" in v for v in violations)


def test_check_fixture_allow_zero_priors_requires_gaps_explanation():
    fixtures = load_fixtures(FIXTURES_PATH)
    fixture = next(f for f in fixtures if f.id == "edge_zero_coverage_heat_sink_interface")

    ok_response = _priors_response([], gaps=["no relevant evidence found"])
    assert check_fixture(fixture, ok_response) == []

    unexplained_response = _priors_response([], gaps=[])
    violations = check_fixture(fixture, unexplained_response)
    assert any("gaps" in v for v in violations)

    nonzero_response = _priors_response(["parameter_range"], gaps=["x"])
    violations = check_fixture(fixture, nonzero_response)
    assert any("0 priors" in v for v in violations)
