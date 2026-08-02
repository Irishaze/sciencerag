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

# check_fixture()'s pass/fail logic only cares about kind/sources/gaps — these
# are just minimal valid per-kind shapes so Prior construction succeeds.
# field/related_fields must still agree with the value's own field name(s)
# (see models.py's cross-check), so relationship kinds use related_fields
# instead of field.
_FIELD_AND_VALUE_BY_KIND = {
    "parameter_range": (
        {"field": "x", "related_fields": []},
        {"field_name": "x", "typical": 1.0, "unit": "mm"},
    ),
    "material_property": (
        {"field": "x", "related_fields": []},
        {"material": "Bi2Te3", "property_name": "seebeck_coefficient"},
    ),
    "scaling_relationship": (
        {"field": None, "related_fields": ["leg_length", "cop"]},
        {"x": "leg_length", "y": "cop", "direction": "unknown"},
    ),
    "candidate_config": (
        {"field": None, "related_fields": ["leg_length", "leg_width"]},
        {"parameters": {"leg_length": 0.07, "leg_width": 0.12}},
    ),
    "caution": ({"field": "x", "related_fields": []}, {"statement": "x"}),
}


def _priors_response(
    kinds: list[str], dois: list[str] | None = None, gaps: list[str] | None = None
) -> PriorsResponse:
    dois = dois or ["10.0000/x"] * len(kinds)
    return PriorsResponse(
        priors=[
            Prior(
                prior_id=f"pr_{i}",
                kind=kind,
                value=_FIELD_AND_VALUE_BY_KIND[kind][1],
                confidence=0.8,
                sources=[SourcePaper(doi=doi, span="p.1")],
                **_FIELD_AND_VALUE_BY_KIND[kind][0],
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


def test_known_flaky_defaults_to_false():
    fixtures = load_fixtures(FIXTURES_PATH)
    fixture = next(f for f in fixtures if f.id == "parameter_range_leg_length")
    assert fixture.known_flaky is False


def test_known_flaky_fixtures_are_marked_in_the_real_file():
    """The 3 fixtures found (2026-08-02) to return substantially different
    priors/evidence run-to-run on an unchanged corpus/code — see each
    fixture's own notes for the traced root cause (PaperQA2's agent_llm
    search-path variance) — must stay marked so run_regression.py runs them
    with majority-vote instead of trusting a single attempt."""
    fixtures = load_fixtures(FIXTURES_PATH)
    flaky_ids = {f.id for f in fixtures if f.known_flaky}
    assert flaky_ids == {
        "candidate_config_heat_sink",
        "caution_max_cop_limits",
        "application_battery_thermal_management",
    }


def test_fixture_file_covers_all_producible_kinds_across_its_must_have_kinds():
    """material_property is intentionally excluded from this check: under the
    sim-contract sync (spec §3.6), material is fixed
    (Bi2Te3, prior_target=false) and the extraction pipeline filters out any
    material_property draft before it becomes a Prior — a real pipeline run
    can never legitimately satisfy must_have_kinds=['material_property']
    anymore. `kind` itself is still a 5-value enum (schema-level, unchanged);
    it's specifically the *regression fixtures* that only ever need to cover
    the 4 kinds the pipeline actually produces."""
    fixtures = load_fixtures(FIXTURES_PATH)
    covered = {k for f in fixtures for k in f.must_have_kinds}
    assert covered == {
        "parameter_range",
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
    fixture = next(f for f in fixtures if f.id == "parameter_range_conductor_thickness")
    response = _priors_response([])  # 0 priors, below min_priors=1
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
    fixture = next(f for f in fixtures if f.id == "caution_max_cop_limits")
    response = _priors_response(
        ["caution"] * 3, dois=["10.1/a", "", "10.2/b"]
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
