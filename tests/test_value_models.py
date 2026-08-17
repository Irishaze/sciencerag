"""Unit tests for the per-kind `Prior.value` discriminated union (Phase A1).

Each of the 5 value models gets a positive case (valid shape) and a negative
case per validator (the specific rule that model enforces). Also checks that
`Prior` dispatches `value` to the right model based on sibling `kind`, and
that the schema exports cleanly.
"""

import pytest
from pydantic import ValidationError

from sciencerag.priors.models import (
    CandidateConfigValue,
    CautionValue,
    MaterialPropertyValue,
    ParameterRangeValue,
    Prior,
    ScalingRelationshipValue,
)

SOURCES = [{"type": "paper", "doi": "10.1234/example"}]


# -- ParameterRangeValue ---------------------------------------------------


def test_parameter_range_value_valid_with_typical():
    v = ParameterRangeValue(field_name="leg_length", typical=0.06, unit="mm")
    assert v.typical == 0.06


def test_parameter_range_value_valid_with_min_max_and_conditions():
    v = ParameterRangeValue(
        field_name="leg_length",
        min=0.02,
        max=0.2,
        unit="mm",
        conditions={"temperature_k": 300},
    )
    assert v.min == 0.02 and v.max == 0.2


def test_parameter_range_value_requires_at_least_one_number():
    with pytest.raises(ValidationError, match="at least one of min/max/typical"):
        ParameterRangeValue(field_name="leg_length", unit="mm")


def test_parameter_range_value_requires_unit():
    with pytest.raises(ValidationError):
        ParameterRangeValue(field_name="leg_length", typical=0.06)


@pytest.mark.parametrize("field", ["min", "max", "typical"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_parameter_range_value_rejects_non_finite_bounds(field: str, bad: float) -> None:
    """Adversarial test: min/max/typical are plain `float | None`, so
    Pydantic accepts NaN/Infinity by default — unlike ValidateRequest's
    design_parameters/scalar_results/latent_state, which already reject
    non-finite values via reject_non_finite_values/_list. Confirmed live
    that this specific gap let a client-supplied parameter_range prior
    (priors are passed in-band to /sciencerag/validate, not resolved from a
    trusted store) with min=-inf/max=inf force evaluation.py's
    `actual < min` / `actual > max` checks to always be False — any finite
    `actual` is trivially not less than -inf or greater than +inf — turning
    a wildly out-of-range design_parameters value into a forged
    verdict="within_range"/"consistent" at HTTP 200."""
    with pytest.raises(ValidationError, match="non-finite value is not allowed"):
        ParameterRangeValue(field_name="leg_length", unit="mm", **{field: bad})


# -- MaterialPropertyValue -------------------------------------------------


def test_material_property_value_valid():
    v = MaterialPropertyValue(
        material="Bi2Te3",
        form="bulk",
        property_name="seebeck_coefficient",
        magnitude=200.0,
        unit="uV/K",
        method="measured",
    )
    assert v.magnitude == 200.0


def test_material_property_value_valid_without_magnitude():
    v = MaterialPropertyValue(material="Bi2Te3", property_name="seebeck_coefficient")
    assert v.magnitude is None and v.method == "unknown"


def test_material_property_value_magnitude_requires_unit():
    with pytest.raises(ValidationError, match="magnitude requires unit"):
        MaterialPropertyValue(
            material="Bi2Te3", property_name="seebeck_coefficient", magnitude=200.0
        )


# -- ScalingRelationshipValue -----------------------------------------------


def test_scaling_relationship_value_valid():
    v = ScalingRelationshipValue(x="leg_length", y="cop", direction="positive")
    assert v.direction == "positive"


def test_scaling_relationship_value_rejects_bad_direction():
    with pytest.raises(ValidationError):
        ScalingRelationshipValue(x="leg_length", y="cop", direction="up")


def test_scaling_relationship_value_requires_direction():
    with pytest.raises(ValidationError):
        ScalingRelationshipValue(x="leg_length", y="cop")


# -- CandidateConfigValue ---------------------------------------------------


def test_candidate_config_value_valid():
    v = CandidateConfigValue(
        parameters={"leg_length": 0.07, "leg_width": 0.12},
        reported_performance={"cop": 1.5},
    )
    assert len(v.parameters) == 2


def test_candidate_config_value_requires_at_least_two_parameters():
    with pytest.raises(ValidationError, match=">= 2 parameters"):
        CandidateConfigValue(parameters={"leg_length": 0.07})


# -- CautionValue ------------------------------------------------------------


def test_caution_value_valid():
    v = CautionValue(statement="Only valid below 400K", applicability_scope="T < 400K")
    assert v.statement


def test_caution_value_requires_statement():
    with pytest.raises(ValidationError):
        CautionValue()


# -- Prior dispatch: value shape follows sibling `kind` ---------------------


def test_prior_dispatches_value_to_parameter_range_model():
    p = Prior(
        prior_id="pr_1",
        kind="parameter_range",
        field="leg_length",
        value={"field_name": "leg_length", "typical": 0.06, "unit": "mm"},
        confidence=0.8,
        sources=SOURCES,
    )
    assert isinstance(p.value, ParameterRangeValue)


def test_prior_dispatches_value_to_scaling_relationship_model():
    p = Prior(
        prior_id="pr_2",
        kind="scaling_relationship",
        related_fields=["leg_length", "leg_width"],
        value={"x": "leg_length", "y": "leg_width", "direction": "positive"},
        confidence=0.7,
        sources=SOURCES,
    )
    assert isinstance(p.value, ScalingRelationshipValue)


def test_prior_rejects_value_that_fails_its_kind_schema():
    """A parameter_range prior whose value has no numeric field must fail at
    Prior construction, not just be silently accepted as a loose dict."""
    with pytest.raises(ValidationError):
        Prior(
            prior_id="pr_bad",
            kind="parameter_range",
            field="leg_length",
            value={"summary": "affects COP"},
            confidence=0.5,
            sources=SOURCES,
        )


def test_prior_rejects_value_shaped_for_a_different_kind():
    """A scaling_relationship-shaped value under kind=parameter_range must
    fail, not silently pass through as some other kind's schema."""
    with pytest.raises(ValidationError):
        Prior(
            prior_id="pr_bad2",
            kind="parameter_range",
            field="leg_length",
            value={"x": "leg_length", "y": "cop", "direction": "positive"},
            confidence=0.5,
            sources=SOURCES,
        )


# -- Prior/value cross-check: field names inside value must match field/related_fields --


def test_prior_rejects_parameter_range_field_name_mismatch():
    with pytest.raises(ValidationError, match="must match prior.field"):
        Prior(
            prior_id="pr_mismatch",
            kind="parameter_range",
            field="leg_length",
            value={"field_name": "leg_width", "typical": 0.06, "unit": "mm"},
            confidence=0.8,
            sources=SOURCES,
        )


def test_prior_rejects_scaling_relationship_related_fields_mismatch():
    with pytest.raises(ValidationError, match="must match prior.related_fields"):
        Prior(
            prior_id="pr_mismatch2",
            kind="scaling_relationship",
            related_fields=["leg_length", "pitch"],  # value says leg_width, not pitch
            value={"x": "leg_length", "y": "leg_width", "direction": "positive"},
            confidence=0.7,
            sources=SOURCES,
        )


def test_prior_rejects_candidate_config_related_fields_mismatch():
    with pytest.raises(ValidationError, match="must match prior.related_fields"):
        Prior(
            prior_id="pr_mismatch3",
            kind="candidate_config",
            related_fields=["leg_length", "leg_width"],
            value={"parameters": {"leg_length": 0.07, "pitch": 0.05}},  # pitch, not leg_width
            confidence=0.7,
            sources=SOURCES,
        )


def test_prior_accepts_candidate_config_when_related_fields_match():
    p = Prior(
        prior_id="pr_ok",
        kind="candidate_config",
        related_fields=["leg_length", "leg_width"],
        value={"parameters": {"leg_length": 0.07, "leg_width": 0.12}},
        confidence=0.7,
        sources=SOURCES,
    )
    assert isinstance(p.value, CandidateConfigValue)


def test_prior_value_schema_exports():
    schema = Prior.model_json_schema()
    assert "$defs" in schema
    for name in [
        "ParameterRangeValue",
        "MaterialPropertyValue",
        "ScalingRelationshipValue",
        "CandidateConfigValue",
        "CautionValue",
    ]:
        assert name in schema["$defs"], f"{name} missing from Prior schema $defs"
