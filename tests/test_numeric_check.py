"""Tests for sciencerag/priors/numeric_check.py (Phase B1: number extraction).

extract_numbers_from_text's format-parsing cases are the ones Step B1 itself
calls out (comma, scientific notation, percent, range, plus-minus). Matching
logic and pipeline wiring are Phase B2/B3, not tested here.
"""

import pytest

from sciencerag.priors.models import (
    CandidateConfigValue,
    CautionValue,
    MaterialPropertyValue,
    Prior,
    ScalingRelationshipValue,
    SourcePaper,
)
from sciencerag.priors.numeric_check import extract_numbers, extract_numbers_from_text

SOURCES = [SourcePaper(doi="10.1234/example")]


# -- extract_numbers_from_text: format-parsing cases -------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("The value is 42.5 mm.", [42.5]),
        ("no numbers here", []),
        ("", []),
        ("A leg length of -40 mm is unphysical here.", [-40.0]),
    ],
)
def test_extract_numbers_from_text_plain(text, expected):
    assert extract_numbers_from_text(text) == expected


def test_extract_numbers_from_text_thousands_comma():
    assert extract_numbers_from_text("Tested over 1,200 cycles.") == [1200.0]


def test_extract_numbers_from_text_thousands_comma_with_decimal():
    assert extract_numbers_from_text("Cost was 1,200.50 dollars.") == [1200.5]


def test_extract_numbers_from_text_scientific_notation():
    assert extract_numbers_from_text("Leakage current of 1.5e-3 A.") == [0.0015]


def test_extract_numbers_from_text_negative_scientific_notation():
    assert extract_numbers_from_text("Coefficient of -2.5E+3.") == [-2500.0]


def test_extract_numbers_from_text_percent_records_both_forms():
    assert extract_numbers_from_text("Efficiency improved by 15%.") == [15.0, 0.15]


@pytest.mark.parametrize(
    "text",
    [
        "Leg length ranges from 20-200 um.",
        "Leg length ranges from 20–200 um.",  # en dash
        "Leg length ranges from 20 to 200 um.",
    ],
)
def test_extract_numbers_from_text_range_variants(text):
    assert extract_numbers_from_text(text) == [20.0, 200.0]


def test_extract_numbers_from_text_plus_minus_unicode():
    assert extract_numbers_from_text("Measured at 60±5 degrees.") == [60.0, 5.0]


def test_extract_numbers_from_text_plus_minus_ascii():
    assert extract_numbers_from_text("Measured at 60+/-5 degrees.") == [60.0, 5.0]


def test_extract_numbers_from_text_mixed_range_and_plain():
    assert extract_numbers_from_text("COP of 1.5 to 1.8 at 300K.") == [1.5, 1.8, 300.0]


# -- extract_numbers(prior): per-kind field coverage -------------------------


def test_extract_numbers_parameter_range_min_max_typical_and_conditions():
    prior = Prior(
        prior_id="pr_1",
        kind="parameter_range",
        field="leg_length",
        value={
            "field_name": "leg_length",
            "min": 0.02,
            "max": 0.2,
            "typical": 0.06,
            "unit": "mm",
            "conditions": {"temperature_k": 300, "leg_shape": "square"},
        },
        confidence=0.8,
        sources=SOURCES,
    )
    assert sorted(extract_numbers(prior)) == [0.02, 0.06, 0.2, 300.0]


def test_extract_numbers_parameter_range_typical_only():
    prior = Prior(
        prior_id="pr_2",
        kind="parameter_range",
        field="leg_length",
        value={"field_name": "leg_length", "typical": 0.06, "unit": "mm"},
        confidence=0.8,
        sources=SOURCES,
    )
    assert extract_numbers(prior) == [0.06]


def test_extract_numbers_material_property_with_magnitude_and_conditions():
    prior = Prior(
        prior_id="pr_3",
        kind="material_property",
        value=MaterialPropertyValue(
            material="Bi2Te3",
            property_name="seebeck_coefficient",
            magnitude=200.0,
            unit="uV/K",
            conditions={"temperature_k": 300},
        ),
        confidence=0.7,
        sources=SOURCES,
    )
    assert sorted(extract_numbers(prior)) == [200.0, 300.0]


def test_extract_numbers_material_property_without_magnitude():
    prior = Prior(
        prior_id="pr_4",
        kind="material_property",
        value=MaterialPropertyValue(material="Bi2Te3", property_name="seebeck_coefficient"),
        confidence=0.7,
        sources=SOURCES,
    )
    assert extract_numbers(prior) == []


def test_extract_numbers_scaling_relationship_has_no_numbers():
    prior = Prior(
        prior_id="pr_5",
        kind="scaling_relationship",
        related_fields=["leg_length", "cop"],
        value=ScalingRelationshipValue(x="leg_length", y="cop", direction="convex"),
        confidence=0.6,
        sources=SOURCES,
    )
    assert extract_numbers(prior) == []


def test_extract_numbers_candidate_config_parameters_and_performance():
    prior = Prior(
        prior_id="pr_6",
        kind="candidate_config",
        related_fields=["leg_length", "leg_width"],
        value=CandidateConfigValue(
            parameters={"leg_length": 0.07, "leg_width": 0.12},
            reported_performance={"cop": 1.5, "note": "measured"},
        ),
        confidence=0.7,
        sources=SOURCES,
    )
    assert sorted(extract_numbers(prior)) == [0.07, 0.12, 1.5]


def test_extract_numbers_caution_has_no_numbers():
    prior = Prior(
        prior_id="pr_7",
        kind="caution",
        field="leg_length",
        value=CautionValue(statement="Only valid below 400K"),
        confidence=0.6,
        sources=SOURCES,
    )
    # Caution's own value fields are prose (not scanned) — this is a known
    # v1 boundary, not a bug: only `notes` gets free-text number extraction.
    assert extract_numbers(prior) == []


def test_extract_numbers_includes_notes_text():
    prior = Prior(
        prior_id="pr_8",
        kind="parameter_range",
        field="leg_length",
        value={"field_name": "leg_length", "typical": 0.06, "unit": "mm"},
        confidence=0.8,
        sources=SOURCES,
        notes="Measured over 5,000 cycles.",
    )
    assert sorted(extract_numbers(prior)) == [0.06, 5000.0]
