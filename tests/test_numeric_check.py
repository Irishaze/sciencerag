"""Tests for sciencerag/priors/numeric_check.py (Phase B1/B2: extraction + matching).

extract_numbers_from_text's format-parsing cases are the ones Step B1 itself
calls out (comma, scientific notation, percent, range, plus-minus).
numbers_match/find_unmatched_numbers are Step B2's matching rule. Pipeline
wiring (Prior-level accept/reject, trace, retries) is Phase B3, not tested
here.
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
from sciencerag.priors.numeric_check import (
    extract_numbers,
    extract_numbers_from_text,
    find_unmatched_numbers,
    numbers_match,
)

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


def test_extract_numbers_from_text_ignores_chemical_formula_subscripts():
    """Found via real-query validation: "Bi2Te3"/"Sb2Te3" is in nearly every
    evidence snippet in this corpus, and without the letter-adjacency guard
    its embedded "2"/"3" were being read as free-floating numbers — quietly
    widening the evidence-number pool and letting an unrelated prior number
    "2" or "3" pass the groundedness check for the wrong reason."""
    assert extract_numbers_from_text("Bi2Te3 thermoelectric coolers show good performance.") == []


def test_extract_numbers_from_text_still_matches_number_abutting_unit():
    """The chemical-formula guard must not block the common no-space
    number+unit style ("60um", not "60 um")."""
    assert extract_numbers_from_text("COP peaks at 60um leg length.") == [60.0]


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


def test_extract_numbers_caution_statement_is_scanned():
    """Regression test: found via a real production sample where a caution
    prior's `statement` asserted "~90 K" with no such number anywhere in
    the cited evidence, and it passed ungrounded because this kind's prose
    fields weren't scanned at all — caution priors can fabricate numbers
    same as any other kind, prose or not."""
    prior = Prior(
        prior_id="pr_7",
        kind="caution",
        field="leg_length",
        value=CautionValue(statement="Only valid below 400K"),
        confidence=0.6,
        sources=SOURCES,
    )
    assert extract_numbers(prior) == [400.0]


def test_extract_numbers_caution_applicability_scope_is_scanned():
    prior = Prior(
        prior_id="pr_7b",
        kind="caution",
        field="leg_length",
        value=CautionValue(
            statement="Reduces cooling capacity under high current.",
            applicability_scope="current above 2.5 A",
        ),
        confidence=0.6,
        sources=SOURCES,
    )
    assert extract_numbers(prior) == [2.5]


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


# -- numbers_match: Step B2's matching rule -----------------------------------


def test_numbers_match_exact_equality():
    assert numbers_match(1.8, 1.8) is True


def test_numbers_match_zero_exact():
    assert numbers_match(0, 0) is True


def test_numbers_match_within_rounding_tolerance():
    # The spec's own worked example: evidence states 1.83, prior rounds to
    # 1.8 (~1.64% relative) — must match under the widened 2% tolerance.
    assert numbers_match(1.8, 1.83) is True


def test_numbers_match_negative_numbers_within_tolerance():
    assert numbers_match(-40, -40.2) is True


def test_numbers_match_rejects_real_drift():
    assert numbers_match(1.8, 2.0) is False


def test_numbers_match_does_not_unit_convert():
    # 20 (e.g. µm) vs 0.02 (e.g. mm) is the same physical length but a
    # different number — v1 deliberately does not know this, by design.
    assert numbers_match(20, 0.02) is False


def test_numbers_match_zero_evidence_number_only_matches_zero():
    assert numbers_match(5, 0) is False
    assert numbers_match(0, 0) is True


# -- find_unmatched_numbers ----------------------------------------------------


def test_find_unmatched_numbers_all_grounded():
    assert find_unmatched_numbers([0.06, 300.0], [0.06, 300.0, 5.0]) == []


def test_find_unmatched_numbers_reports_ungrounded():
    assert find_unmatched_numbers([0.06, 999.0], [0.06, 300.0]) == [999.0]


def test_find_unmatched_numbers_empty_prior_numbers():
    assert find_unmatched_numbers([], [1.0, 2.0, 3.0]) == []


def test_find_unmatched_numbers_uses_tolerance():
    assert find_unmatched_numbers([1.8], [1.83]) == []
