"""Tests for the deterministic kind/field classifier (sciencerag/priors/classify.py)."""

import pytest

from sciencerag.priors.classify import DEFAULT_FIELD, DEFAULT_KIND, classify


@pytest.mark.parametrize(
    ("text", "expected_kind"),
    [
        (
            "The Seebeck coefficient and electrical resistivity are key material properties.",
            "material_property",
        ),
        (
            "The COP is a strongly convex function of the driving voltage.",
            "scaling_relationship",
        ),
        (
            "However, below a certain threshold the COP drops to zero, a limitation of this design.",
            "caution",
        ),
        (
            "A pulse-frequency modulation driving method with a 50% duty cycle improves performance.",
            "candidate_config",
        ),
        (
            "An optimal voltage of 2.3 V achieves maximum COP at these operating temperatures.",
            "parameter_range",
        ),
    ],
)
def test_classify_matches_expected_kind(text, expected_kind):
    kind, _field = classify(text)
    assert kind == expected_kind


def test_classify_falls_back_to_default_when_no_rule_matches():
    kind, field = classify("This paper discusses general cooling trends in industry.")
    assert kind == DEFAULT_KIND
    assert field == DEFAULT_FIELD


def test_classify_is_case_insensitive():
    kind, _field = classify("THE SEEBECK COEFFICIENT IS IMPORTANT.")
    assert kind == "material_property"
