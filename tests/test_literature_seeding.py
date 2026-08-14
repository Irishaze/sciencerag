"""Unit tests for sciencerag/validate/literature_seeding.py — converting
literature Priors into knowledge-graph candidates (the spec's "cold-start
seeding" mechanism). Pure conversion logic, no retrieval/LLM calls."""

from sciencerag.priors.kg import compute_entity_id
from sciencerag.priors.models import Prior, SourcePaper
from sciencerag.validate.literature_seeding import prior_to_kg_candidates


def _prior(kind: str, value: dict, **overrides) -> Prior:
    # Prior._value_field_names_match_prior_fields cross-checks field/
    # related_fields against the value's own parameter name(s) — derive the
    # right ones by kind so callers only need to supply `value`.
    field = None
    related_fields: list[str] = []
    if kind == "parameter_range":
        field = value["field_name"]
    elif kind == "scaling_relationship":
        related_fields = [value["x"], value["y"]]
    elif kind == "candidate_config":
        related_fields = list(value["parameters"])

    defaults = dict(
        prior_id="pr_test",
        kind=kind,
        field=field,
        related_fields=related_fields,
        value=value,
        confidence=0.6,
        sources=[SourcePaper(doi="10.1/fake", span="pages 1-2")],
        notes=None,
    )
    defaults.update(overrides)
    return Prior.model_validate(defaults)


def test_parameter_range_becomes_numeric_candidate():
    prior = _prior(
        "parameter_range",
        {"field_name": "leg_length", "typical": 0.8, "unit": "mm"},
    )
    candidates = prior_to_kg_candidates(prior)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.relation == "literature_range_leg_length"
    assert c.object_value == 0.8
    assert c.object_unit == "mm"
    assert c.object_entity_id is None
    assert c.entity_type == "TECDesign"
    assert c.supporting_evidence["source_doi"] == "10.1/fake"


def test_parameter_range_falls_back_to_min_max_midpoint():
    prior = _prior(
        "parameter_range",
        {"field_name": "pitch", "min": 0.2, "max": 0.6, "unit": "mm"},
    )
    candidates = prior_to_kg_candidates(prior)
    assert candidates[0].object_value == 0.4


def test_parameter_range_with_no_numbers_is_skipped():
    # Shouldn't happen per ParameterRangeValue's own validator (requires at
    # least one of min/max/typical), but the converter must not crash if it
    # somehow does.
    prior = _prior("parameter_range", {"field_name": "pitch", "typical": 0.4, "unit": "mm"})
    prior.value.typical = None  # bypass the pydantic validator post-construction
    assert prior_to_kg_candidates(prior) == []


def test_material_property_becomes_numeric_candidate_on_material_subject():
    prior = _prior(
        "material_property",
        {"material": "Bi2Te3", "property_name": "seebeck_coefficient", "magnitude": 210.0, "unit": "uV/K"},
    )
    candidates = prior_to_kg_candidates(prior)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.subject == "Bi2Te3"
    assert c.relation == "literature_seebeck_coefficient"
    assert c.object_value == 210.0
    assert c.entity_type == "Material"


def test_material_property_without_magnitude_is_skipped():
    prior = _prior(
        "material_property",
        {"material": "Bi2Te3", "property_name": "unknown_thing", "magnitude": None},
    )
    assert prior_to_kg_candidates(prior) == []


def test_scaling_relationship_becomes_link_candidate_between_parameters():
    prior = _prior(
        "scaling_relationship",
        {"x": "leg_length", "y": "total_resistance_ohm", "direction": "positive"},
        field=None,
        related_fields=["leg_length", "total_resistance_ohm"],
    )
    candidates = prior_to_kg_candidates(prior)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.subject == "leg_length"
    assert c.relation == "POSITIVELY_CORRELATES_WITH"
    assert c.object_entity_id == compute_entity_id("total_resistance_ohm", {})
    assert c.object_entity_label == "total_resistance_ohm"
    assert c.object_value is None
    assert c.relation_description == "正相关"
    assert c.entity_type == "GenericParameter"


def test_negative_direction_maps_to_negative_relation():
    prior = _prior(
        "scaling_relationship",
        {"x": "a", "y": "b", "direction": "negative"},
        related_fields=["a", "b"],
    )
    assert prior_to_kg_candidates(prior)[0].relation == "NEGATIVELY_CORRELATES_WITH"


def test_candidate_config_and_caution_are_skipped_not_forced():
    config_prior = _prior(
        "candidate_config",
        {"parameters": {"leg_length": 0.8, "pitch": 0.3}, "reported_performance": {}},
    )
    caution_prior = _prior("caution", {"statement": "results may not generalize below 200K"})
    assert prior_to_kg_candidates(config_prior) == []
    assert prior_to_kg_candidates(caution_prior) == []


def test_no_doi_falls_back_to_no_source_doi_in_evidence():
    prior = _prior(
        "parameter_range",
        {"field_name": "leg_length", "typical": 0.8, "unit": "mm"},
        sources=[SourcePaper(doi="", span="pages 1-2")],
    )
    candidates = prior_to_kg_candidates(prior)
    assert "source_doi" not in candidates[0].supporting_evidence
