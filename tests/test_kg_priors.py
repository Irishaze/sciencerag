"""Tests for wiring sciencerag.priors' knowledge-graph query into its
response (spec §3.2: KG queried first, always additive to literature
retrieval — see sciencerag/priors/retrieval.py's _kg_priors_for_query).

Every test isolates kg.GRAPH_PATH to a temp file via add_triple() (the
real write path) rather than touching the project's real,
growing data/kg/graph.json.
"""

import pytest

from sciencerag.priors import kg, retrieval
from sciencerag.priors.contract import GEOMETRY_FREE_NAMES
from sciencerag.priors.kg import KGSource, add_triple


@pytest.fixture(autouse=True)
def _isolated_graph(tmp_path, monkeypatch):
    monkeypatch.setattr(kg, "GRAPH_PATH", tmp_path / "graph.json")


def _seed_design(*, subject="Bi2Te3 single-stage TEC", conditions, achieves, run_id="run_x"):
    """Writes one achieves_<field> triple per (field, value) pair in
    `achieves`, all sharing the same conditions -> same entity_id, the way
    a real sciencerag.validate run's kg_candidates.py path does."""
    for field, value in achieves.items():
        add_triple(
            subject=subject,
            relation=f"achieves_{field}",
            object_value=value,
            object_unit=None,
            conditions=conditions,
            confidence=0.7,
            run_id=run_id,
            sources=[KGSource(type="run", run_id=run_id)],
            entity_type="TECDesign",
        )


def _seed_literature_range(field, typical, unit, confidence=0.5, doi="10.1/kg-test"):
    add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation=f"literature_range_{field}",
        object_value=typical,
        object_unit=unit,
        conditions={},
        confidence=confidence,
        run_id="seed",
        sources=[KGSource(type="paper", doi=doi)],
        entity_type="TECDesign",
    )


def test_simulation_result_triples_become_one_candidate_config_prior():
    _seed_design(
        conditions={"leg_length": 0.8, "pitch": 0.29, "length": 4.9, "n_pairs": 1.0},
        achieves={"delta_T_max_K": 71.7, "optimal_current_A": 6.5},
    )
    priors = retrieval._kg_priors_for_query("leg_length delta_T_max_K")
    assert len(priors) == 1
    prior = priors[0]
    assert prior.kind == "candidate_config"
    assert prior.provenance == "internal"
    # n_pairs is not a contract geometry_free name — must be dropped.
    assert "n_pairs" not in prior.value.parameters
    assert set(prior.value.parameters) == {"leg_length", "pitch", "length"}
    assert set(prior.related_fields) == {"leg_length", "pitch", "length"}
    assert prior.value.reported_performance == {"delta_T_max_K": 71.7, "optimal_current_A": 6.5}
    assert len(prior.sources) == 2  # one per achieves_* triple
    assert all(s.type == "kg_triple" for s in prior.sources)


def test_design_with_fewer_than_two_contract_params_is_skipped():
    """CandidateConfigValue requires >= 2 parameters — a design whose
    conditions only carry 1 real contract name (rest filtered out, e.g.
    only n_pairs plus one geometry field) can't form a valid Prior."""
    _seed_design(
        conditions={"leg_length": 0.8, "n_pairs": 1.0},
        achieves={"delta_T_max_K": 71.7},
    )
    priors = retrieval._kg_priors_for_query("leg_length")
    assert priors == []


def test_literature_seeded_triple_becomes_parameter_range_prior():
    _seed_literature_range("leg_length", typical=2.75, unit="mm")
    priors = retrieval._kg_priors_for_query("leg_length range")
    assert len(priors) == 1
    prior = priors[0]
    assert prior.kind == "parameter_range"
    assert prior.field == "leg_length"
    assert prior.value.typical == 2.75
    assert prior.value.unit == "mm"
    assert prior.sources[0].type == "kg_triple"


def test_literature_seeded_triple_without_unit_is_skipped():
    """No guessing a unit — matches the rest of the pipeline's "don't
    guess, drop it" principle for missing units."""
    _seed_literature_range("leg_length", typical=2.75, unit=None)
    assert retrieval._kg_priors_for_query("leg_length") == []


def test_non_contract_field_literature_triple_is_skipped():
    add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="literature_range_not_a_real_contract_field",
        object_value=1.0,
        object_unit="mm",
        conditions={},
        confidence=0.5,
        run_id="seed",
        sources=[KGSource(type="paper", doi="10.1/x")],
    )
    assert retrieval._kg_priors_for_query("not_a_real_contract_field") == []


def test_structural_link_only_entity_produces_no_prior():
    """SIMULATION_USES_DESIGN/MATERIAL triples (conditions={}, no
    achieves_*/literature_range_* relation) carry no performance or range
    data — nothing prior-worthy to extract."""
    add_triple(
        subject="run_x",
        relation="SIMULATION_USES_MATERIAL",
        object_entity_id=kg.compute_entity_id("Bi2Te3", {}),
        object_entity_label="Bi2Te3",
        object_entity_type="Material",
        conditions={},
        confidence=1.0,
        run_id="run_x",
        sources=[KGSource(type="run", run_id="run_x")],
    )
    assert retrieval._kg_priors_for_query("Bi2Te3 material") == []


def test_kg_prior_closes_the_geometry_gap_for_its_parameters(monkeypatch):
    """Integration: a KG-covered parameter must not show up in
    coverage.gaps, and must not trigger unnecessary external augmentation
    (which only fires on non-empty gaps) for that field — even when
    literature retrieval itself finds nothing for the query."""
    _seed_design(
        conditions={
            "leg_length": 0.8,
            "leg_width": 1.1,
            "pitch": 0.29,
            "d_conductor": 125.0,
            "d_ceramics": 0.37,
            "length": 4.9,
            "width": 1.75,
            "height": 1.57,
            "n_pairs": 1.0,
        },
        achieves={"delta_T_max_K": 71.7},
    )

    class _FakeSession:
        contexts = []

    class _FakeResponse:
        session = _FakeSession()

    monkeypatch.setattr(retrieval, "run_query", lambda query: _FakeResponse())
    monkeypatch.setattr(retrieval, "search_semantic_scholar", lambda query: [])
    monkeypatch.setattr(retrieval, "search_arxiv", lambda query: [])

    response, _filtered_material_count = retrieval.build_priors_response(
        "leg_length leg_width pitch d_conductor d_ceramics length width height"
    )

    assert len(response.priors) == 1
    assert response.priors[0].kind == "candidate_config"
    for field in ("leg_length", "leg_width", "pitch", "d_conductor", "d_ceramics", "length", "width", "height"):
        assert not any(field in gap for gap in response.coverage.gaps), (
            f"{field} should be covered by the KG prior, found in gaps: {response.coverage.gaps}"
        )
