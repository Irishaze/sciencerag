"""Sanity checks for the sim_params.json contract loader (sciencerag/priors/contract.py)."""

from sciencerag.priors.contract import CONTRACT, GEOMETRY_FREE_NAMES, GEOMETRY_FREE_PARAMS


def test_contract_has_12_geometry_free_params():
    assert len(GEOMETRY_FREE_PARAMS) == 12
    assert len(GEOMETRY_FREE_NAMES) == 12


def test_geometry_free_params_have_name_and_unit():
    for param in GEOMETRY_FREE_PARAMS:
        assert param["name"]
        assert param["unit"]


def test_material_and_operating_condition_are_not_prior_targets():
    assert CONTRACT["categories"]["material"]["prior_target"] is False
    assert CONTRACT["categories"]["operating_condition"]["prior_target"] is False
    assert CONTRACT["categories"]["geometry_free"]["prior_target"] is True
