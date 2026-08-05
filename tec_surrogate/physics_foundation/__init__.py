"""Reusable component-graph neural operators for TEC field prediction."""

from .graph import BoundarySamples, ComponentGraph, InterfaceSamples, batch_component_graphs
from .field_service import MultiPairFieldService
from .losses import (
    LossScales,
    boundary_value_loss,
    interface_conservation_loss,
    steady_thermoelectric_residual_loss,
    supervised_field_loss,
)
from .model import ComponentGraphSepONet
from .multi_pair import (
    MAX_SUPPORTED_PAIRS,
    MULTI_COMPONENT_TYPES,
    MultiPairLayout,
    compose_multi_pair_graph,
)
from .real_data import load_component_case, load_component_interfaces

__all__ = [
    "BoundarySamples",
    "ComponentGraph",
    "ComponentGraphSepONet",
    "InterfaceSamples",
    "LossScales",
    "MAX_SUPPORTED_PAIRS",
    "MULTI_COMPONENT_TYPES",
    "MultiPairLayout",
    "MultiPairFieldService",
    "boundary_value_loss",
    "batch_component_graphs",
    "compose_multi_pair_graph",
    "interface_conservation_loss",
    "load_component_case",
    "load_component_interfaces",
    "steady_thermoelectric_residual_loss",
    "supervised_field_loss",
]
