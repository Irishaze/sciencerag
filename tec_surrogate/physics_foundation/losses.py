"""Data, boundary, interface, and conservative thermoelectric losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .graph import BoundarySamples, ComponentGraph, InterfaceSamples, Tensor
from .model import ComponentGraphSepONet


@dataclass(frozen=True)
class LossScales:
    """Reference magnitudes used to nondimensionalize heterogeneous losses."""

    temperature: float = 50.0
    potential: float = 1.0
    heat_flux: float = 1.0
    current_density: float = 1.0
    charge_residual: float = 1.0
    energy_residual: float = 1.0


def supervised_field_loss(
    prediction: Tensor,
    target: Tensor,
    scales: LossScales,
    mask: Tensor | None = None,
) -> Tensor:
    scale = prediction.new_tensor((scales.temperature, scales.potential))
    difference = (prediction - target) / scale
    if mask is None:
        return difference.square().mean()
    selected = difference[mask.bool()]
    return selected.square().mean() if selected.numel() else prediction.sum() * 0


def boundary_value_loss(
    model: ComponentGraphSepONet,
    graph: ComponentGraph,
    node_state: Tensor,
    samples: BoundarySamples,
    scales: LossScales,
) -> Tensor:
    samples.validate()
    prediction = model.decode(graph, node_state, samples.components, samples.coords)
    scale = prediction.new_tensor((scales.temperature, scales.potential))
    difference = (prediction - samples.targets) / scale
    selected = difference[samples.mask.bool()]
    return selected.square().mean() if selected.numel() else prediction.sum() * 0


def _field_gradients(fields: Tensor, coords: Tensor, sizes: Tensor) -> tuple[Tensor, Tensor]:
    gradients = []
    for field_index in range(2):
        local_gradient = torch.autograd.grad(
            fields[:, field_index].sum(),
            coords,
            create_graph=True,
            retain_graph=True,
        )[0]
        gradients.append(local_gradient / sizes)
    return gradients[0], gradients[1]


def _constitutive_fields(
    model: ComponentGraphSepONet,
    graph: ComponentGraph,
    node_state: Tensor,
    component_ids: Tensor,
    coords: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    local_coords = coords.detach().clone().requires_grad_(True)
    fields = model.decode(graph, node_state, component_ids, local_coords)
    sizes = graph.node_sizes[component_ids]
    grad_temperature, grad_potential = _field_gradients(fields, local_coords, sizes)
    electric_field = -grad_potential
    sigma = graph.electrical_conductivity[component_ids, None]
    seebeck = graph.seebeck_coefficient[component_ids, None]
    conductivity = graph.thermal_conductivity[component_ids, None]
    current_density = sigma * (electric_field - seebeck * grad_temperature)
    heat_flux = -conductivity * grad_temperature + seebeck * fields[:, :1] * current_density
    return fields, electric_field, current_density, heat_flux, local_coords


def interface_conservation_loss(
    model: ComponentGraphSepONet,
    graph: ComponentGraph,
    node_state: Tensor,
    samples: InterfaceSamples,
    scales: LossScales,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Enforce jumps and normal flux conservation across material interfaces."""

    samples.validate()
    source = _constitutive_fields(
        model, graph, node_state, samples.source_components, samples.source_coords
    )
    target = _constitutive_fields(
        model, graph, node_state, samples.target_components, samples.target_coords
    )
    source_fields, _, source_current, source_heat, _ = source
    target_fields, _, target_current, target_heat, _ = target
    normal = samples.source_normals

    source_current_normal = (source_current * normal).sum(dim=-1)
    target_current_normal = (target_current * -normal).sum(dim=-1)
    source_heat_normal = (source_heat * normal).sum(dim=-1)
    target_heat_normal = (target_heat * -normal).sum(dim=-1)

    temperature_jump = (
        source_fields[:, 0]
        - target_fields[:, 0]
        - samples.thermal_resistance * source_heat_normal
    ) / scales.temperature
    potential_jump = (
        source_fields[:, 1]
        - target_fields[:, 1]
        - samples.electrical_resistance * source_current_normal
    ) / scales.potential
    heat_balance = (source_heat_normal + target_heat_normal) / scales.heat_flux
    current_balance = (source_current_normal + target_current_normal) / scales.current_density

    def masked_square_mean(values: Tensor, mask: Tensor | None) -> Tensor:
        selected = values if mask is None else values[mask.bool()]
        return selected.square().mean() if selected.numel() else values.sum() * 0

    terms = {
        "temperature_jump": masked_square_mean(
            temperature_jump, samples.thermal_mask
        ),
        "potential_jump": masked_square_mean(
            potential_jump, samples.electrical_mask
        ),
        "heat_balance": masked_square_mean(heat_balance, samples.thermal_mask),
        "current_balance": masked_square_mean(
            current_balance, samples.electrical_mask
        ),
    }
    return sum(terms.values()), terms


def _divergence(vector: Tensor, coords: Tensor, sizes: Tensor) -> Tensor:
    result = torch.zeros(len(coords), device=coords.device, dtype=coords.dtype)
    for axis in range(3):
        local_gradient = torch.autograd.grad(
            vector[:, axis].sum(),
            coords,
            create_graph=True,
            retain_graph=True,
        )[0]
        result = result + local_gradient[:, axis] / sizes[:, axis]
    return result


def steady_thermoelectric_residual_loss(
    model: ComponentGraphSepONet,
    graph: ComponentGraph,
    node_state: Tensor,
    component_ids: Tensor,
    local_coords: Tensor,
    scales: LossScales,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Steady conservative residual for constant properties inside each component.

    The convention is ``J = sigma * (E - S grad(T))``, ``q = -k grad(T) +
    S*T*J``, ``div(J)=0``, and ``div(q)-J dot E=0``. Temperature-dependent
    properties can be added later by replacing the material lookup in
    ``_constitutive_fields``.
    """

    fields, electric_field, current_density, heat_flux, coords = _constitutive_fields(
        model, graph, node_state, component_ids, local_coords
    )
    del fields
    sizes = graph.node_sizes[component_ids]
    charge = _divergence(current_density, coords, sizes) / scales.charge_residual
    joule_heating = (current_density * electric_field).sum(dim=-1)
    energy = (_divergence(heat_flux, coords, sizes) - joule_heating) / scales.energy_residual
    terms = {
        "charge": charge.square().mean(),
        "energy": energy.square().mean(),
    }
    return sum(terms.values()), terms
