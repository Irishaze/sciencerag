"""Export component-wise fields from a solved COMSOL TEC model.

The module intentionally imports no ``mph`` package at import time. Callers
pass an already loaded ``mph.Model`` so dataset parsing and unit tests remain
usable in the PyTorch-only training environment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from uuid import uuid4

import numpy as np


COMPONENT_TYPE_NAMES = ("ceramic", "conductor", "p_leg", "n_leg")
COMPONENT_SELECTIONS = (
    ("ceramic_bottom", "geom1_csel5_dom", "ceramic"),
    ("ceramic_top", "geom1_csel6_dom", "ceramic"),
    ("conductor", "geom1_csel3_dom", "conductor"),
    ("p_leg", "geom1_csel2_dom", "p_leg"),
    ("n_leg", "geom1_csel1_dom", "n_leg"),
)
DEFAULT_PROBE_GRIDS = {
    "ceramic": (4, 4, 3),
    "conductor": (4, 4, 3),
    "p_leg": (6, 6, 10),
    "n_leg": (6, 6, 10),
}

# Seebeck fallback values. Thermal and electrical conductivities are exported
# directly from the solved COMSOL fields for every solution and component.
DEFAULT_MATERIALS = {
    "ceramic": (175.0, 0.0, 0.0),
    "conductor": (400.0, 5.8e7, 0.0),
    "p_leg": (1.5, 1.0e5, 2.0e-4),
    "n_leg": (1.5, 1.0e5, -2.0e-4),
}


@dataclass(frozen=True)
class ComponentDefinition:
    name: str
    component_type: str
    domain: int
    bbox: tuple[float, float, float, float, float, float]

    @property
    def minimum(self) -> np.ndarray:
        return np.asarray(self.bbox[::2], dtype=float)

    @property
    def maximum(self) -> np.ndarray:
        return np.asarray(self.bbox[1::2], dtype=float)

    @property
    def size(self) -> np.ndarray:
        return self.maximum - self.minimum

    @property
    def center(self) -> np.ndarray:
        return (self.minimum + self.maximum) / 2


@dataclass(frozen=True)
class ContactDefinition:
    source: int
    target: int
    axis: int
    normal: tuple[float, float, float]
    area: float


def _temporary_tag(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:10]}"


def _normalise_result(data: object, point_count: int) -> np.ndarray:
    """Return COMSOL interpolation output as ``(solutions, points)``."""

    array = np.asarray(data)
    if array.ndim >= 1 and array.shape[0] == 1:
        array = array[0]
    array = np.squeeze(array)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        if array.size != point_count:
            raise ValueError(
                f"COMSOL returned {array.size} values for {point_count} points"
            )
        array = array.reshape(1, point_count)
    elif array.ndim == 2:
        if array.shape[-1] == point_count:
            pass
        elif array.shape[0] == point_count:
            array = array.T
        else:
            raise ValueError(
                f"Cannot interpret COMSOL result shape {array.shape} for {point_count} points"
            )
    else:
        raise ValueError(f"Unexpected COMSOL result shape: {array.shape}")
    return np.asarray(np.real_if_close(array), dtype=float)


def domain_bounding_boxes(
    model: object,
    domains: list[int],
    dataset_tag: str,
) -> dict[int, tuple[float, float, float, float, float, float]]:
    """Measure solved-domain bounds using an entity-restricted evaluation."""

    numerical = model.java.result().numerical()
    tag = _temporary_tag("tec_bbox")
    feature = numerical.create(tag, "Eval")
    try:
        feature.set("data", dataset_tag)
        feature.set("expr", ["x", "y", "z"])
        result = {}
        for domain in domains:
            feature.selection().set([int(domain)])
            values = np.asarray(feature.getData(), dtype=float)
            if values.ndim != 3 or values.shape[0] != 3:
                raise ValueError(f"Unexpected coordinate result shape {values.shape}")
            coordinates = values[:, 0, :]
            bounds = []
            for axis in range(3):
                bounds.extend(
                    (float(np.nanmin(coordinates[axis])), float(np.nanmax(coordinates[axis])))
                )
            result[domain] = tuple(bounds)
        return result
    finally:
        numerical.remove(tag)


def discover_components(
    model: object,
    geometry_dataset_tag: str = "dset2",
) -> list[ComponentDefinition]:
    """Resolve stable cumulative selections into individual physical domains."""

    component_root = model.java.component("comp1")
    records: list[tuple[str, str, int]] = []
    claimed: set[int] = set()
    for role, selection_tag, component_type in COMPONENT_SELECTIONS:
        try:
            domains = [
                int(value)
                for value in component_root.selection(selection_tag).entities()
            ]
        except Exception as exc:
            raise RuntimeError(f"Required COMSOL selection is missing: {selection_tag}") from exc
        for index, domain in enumerate(domains):
            if domain in claimed:
                raise ValueError(f"COMSOL domain {domain} belongs to multiple component selections")
            claimed.add(domain)
            name = role if len(domains) == 1 else f"{role}_{index + 1}"
            records.append((name, component_type, domain))

    bounds = domain_bounding_boxes(
        model, [record[2] for record in records], geometry_dataset_tag
    )
    components = [
        ComponentDefinition(name, component_type, domain, bounds[domain])
        for name, component_type, domain in records
    ]
    return sorted(components, key=lambda component: component.domain)


def local_probe_grid(shape: tuple[int, int, int]) -> np.ndarray:
    """Cell-centred local tensor grid that avoids ambiguous interface points."""

    axes = [(np.arange(count, dtype=float) + 0.5) / count for count in shape]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([axis.ravel() for axis in mesh])


def build_component_probes(
    components: list[ComponentDefinition],
    grids: dict[str, tuple[int, int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local coordinates, component indices, and physical coordinates."""

    grids = grids or DEFAULT_PROBE_GRIDS
    local_parts = []
    component_parts = []
    physical_parts = []
    for index, component in enumerate(components):
        local = local_probe_grid(grids[component.component_type])
        physical = component.minimum + local * component.size
        local_parts.append(local)
        component_parts.append(np.full(len(local), index, dtype=np.int64))
        physical_parts.append(physical)
    return (
        np.vstack(local_parts),
        np.concatenate(component_parts),
        np.vstack(physical_parts),
    )


def infer_contacts(
    components: list[ComponentDefinition],
    tolerance: float | None = None,
) -> list[ContactDefinition]:
    """Infer face contacts between axis-aligned component bounding boxes."""

    if not components:
        return []
    overall_size = np.ptp(np.vstack([component.minimum for component in components] + [component.maximum for component in components]), axis=0)
    tolerance = tolerance if tolerance is not None else max(float(overall_size.max()) * 1e-8, 1e-12)
    contacts = []
    for source in range(len(components)):
        for target in range(source + 1, len(components)):
            first, second = components[source], components[target]
            for axis in range(3):
                forward_gap = second.minimum[axis] - first.maximum[axis]
                reverse_gap = first.minimum[axis] - second.maximum[axis]
                if abs(forward_gap) <= tolerance:
                    normal_sign = 1.0
                elif abs(reverse_gap) <= tolerance:
                    normal_sign = -1.0
                else:
                    continue
                other_axes = [value for value in range(3) if value != axis]
                overlaps = [
                    min(first.maximum[value], second.maximum[value])
                    - max(first.minimum[value], second.minimum[value])
                    for value in other_axes
                ]
                if min(overlaps) <= tolerance:
                    continue
                normal = np.zeros(3)
                normal[axis] = normal_sign
                contacts.append(
                    ContactDefinition(
                        source=source,
                        target=target,
                        axis=axis,
                        normal=tuple(normal.tolist()),
                        area=float(np.prod(overlaps)),
                    )
                )
                break
    return contacts


def interpolate_expression(
    model: object,
    dataset_tag: str,
    expression: str,
    physical_coords: np.ndarray,
) -> np.ndarray:
    """Interpolate one scalar expression at physical points for all solutions."""

    if physical_coords.ndim != 2 or physical_coords.shape[1] != 3:
        raise ValueError("physical_coords must have shape (P, 3)")
    numerical = model.java.result().numerical()
    tag = _temporary_tag("tec_interp")
    feature = numerical.create(tag, "Interp")
    try:
        feature.set("data", dataset_tag)
        feature.set("expr", [expression])
        feature.set("coord", physical_coords.T.tolist())
        return _normalise_result(feature.getData(), len(physical_coords))
    finally:
        numerical.remove(tag)


def interpolate_first_available(
    model: object,
    dataset_tag: str,
    expressions: tuple[str, ...],
    physical_coords: np.ndarray,
) -> tuple[np.ndarray | None, str | None]:
    """Try COMSOL internal variable names without making export brittle."""

    for expression in expressions:
        try:
            return (
                interpolate_expression(
                    model, dataset_tag, expression, physical_coords
                ),
                expression,
            )
        except Exception:
            continue
    return None, None


def _solution_values(model: object, dataset_tag: str, solution_count: int) -> np.ndarray:
    dataset_name = next(
        dataset.name() for dataset in model / "datasets" if dataset.tag() == dataset_tag
    )
    try:
        _, values = model.inner(dataset_name)
        array = np.asarray(values, dtype=float).reshape(-1)
        if len(array) == solution_count:
            return array
    except Exception:
        pass
    return np.arange(solution_count, dtype=float)


def _parameter(model: object, name: str, default: float) -> float:
    try:
        return float(model.parameter(name, evaluate=True))
    except Exception:
        return default


def export_component_case(
    model: object,
    output_path: Path,
    dataset_tag: str = "dset3",
    geometry_dataset_tag: str = "dset2",
    grids: dict[str, tuple[int, int, int]] | None = None,
) -> dict[str, object]:
    """Export a solved COMSOL model to a training-ready component case file."""

    components = discover_components(model, geometry_dataset_tag)
    contacts = infer_contacts(components)
    local_coords, probe_components, physical_coords = build_component_probes(
        components, grids
    )
    temperature = interpolate_expression(model, dataset_tag, "T", physical_coords)

    conductive_components = np.asarray(
        [component.component_type != "ceramic" for component in components], dtype=bool
    )
    potential_probe_mask = conductive_components[probe_components]
    potential = np.zeros_like(temperature)
    if np.any(potential_probe_mask):
        conductive_potential = interpolate_expression(
            model, dataset_tag, "V", physical_coords[potential_probe_mask]
        )
        if len(conductive_potential) != len(temperature):
            raise ValueError("T and V interpolation returned different solution counts")
        potential[:, potential_probe_mask] = conductive_potential

    thermal_field = interpolate_expression(
        model, dataset_tag, "ht.kxx", physical_coords
    )
    electrical_field = np.zeros_like(temperature)
    if np.any(potential_probe_mask):
        electrical_field[:, potential_probe_mask] = interpolate_expression(
            model,
            dataset_tag,
            "ec.sigmaxx",
            physical_coords[potential_probe_mask],
        )

    bounds_min = np.min(np.vstack([component.minimum for component in components]), axis=0)
    bounds_max = np.max(np.vstack([component.maximum for component in components]), axis=0)
    overall_size = bounds_max - bounds_min
    centers = np.vstack([component.center for component in components])
    sizes = np.vstack([component.size for component in components])
    normalized_centers = (centers - bounds_min) / overall_size
    normalized_sizes = sizes / overall_size
    type_ids = np.asarray(
        [COMPONENT_TYPE_NAMES.index(component.component_type) for component in components],
        dtype=np.int64,
    )
    type_one_hot = np.eye(len(COMPONENT_TYPE_NAMES), dtype=float)[type_ids]
    node_features = np.column_stack((normalized_centers, normalized_sizes, type_one_hot))
    node_field_mask = np.column_stack(
        (np.ones(len(components), dtype=bool), conductive_components)
    )

    thermal_conductivity = np.empty((len(temperature), len(components)), dtype=float)
    electrical_conductivity = np.zeros_like(thermal_conductivity)
    for component_index in range(len(components)):
        probe_mask = probe_components == component_index
        thermal_conductivity[:, component_index] = np.nanmedian(
            thermal_field[:, probe_mask], axis=1
        )
        if conductive_components[component_index]:
            electrical_conductivity[:, component_index] = np.nanmedian(
                electrical_field[:, probe_mask], axis=1
            )
    seebeck_coefficient = np.tile(
        np.asarray(
            [DEFAULT_MATERIALS[component.component_type][2] for component in components],
            dtype=float,
        ),
        (len(temperature), 1),
    )
    leg_probe_mask = np.asarray(
        [
            components[index].component_type in {"p_leg", "n_leg"}
            for index in probe_components
        ],
        dtype=bool,
    )
    seebeck_field, seebeck_expression = interpolate_first_available(
        model,
        dataset_tag,
        ("tee1.Sxx", "tee1.S", "ht.tee1.Sxx", "ec.tee1.Sxx"),
        physical_coords[leg_probe_mask],
    )
    if seebeck_field is not None:
        for component_index, component in enumerate(components):
            if component.component_type not in {"p_leg", "n_leg"}:
                continue
            component_in_leg_subset = probe_components[leg_probe_mask] == component_index
            seebeck_coefficient[:, component_index] = np.nanmedian(
                seebeck_field[:, component_in_leg_subset], axis=1
            )
    max_contact_area = max((contact.area for contact in contacts), default=1.0)
    edge_index = []
    edge_features = []
    contact_rows = []
    for contact in contacts:
        source = contact.source
        target = contact.target
        electrical = float(conductive_components[source] and conductive_components[target])
        delta = normalized_centers[target] - normalized_centers[source]
        feature = [1.0, electrical, contact.area / max_contact_area, 0.0, 0.0, *delta]
        reverse = [1.0, electrical, contact.area / max_contact_area, 0.0, 0.0, *(-delta)]
        edge_index.extend(((source, target), (target, source)))
        edge_features.extend((feature, reverse))
        contact_rows.append(
            [source, target, contact.axis, *contact.normal, contact.area]
        )

    solution_values = _solution_values(model, dataset_tag, len(temperature))
    tref = _parameter(model, "Tref", float(np.nanmean(temperature)))
    dt0 = _parameter(model, "dT0", 0.0)
    global_features = np.column_stack(
        (
            solution_values,
            np.full(len(solution_values), tref),
            np.full(len(solution_values), dt0),
        )
    )
    probe_target_mask = np.column_stack(
        (np.ones(len(local_coords), dtype=bool), potential_probe_mask)
    )

    metadata = {
        "format_version": 1,
        "dataset_tag": dataset_tag,
        "geometry_dataset_tag": geometry_dataset_tag,
        "component_types": COMPONENT_TYPE_NAMES,
        "field_names": ("temperature", "potential"),
        "global_feature_names": ("solution_value", "Tref", "dT0"),
        "material_property_names": (
            "thermal_conductivity",
            "electrical_conductivity",
            "seebeck_coefficient",
        ),
        "material_source": {
            "thermal_conductivity": "COMSOL ht.kxx median over component probes",
            "electrical_conductivity": "COMSOL ec.sigmaxx median over component probes",
            "seebeck_coefficient": (
                f"COMSOL {seebeck_expression} median over leg probes"
                if seebeck_expression
                else "type fallback: +200 uV/K P-leg, -200 uV/K N-leg, zero elsewhere"
            ),
        },
        "components": [asdict(component) for component in components],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        node_features=node_features,
        node_types=type_ids,
        node_field_mask=node_field_mask,
        node_sizes=sizes,
        thermal_conductivity=thermal_conductivity,
        electrical_conductivity=electrical_conductivity,
        seebeck_coefficient=seebeck_coefficient,
        edge_index=(
            np.asarray(edge_index, dtype=np.int64).T
            if edge_index
            else np.empty((2, 0), dtype=np.int64)
        ),
        edge_features=(
            np.asarray(edge_features, dtype=float)
            if edge_features
            else np.empty((0, 8), dtype=float)
        ),
        contacts=np.asarray(contact_rows, dtype=float),
        global_features=global_features,
        probe_coords=local_coords,
        probe_components=probe_components,
        probe_target_mask=probe_target_mask,
        temperature=temperature,
        potential=potential,
        physical_coords=physical_coords,
    )
    return {
        "output_path": str(output_path),
        "component_count": len(components),
        "contact_count": len(contacts),
        "probe_count": len(local_coords),
        "solution_count": len(temperature),
        "components": [asdict(component) for component in components],
    }
