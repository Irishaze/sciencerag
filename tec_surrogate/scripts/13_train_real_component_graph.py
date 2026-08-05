"""Train and evaluate the component-graph SepONet on a COMSOL field export.

The default split holds out the fifth and tenth current solutions (0.5 A and
1.0 A in the bundled one-pair case). This measures interpolation across an
operating parameter only; it does not measure generalization to new geometry
or a different PN-pair count.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import torch

from physics_foundation import (
    ComponentGraph,
    ComponentGraphSepONet,
    LossScales,
    batch_component_graphs,
    interface_conservation_loss,
    load_component_case,
    load_component_interfaces,
    supervised_field_loss,
)

DEFAULT_DATA = PROJECT / "data" / "component_cases" / "tec_1pair_dset3.npz"
DEFAULT_CHECKPOINT = PROJECT / "outputs" / "component_graph_seponet_real.pt"
DEFAULT_SUMMARY = PROJECT / "outputs" / "component_graph_seponet_real.json"


def parse_indices(value: str, solution_count: int) -> list[int]:
    indices = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not indices:
        raise ValueError("at least one holdout index is required")
    if indices[0] < 0 or indices[-1] >= solution_count:
        raise ValueError(f"holdout indices must lie in [0, {solution_count})")
    if len(indices) == solution_count:
        raise ValueError("the holdout cannot contain every solution")
    return indices


def normalise_globals(
    graphs: list[ComponentGraph], mean: torch.Tensor, scale: torch.Tensor
) -> list[ComponentGraph]:
    return [
        replace(graph, global_features=(graph.global_features - mean) / scale)
        for graph in graphs
    ]


def sample_probes(
    graph: ComponentGraph, count: int, generator: torch.Generator
) -> ComponentGraph:
    if count >= len(graph.probe_coords):
        return graph
    indices = torch.randperm(len(graph.probe_coords), generator=generator)[:count]
    return replace(
        graph,
        probe_coords=graph.probe_coords[indices],
        probe_components=graph.probe_components[indices],
        probe_targets=graph.probe_targets[indices],
        probe_target_mask=graph.probe_target_mask[indices],
    )


@torch.no_grad()
def evaluate(
    model: ComponentGraphSepONet,
    graphs: list[ComponentGraph],
    raw_currents: list[float],
) -> dict[str, object]:
    model.eval()
    rows = []
    squared_temperature = []
    squared_potential = []
    for graph, current in zip(graphs, raw_currents):
        prediction = model(graph)
        error = prediction - graph.probe_targets
        temperature_error = error[:, 0][graph.probe_target_mask[:, 0]]
        potential_error = error[:, 1][graph.probe_target_mask[:, 1]]
        squared_temperature.append(temperature_error.square())
        squared_potential.append(potential_error.square())
        rows.append(
            {
                "current": current,
                "temperature_rmse_K": float(temperature_error.square().mean().sqrt()),
                "potential_rmse_V": float(potential_error.square().mean().sqrt()),
            }
        )
    return {
        "temperature_rmse_K": float(torch.cat(squared_temperature).mean().sqrt()),
        "potential_rmse_V": float(torch.cat(squared_potential).mean().sqrt()),
        "by_solution": rows,
    }


def train(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object]]:
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    first = load_component_case(args.data, 0)
    import numpy as np

    with np.load(args.data, allow_pickle=False) as data:
        solution_count = len(data["global_features"])
        raw_globals = torch.as_tensor(data["global_features"], dtype=torch.float32)
    holdout_indices = parse_indices(args.holdout, solution_count)
    train_indices = [index for index in range(solution_count) if index not in holdout_indices]
    graphs = [first] + [
        load_component_case(args.data, index) for index in range(1, solution_count)
    ]

    global_mean = raw_globals[train_indices].mean(dim=0, keepdim=True)
    global_scale = raw_globals[train_indices].std(dim=0, keepdim=True, unbiased=False)
    global_scale = torch.where(global_scale > 1.0e-8, global_scale, torch.ones_like(global_scale))
    graphs = normalise_globals(graphs, global_mean, global_scale)
    train_graphs = [graphs[index] for index in train_indices]
    holdout_graphs = [graphs[index] for index in holdout_indices]

    model_config = {
        "node_dim": first.node_features.shape[1],
        "edge_dim": first.edge_features.shape[1],
        "global_dim": first.global_features.shape[1],
        "num_component_types": 4,
        "hidden_dim": args.hidden_dim,
        "rank": args.rank,
        "message_passing_steps": args.message_passing_steps,
    }
    model = ComponentGraphSepONet(**model_config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scales = LossScales(
        temperature=50.0,
        potential=1.0,
        heat_flux=1.0e6,
        current_density=1.0e6,
    )
    interfaces = load_component_interfaces(args.data, args.interface_grid)

    history = []
    started = time.perf_counter()
    model.train()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        sampled = [
            sample_probes(graph, args.probes_per_solution, generator)
            for graph in train_graphs
        ]
        batch = batch_component_graphs(sampled)
        prediction = model(batch)
        data_loss = supervised_field_loss(
            prediction, batch.probe_targets, scales, batch.probe_target_mask
        )
        interface_loss = prediction.sum() * 0.0
        if args.interface_weight > 0 and step % args.interface_every == 0:
            graph_index = step % len(train_graphs)
            interface_graph = train_graphs[graph_index]
            node_state = model.encode_nodes(interface_graph)
            interface_loss, _ = interface_conservation_loss(
                model, interface_graph, node_state, interfaces, scales
            )
        loss = data_loss + args.interface_weight * interface_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        report_every = max(1, args.steps // 10)
        if step == 0 or (step + 1) % report_every == 0 or step + 1 == args.steps:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "data_loss": float(data_loss.detach()),
                    "interface_loss": float(interface_loss.detach()),
                }
            )

    elapsed = time.perf_counter() - started
    train_currents = [float(raw_globals[index, 0]) for index in train_indices]
    holdout_currents = [float(raw_globals[index, 0]) for index in holdout_indices]
    train_metrics = evaluate(model, train_graphs, train_currents)
    holdout_metrics = evaluate(model, holdout_graphs, holdout_currents)

    checkpoint = {
        "model_config": model_config,
        "model_state": model.state_dict(),
        "global_mean": global_mean,
        "global_scale": global_scale,
        "component_types": ("ceramic", "conductor", "p_leg", "n_leg"),
        "field_names": model.field_names,
        "train_indices": train_indices,
        "holdout_indices": holdout_indices,
        "source_data": str(args.data),
    }
    summary = {
        "training_kind": "real_COMSOL_single_geometry_current_interpolation",
        "source_data": str(args.data),
        "steps": args.steps,
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / args.steps,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_indices": train_indices,
        "holdout_indices": holdout_indices,
        "train_metrics": train_metrics,
        "holdout_metrics": holdout_metrics,
        "history": history,
        "interface_weight": args.interface_weight,
        "note": (
            "This split tests unseen currents for one geometry. New geometry and "
            "variable PN counts require multi-geometry exports."
        ),
    }
    return checkpoint, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--holdout", default="4,9", help="zero-based solution indices")
    parser.add_argument("--probes-per-solution", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--message-passing-steps", type=int, default=3)
    parser.add_argument("--interface-grid", type=int, default=3)
    parser.add_argument("--interface-weight", type=float, default=1.0e-4)
    parser.add_argument("--interface-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    if args.steps < 1 or args.probes_per_solution < 1 or args.interface_every < 1:
        parser.error("steps, probes-per-solution, and interface-every must be positive")
    if args.interface_weight < 0:
        parser.error("interface-weight cannot be negative")

    checkpoint, summary = train(args)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.checkpoint)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {args.checkpoint}")
    print(f"Elapsed: {summary['elapsed_seconds']:.1f} s")
    print(
        "Train RMSE: "
        f"T={summary['train_metrics']['temperature_rmse_K']:.4g} K, "
        f"V={summary['train_metrics']['potential_rmse_V']:.4g} V"
    )
    print(
        "Holdout RMSE: "
        f"T={summary['holdout_metrics']['temperature_rmse_K']:.4g} K, "
        f"V={summary['holdout_metrics']['potential_rmse_V']:.4g} V"
    )
    print(summary["note"])


if __name__ == "__main__":
    main()
