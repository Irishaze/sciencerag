"""Train the hierarchical 1-20 PN component graph on composed COMSOL fields.

The targets replicate the measured one-pair component fields and accumulate
electric potential along the series path. This is a topology/composition model,
not a substitute for multi-pair COMSOL calibration data.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import torch

from physics_foundation import (
    ComponentGraph,
    ComponentGraphSepONet,
    LossScales,
    MULTI_COMPONENT_TYPES,
    batch_component_graphs,
    compose_multi_pair_graph,
    load_component_case,
    supervised_field_loss,
)

DEFAULT_DATA = PROJECT / "data" / "component_cases" / "tec_1pair_dset3.npz"
DEFAULT_CHECKPOINT = PROJECT / "outputs" / "component_graph_seponet_20pairs.pt"
DEFAULT_SUMMARY = PROJECT / "outputs" / "component_graph_seponet_20pairs.json"
HOLDOUT_PAIR_COUNTS = (3, 7, 13, 19)
HOLDOUT_SOLUTION_INDICES = (4, 9)
FIELD_OFFSET = torch.tensor((300.0, 0.0), dtype=torch.float32)
FIELD_SCALE = torch.tensor((50.0, 5.0), dtype=torch.float32)


def normalise_targets(graph: ComponentGraph) -> ComponentGraph:
    return replace(
        graph,
        probe_targets=(graph.probe_targets - FIELD_OFFSET) / FIELD_SCALE,
    )


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
def evaluate_cases(
    model: ComponentGraphSepONet,
    base_graphs: list[ComponentGraph],
    cases: list[tuple[int, int]],
    probes_per_component: int,
) -> dict[str, object]:
    model.eval()
    temperature_errors = []
    potential_errors = []
    rows = []
    for n_pairs, solution_index in cases:
        graph, _ = compose_multi_pair_graph(
            base_graphs[solution_index],
            n_pairs,
            max_probes_per_component=probes_per_component,
        )
        physical_target = graph.probe_targets
        normalised_graph = normalise_targets(graph)
        physical_prediction = model(normalised_graph) * FIELD_SCALE + FIELD_OFFSET
        error = physical_prediction - physical_target
        temperature = error[:, 0][graph.probe_target_mask[:, 0]]
        potential = error[:, 1][graph.probe_target_mask[:, 1]]
        temperature_errors.append(temperature.square())
        potential_errors.append(potential.square())
        rows.append(
            {
                "n_pairs": n_pairs,
                "current_A": float(base_graphs[solution_index].global_features[0, 0]),
                "temperature_rmse_K": float(temperature.square().mean().sqrt()),
                "potential_rmse_V": float(potential.square().mean().sqrt()),
            }
        )
    model.train()
    return {
        "temperature_rmse_K": float(torch.cat(temperature_errors).mean().sqrt()),
        "potential_rmse_V": float(torch.cat(potential_errors).mean().sqrt()),
        "cases": rows,
    }


def train(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object]]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    base_graphs = [load_component_case(args.data, index) for index in range(11)]
    train_pair_counts = [
        value for value in range(1, 21) if value not in HOLDOUT_PAIR_COUNTS
    ]
    train_solution_indices = [
        value for value in range(11) if value not in HOLDOUT_SOLUTION_INDICES
    ]

    example, _ = compose_multi_pair_graph(
        base_graphs[0], 1, max_probes_per_component=args.probes_per_component
    )
    model_config = {
        "node_dim": example.node_features.shape[1],
        "edge_dim": example.edge_features.shape[1],
        "global_dim": example.global_features.shape[1],
        "num_component_types": len(MULTI_COMPONENT_TYPES),
        "hidden_dim": args.hidden_dim,
        "rank": args.rank,
        "message_passing_steps": args.message_passing_steps,
    }
    model = ComponentGraphSepONet(**model_config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scales = LossScales(temperature=1.0, potential=1.0)

    cache: dict[tuple[int, int], ComponentGraph] = {}

    def get_graph(n_pairs: int, solution_index: int) -> ComponentGraph:
        key = (n_pairs, solution_index)
        if key not in cache:
            graph, _ = compose_multi_pair_graph(
                base_graphs[solution_index],
                n_pairs,
                max_probes_per_component=args.probes_per_component,
            )
            cache[key] = normalise_targets(graph)
        return cache[key]

    history = []
    started = time.perf_counter()
    model.train()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        graphs = []
        for _ in range(args.graphs_per_step):
            n_pairs = random.choice(train_pair_counts)
            solution_index = random.choice(train_solution_indices)
            graphs.append(
                sample_probes(
                    get_graph(n_pairs, solution_index),
                    args.probes_per_graph,
                    generator,
                )
            )
        batch = batch_component_graphs(graphs)
        prediction = model(batch)
        loss = supervised_field_loss(
            prediction,
            batch.probe_targets,
            scales,
            batch.probe_target_mask,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        report_every = max(1, args.steps // 12)
        if step == 0 or (step + 1) % report_every == 0 or step + 1 == args.steps:
            history.append({"step": step + 1, "loss": float(loss.detach())})

    elapsed = time.perf_counter() - started
    seen_pair_unseen_current = [
        (n_pairs, solution)
        for n_pairs in (2, 8, 14, 20)
        for solution in HOLDOUT_SOLUTION_INDICES
    ]
    unseen_pair_seen_current = [
        (n_pairs, solution)
        for n_pairs in HOLDOUT_PAIR_COUNTS
        for solution in (2, 7)
    ]
    double_holdout = [
        (n_pairs, solution)
        for n_pairs in HOLDOUT_PAIR_COUNTS
        for solution in HOLDOUT_SOLUTION_INDICES
    ]
    metrics = {
        "unseen_current": evaluate_cases(
            model, base_graphs, seen_pair_unseen_current, args.eval_probes_per_component
        ),
        "unseen_pair_count": evaluate_cases(
            model, base_graphs, unseen_pair_seen_current, args.eval_probes_per_component
        ),
        "unseen_pair_and_current": evaluate_cases(
            model, base_graphs, double_holdout, args.eval_probes_per_component
        ),
    }
    checkpoint = {
        "model_config": model_config,
        "model_state": model.state_dict(),
        "field_offset": FIELD_OFFSET,
        "field_scale": FIELD_SCALE,
        "component_types": MULTI_COMPONENT_TYPES,
        "max_supported_pairs": 20,
        "holdout_pair_counts": HOLDOUT_PAIR_COUNTS,
        "holdout_solution_indices": HOLDOUT_SOLUTION_INDICES,
        "source_data": str(args.data),
        "training_kind": "one_pair_COMSOL_compositional_topology_model",
    }
    summary = {
        "training_kind": checkpoint["training_kind"],
        "source_data": str(args.data),
        "supported_pair_counts": [1, 20],
        "train_pair_counts": train_pair_counts,
        "holdout_pair_counts": list(HOLDOUT_PAIR_COUNTS),
        "train_solution_indices": train_solution_indices,
        "holdout_solution_indices": list(HOLDOUT_SOLUTION_INDICES),
        "steps": args.steps,
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / args.steps,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
        "validation": metrics,
        "limitations": [
            "Only the one-pair local fields are directly calibrated by COMSOL.",
            "Multi-pair targets assume repeated cells and series-voltage composition.",
            "New multi-pair COMSOL exports are required for engineering accuracy claims.",
        ],
    }
    return checkpoint, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--graphs-per-step", type=int, default=4)
    parser.add_argument("--probes-per-graph", type=int, default=256)
    parser.add_argument("--probes-per-component", type=int, default=12)
    parser.add_argument("--eval-probes-per-component", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--rank", type=int, default=12)
    parser.add_argument("--message-passing-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    if min(
        args.steps,
        args.graphs_per_step,
        args.probes_per_graph,
        args.probes_per_component,
        args.eval_probes_per_component,
    ) < 1:
        parser.error("training counts must be positive")

    checkpoint, summary = train(args)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.checkpoint)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {args.checkpoint}")
    print(f"Elapsed: {summary['elapsed_seconds']:.1f} s")
    for name, metric in summary["validation"].items():
        print(
            f"{name}: T RMSE={metric['temperature_rmse_K']:.4g} K, "
            f"V RMSE={metric['potential_rmse_V']:.4g} V"
        )
    print("Multi-pair COMSOL calibration is still required for engineering accuracy.")


if __name__ == "__main__":
    main()
