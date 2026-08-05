"""Train the component-graph SepONet on synthetic fields as a smoke test.

This script validates the variable-topology architecture only. It does not
produce a calibrated TEC surrogate; real training requires component-wise
temperature and potential fields exported from COMSOL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import torch

from physics_foundation import (
    ComponentGraphSepONet,
    LossScales,
    batch_component_graphs,
    supervised_field_loss,
)
from physics_foundation.synthetic import COMPONENT_TYPES, build_synthetic_tec_graph

DEFAULT_CHECKPOINT = PROJECT / "outputs" / "component_graph_seponet_demo.pt"
DEFAULT_SUMMARY = PROJECT / "outputs" / "component_graph_seponet_demo.json"


def train_demo(steps: int = 200, learning_rate: float = 2e-3) -> tuple[dict, dict]:
    torch.manual_seed(42)
    examples = [build_synthetic_tec_graph(n_pairs)[0] for n_pairs in (1, 2, 3, 4)]
    graph = batch_component_graphs(examples)
    model_config = {
        "node_dim": graph.node_features.shape[1],
        "edge_dim": graph.edge_features.shape[1],
        "global_dim": graph.global_features.shape[1],
        "num_component_types": len(COMPONENT_TYPES),
        "hidden_dim": 32,
        "rank": 12,
        "message_passing_steps": 3,
    }
    model = ComponentGraphSepONet(**model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scales = LossScales(temperature=50.0, potential=5.0)

    history = []
    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(graph)
        loss = supervised_field_loss(
            prediction, graph.probe_targets, scales, graph.probe_target_mask
        )
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % max(1, steps // 10) == 0:
            history.append({"step": step + 1, "loss": float(loss.detach())})

    checkpoint = {
        "model_config": model_config,
        "model_state": model.state_dict(),
        "component_types": COMPONENT_TYPES,
        "field_names": model.field_names,
        "training_kind": "synthetic_architecture_smoke_test",
    }
    summary = {
        "training_kind": "synthetic_architecture_smoke_test",
        "steps": steps,
        "pair_counts": [1, 2, 3, 4],
        "node_count": len(graph.node_features),
        "probe_count": len(graph.probe_coords),
        "initial_loss": history[0]["loss"],
        "final_loss": history[-1]["loss"],
        "history": history,
        "warning": "Architecture validation only; not a calibrated COMSOL surrogate.",
    }
    return checkpoint, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    checkpoint, summary = train_demo(args.steps, args.learning_rate)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.checkpoint)
    DEFAULT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved architecture checkpoint: {args.checkpoint}")
    print(f"Initial loss: {summary['initial_loss']:.6g}")
    print(f"Final loss:   {summary['final_loss']:.6g}")
    print(summary["warning"])


if __name__ == "__main__":
    main()
