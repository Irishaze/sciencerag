# TEC Component Graph + Local SepONet Prototype

This prototype combines two levels of representation:

- A component graph exchanges thermal and electrical context between ceramic,
  conductor, P-leg, and N-leg nodes.
- A local SepONet decoder maps each node state and normalized local coordinates
  `(xi, eta, zeta)` to temperature and electric potential.

Weights are shared by component type. The same model therefore accepts graphs
with different PN-pair counts without changing its parameter count.

## Current status

Implemented:

- Variable-size component graph tensors and graph batching.
- Directed thermal/electrical message passing in pure PyTorch.
- Per-component-type separable trunks for continuous 3D fields.
- Supervised temperature/potential loss.
- Conservative interface temperature, potential, heat-flux, and current loss.
- Steady thermoelectric PDE residual for constant properties per component.
- Synthetic variable-PN graphs and an end-to-end architecture smoke test.
- COMSOL cumulative-selection discovery for individual component domains.
- Component bounding boxes, local tensor probes, T/V interpolation, and contact graphs.
- Loading any exported current solution as a PyTorch `ComponentGraph`.
- Exact solved COMSOL `ht.kxx`, `ec.sigmaxx`, and `tee1.Sxx` material fields.
- Matched interface face grids with separate thermal and electrical masks.
- Real-data training with held-out operating-current evaluation.
- Hierarchical pair/module virtual nodes for short graph paths up to 20 PN pairs.
- One-pair COMSOL field composition, series-voltage hard constraints, and a
  dedicated 1-20 pair field-prediction service.

Not implemented yet:

- Temperature-dependent material property functions.
- Calibrated loss scales from the COMSOL dataset.
- Multi-geometry and multi-PN-pair training data.
- Production validation, uncertainty, and inference APIs.

The synthetic checkpoint is not a physical TEC surrogate.

## Run

Use the existing environment that contains CPU PyTorch:

```powershell
..\.venv-deepxde\Scripts\python.exe -m unittest tests.test_physics_foundation
..\.venv-deepxde\Scripts\python.exe scripts\11_train_component_graph_demo.py --steps 200
..\.venv-deepxde\Scripts\python.exe scripts\13_train_real_component_graph.py
..\.venv-deepxde\Scripts\python.exe scripts\14_train_multipair_component_graph.py
```

Export the already solved one-pair COMSOL model with the system Python that
contains `mph`:

```powershell
python scripts\12_export_comsol_component_case.py
```

For future Sobol cases, add `--export-fields` to `scripts/05_run_batch.py`. Field export
is opt-in because interpolation adds work and storage to every COMSOL case.

The real-data script defaults to holding out solution indices 4 and 9, which
are 0.5 A and 1.0 A in the reference export. On the current CPU-only machine,
600 steps took about 26 seconds and produced aggregate holdout RMSE of 4.55 K
for temperature and 0.0547 V for potential. This is an integration baseline,
not a production accuracy claim. It tests unseen currents on the same geometry
only.

Checkpoints and JSON training summaries are written under `outputs/`.

## 1-20 pair prototype

`scripts/14_train_multipair_component_graph.py` repeats the five calibrated physical
components of the one-pair COMSOL cell, adds one virtual node per PN pair, and
connects every pair to a module node. The hierarchy gives a physical component
a short path to module-wide context even for a 20-pair graph.

The model trains on pair counts 1-20 while holding out 3, 7, 13, and 19. It also
holds out the 0.5 A and 1.0 A solutions. The current checkpoint produced the
following errors relative to the composed one-pair targets:

- Unseen current: 2.42 K temperature RMSE, 0.334 V potential RMSE.
- Unseen pair count: 1.78 K temperature RMSE, 0.237 V potential RMSE.
- Unseen pair count and current: 2.41 K temperature RMSE, 0.308 V potential RMSE.

Inference applies two hard constraints after SepONet decoding: module
temperature limits are interpolated from the one-pair COMSOL current sweep, and
terminal voltage is the interpolated one-pair span multiplied by the PN-pair
count. SepONet therefore learns the local spatial shape while the known series
law fixes the module voltage scale.

Start the field UI with the environment that contains PyTorch and scikit-learn:

```powershell
..\.venv-deepxde\Scripts\python.exe prediction_server.py --port 8765
```

The 1-pair field is directly anchored to COMSOL. Results for 2-20 pairs are
compositional predictions and must not be treated as multi-pair COMSOL accuracy.
The generated `outputs/comsol_training_batch_50.csv` now includes required
coverage at 8, 12, 16, and 20 pairs for the next calibration run.

## Exported reference materials

The bundled one-pair case contains solved, component-wise median properties:

- Tungsten ceramic domains: approximately 175 W/(m K).
- Copper conductors: approximately 400 W/(m K) and 5.998e7 S/m.
- Thermoelectric legs: approximately 1.568 W/(m K) and 7.893e4 S/m.
- COMSOL `tee1.Sxx`: approximately +217.27 uV/K on the selected N-leg domain
  and -217.27 uV/K on the selected P-leg domain.

The last sign convention is preserved exactly as COMSOL exports it. It must be
checked against terminal-current orientation before interpreting the P/N labels
or adding independently authored constitutive data.

## Required COMSOL case schema

Each real training case must provide:

```text
global_features: I, T_hot, T_cold, convection and other operating values
nodes: component id/type, dimensions, position, material properties
edges: source/target, interface area/normal, contact thermal/electrical resistance
probes: component id, normalized local coordinate, T, V
interfaces: matched coordinates on both sides, normal, T, V, normal q and J
global_targets: Qc, Qh, terminal voltage, COP
```

The current one-pair reference export is stored at
`data/component_cases/tec_1pair_dset3.npz`. The next milestone is exporting a
multi-geometry batch with `scripts/05_run_batch.py --export-fields`, including several
PN-pair counts. Without that dataset, the shared-weight architecture can accept
variable graphs but has no evidence-based claim of variable-topology accuracy.
