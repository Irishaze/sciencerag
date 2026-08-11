"""End-to-end demo: priors -> validate -> report -> KG approval -> ask.

Exercises the real pipeline across all four spec endpoints in one run, in
the order Hermes is expected to call them (spec §2's pipeline order).
Makes real API calls (DeepSeek for extraction/synthesis, PaperQA2 over the
internal corpus) — costs a small amount and takes ~1-2 minutes, same as
scripts/run_regression.py.

Uses one of tec_surrogate's known-solved COMSOL cases (report row 0) as the
"simulation result" being validated, since this script has no live COMSOL
run to draw from — every number that isn't from a live API call is real
bundled data, not fabricated.

    uv run python scripts/demo_end_to_end.py
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from sciencerag.app import app
from sciencerag.priors.kg import query_kg
from sciencerag.validate import tec_bridge

client = TestClient(app)


def _step(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _known_report_case() -> tuple[dict[str, float], dict[str, float]]:
    dataset = tec_bridge.load_report_dataset()
    input_names = list(dataset["input_names"])
    scalar_names = list(dataset["scalar_names"])
    inverse_names = {v: k for k, v in tec_bridge.CONTRACT_TO_LATENT_INPUT.items()}
    design_parameters = {
        inverse_names[name]: float(dataset["X"][0, input_names.index(name)])
        for name in input_names
        if name in inverse_names
    }
    scalar_results = {
        name: float(dataset["scalar_outputs"][0, scalar_names.index(name)]) for name in scalar_names
    }
    return design_parameters, scalar_results


def main() -> None:
    run_id = "demo_run_2026_08_04"

    _step("1/5  sciencerag.priors — retrieving literature priors")
    priors_response = client.post(
        "/sciencerag/priors",
        json={"query": "Bi2Te3 thermoelectric cooler leg length effect on COP and delta_T_max"},
    )
    priors_response.raise_for_status()
    priors_body = priors_response.json()
    print(f"status={priors_body['status']}  priors returned={len(priors_body['priors'])}")
    for prior in priors_body["priors"]:
        print(f"  - [{prior['kind']}] field={prior.get('field')} confidence={prior['confidence']}")
    print(f"coverage: {priors_body['coverage']}")

    _step("2/5  sciencerag.validate — anomaly checks + evaluation")
    design_parameters, scalar_results = _known_report_case()
    latent_state = tec_bridge.load_training_latent()[0].tolist()
    validate_response = client.post(
        "/sciencerag/validate",
        json={
            "run_id": run_id,
            "design_parameters": design_parameters,
            "n_pairs": 1,
            "scalar_results": scalar_results,
            "latent_state": latent_state,
            "priors": priors_body["priors"],
        },
    )
    validate_response.raise_for_status()
    validate_body = validate_response.json()
    print(f"anomalies: {[(a['check'], a['severity']) for a in validate_body['anomalies']]}")
    print(f"evaluation.verdict: {validate_body['evaluation']['verdict']}")
    print(f"update_package.blocked: {validate_body['update_package']['blocked']}")
    kg_candidates = validate_body["update_package"]["kg_candidates"]
    print(f"kg_candidates proposed: {len(kg_candidates)}")

    _step("3/5  sciencerag.report — citation-backed run report")
    report_response = client.post(
        "/sciencerag/report",
        json={
            "run_id": run_id,
            "task_context": {"objective": "characterize a known single-pair Bi2Te3 TEC design"},
            "design_parameters": design_parameters,
            "n_pairs": 1,
            "scalar_results": scalar_results,
            "priors": priors_body["priors"],
            "anomalies": validate_body["anomalies"],
            "evaluation": validate_body["evaluation"],
            "update_package": validate_body["update_package"],
        },
    )
    report_response.raise_for_status()
    report_body = report_response.json()
    print(f"report stored, trace_id={report_body['trace_id']}")
    print("--- markdown preview (first 12 lines) ---")
    print("\n".join(report_body["markdown"].splitlines()[:12]))

    _step("4/5  KG candidate approval (scripts/approve_kg_candidates.py path)")
    if not kg_candidates:
        print("no KG candidates proposed for this run (nothing to approve) — skipping to step 5")
    else:
        from sciencerag.priors.kg import KGSource, add_triple

        for candidate in kg_candidates:
            triple, status = add_triple(
                subject=candidate["subject"],
                relation=candidate["relation"],
                object_value=candidate["object_value"],
                object_unit=candidate["object_unit"],
                conditions=candidate["conditions"],
                confidence=candidate["confidence"],
                run_id=candidate["run_id"],
                sources=[KGSource(type="run", run_id=candidate["run_id"])],
            )
            print(f"  approved: {candidate['relation']} -> triple_id={triple.triple_id} ({status})")

    _step("5/5  sciencerag.ask — question answered from the graph just populated")
    if not kg_candidates:
        print("(skipped — no graph data was added this run)")
    else:
        question = f"What {kg_candidates[0]['relation'].removeprefix('achieves_')} does the Bi2Te3 single-stage TEC achieve?"
        print(f"question: {question!r}")
        ask_response = client.post("/sciencerag/ask", json={"question": question})
        ask_response.raise_for_status()
        ask_body = ask_response.json()
        print(f"fallback_used: {ask_body['fallback_used']}")
        print(f"answer: {ask_body['answer']}")
        print(f"subgraph nodes/edges: {len(ask_body['subgraph']['nodes'])}/{len(ask_body['subgraph']['edges'])}")

    _step("done")
    print(json.dumps({"run_id": run_id, "kg_hits_for_subject": len(query_kg("Bi2Te3 single-stage TEC"))}, indent=2))


if __name__ == "__main__":
    main()
