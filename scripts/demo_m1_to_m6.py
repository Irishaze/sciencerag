"""Complete demo: M1 through M6, one coherent run.

Extends scripts/demo_end_to_end.py (which covered M1/M2/M4/M5's happy
path) to explicitly exercise every milestone, including the parts that
script didn't reach:

  M1  sciencerag.priors            — literature retrieval
  M2  sciencerag.validate          — (a) a clean "consistent" case,
                                      (b) a moderately-OOD case that
                                      triggers a *warning* (not blocking)
                                      so M3's uncertainty-driven signal
                                      actually fires
  M3  fine-tune suggestion + KG candidates
  M4  sciencerag.report
  M5  KG approval -> sciencerag.ask, BOTH the graph-hit path and the
      fallback-to-literature path
  M6  external retrieval (allow_external=true on a thin-coverage query)
      + batch evidence (spec §3.4)

Real API calls throughout (DeepSeek + PaperQA2 + Semantic Scholar) — this
takes roughly 10-15 minutes and a small real cost, same as
scripts/run_regression.py. Every number that isn't from a live call is
real bundled tec_surrogate data, not fabricated.

    uv run python scripts/demo_m1_to_m6.py
"""

from __future__ import annotations

import json

import numpy as np
from fastapi.testclient import TestClient

from sciencerag.app import app
from sciencerag.priors.kg import KGSource, add_triple, query_kg
from sciencerag.validate import tec_bridge

client = TestClient(app)
RUN_ID = "demo_m1_m6_2026_08_05"


def _step(n: str, title: str) -> None:
    print(f"\n{'=' * 74}\n[{n}] {title}\n{'=' * 74}")


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


def _moderately_ood_latent() -> list[float]:
    """A latent_state at ~93rd percentile of the training self-distance
    distribution — clears the "info" threshold without hitting "blocking",
    so M2's OOD check reports a genuine warning and M3's uncertainty-driven
    fine-tune signal has something real to react to."""
    z = tec_bridge.load_training_latent()
    mean = z.mean(axis=0)
    std = np.where(z.std(axis=0, ddof=1) > 1e-8, z.std(axis=0, ddof=1), 1.0)
    return (mean + 2.0 * std * np.array([1, -1, 1, -1, 1])).tolist()


def main() -> None:
    design_parameters, scalar_results = _known_report_case()

    # ---------------------------------------------------------------- M1
    _step("M1", "sciencerag.priors — literature retrieval")
    priors_resp = client.post(
        "/sciencerag/priors",
        json={"query": "Bi2Te3 thermoelectric cooler leg length effect on COP and delta_T_max"},
    )
    priors_resp.raise_for_status()
    priors_body = priors_resp.json()
    print(f"priors returned: {len(priors_body['priors'])}, coverage: internal_hits={priors_body['coverage']['internal_hits']}")
    for p in priors_body["priors"]:
        print(f"  - [{p['kind']}] field={p.get('field')} confidence={p['confidence']}")

    # ------------------------------------------------------------- M2(a)
    _step("M2a", "sciencerag.validate — clean case (known solved COMSOL geometry)")
    validate_a = client.post(
        "/sciencerag/validate",
        json={
            "run_id": RUN_ID + "_a",
            "design_parameters": design_parameters,
            "n_pairs": 1,
            "scalar_results": scalar_results,
            "field_case_index": 0,
            "priors": [],  # isolate this case from M1's leg_length prior on purpose
        },
    )
    validate_a.raise_for_status()
    validate_a_body = validate_a.json()
    print(f"anomalies: {[(a['check'], a['severity']) for a in validate_a_body['anomalies']]}")
    print(f"evaluation.verdict: {validate_a_body['evaluation']['verdict']}")
    print(f"kg_candidates: {len(validate_a_body['update_package']['kg_candidates'])}")

    # ------------------------------------------------------------- M2(b)
    _step("M2b", "sciencerag.validate — moderately out-of-distribution case (warning, not blocking)")
    validate_b = client.post(
        "/sciencerag/validate",
        json={
            "run_id": RUN_ID + "_b",
            "design_parameters": design_parameters,
            "n_pairs": 1,
            "latent_state": _moderately_ood_latent(),
            "priors": [],
        },
    )
    validate_b.raise_for_status()
    validate_b_body = validate_b.json()
    print(f"anomalies: {[(a['check'], a['severity']) for a in validate_b_body['anomalies']]}")
    print(f"update_package.blocked: {validate_b_body['update_package']['blocked']}")

    # ---------------------------------------------------------------- M3
    _step("M3", "fine-tune suggestion (from M2b's warning) + KG candidates (from M2a)")
    surrogate_update = validate_b_body["update_package"]["surrogate_update"]
    if surrogate_update:
        print(f"recommended_training_samples: {surrogate_update['recommended_training_samples']}")
        print(f"hyperparameter_direction: {surrogate_update['hyperparameter_direction']}")
    else:
        print("(no surrogate_update — unexpected, M2b should have triggered one)")
    kg_candidates = validate_a_body["update_package"]["kg_candidates"]
    print(f"kg_candidates from the clean case: {len(kg_candidates)}")

    # ---------------------------------------------------------------- M4
    _step("M4", "sciencerag.report — citation-backed report for the clean case")
    report_resp = client.post(
        "/sciencerag/report",
        json={
            "run_id": RUN_ID + "_a",
            "task_context": {"objective": "characterize a known single-pair Bi2Te3 TEC design"},
            "design_parameters": design_parameters,
            "n_pairs": 1,
            "scalar_results": scalar_results,
            "priors": [],
            "anomalies": validate_a_body["anomalies"],
            "evaluation": validate_a_body["evaluation"],
            "update_package": validate_a_body["update_package"],
        },
    )
    report_resp.raise_for_status()
    report_body = report_resp.json()
    print(f"report stored, trace_id={report_body['trace_id']}")
    print("\n".join(report_body["markdown"].splitlines()[:10]))

    # ---------------------------------------------------------------- M5
    _step("M5a", "KG approval + sciencerag.ask (graph-hit path)")
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
        print(f"  approved {candidate['relation']} -> {triple.triple_id} ({status})")

    hit_question = "What delta_T_max_K does the Bi2Te3 single-stage TEC achieve?"
    ask_hit = client.post("/sciencerag/ask", json={"question": hit_question})
    ask_hit.raise_for_status()
    ask_hit_body = ask_hit.json()
    print(f"\nQ: {hit_question}")
    print(f"fallback_used={ask_hit_body['fallback_used']}")
    print(f"A: {ask_hit_body['answer']}")

    _step("M5b", "sciencerag.ask — fallback-to-literature path (no graph coverage)")
    fallback_question = "How does pitch between thermoelectric legs affect power density?"
    ask_fallback = client.post("/sciencerag/ask", json={"question": fallback_question})
    ask_fallback.raise_for_status()
    ask_fallback_body = ask_fallback.json()
    print(f"Q: {fallback_question}")
    print(f"fallback_used={ask_fallback_body['fallback_used']}")
    print(f"coverage_note: {ask_fallback_body['coverage_note']}")
    print(f"A: {ask_fallback_body['answer'][:400]}")

    # ---------------------------------------------------------------- M6
    _step("M6a", "external retrieval (allow_external=true, thin-coverage query)")
    external_resp = client.post(
        "/sciencerag/priors",
        json={
            "query": "heatsink fin height and thickness values for thermoelectric cooler cold-side dissipation",
            "allow_external": True,
        },
    )
    external_resp.raise_for_status()
    external_body = external_resp.json()
    print(f"internal_hits={external_body['coverage']['internal_hits']}  external_hits={external_body['coverage']['external_hits']}")
    external_priors = [p for p in external_body["priors"] if p.get("provenance") == "external_unverified"]
    print(f"external_unverified priors: {len(external_priors)}")
    for p in external_priors:
        print(f"  - field={p.get('field')} confidence={p['confidence']} source={p['sources']}")
    if not external_priors:
        print("(no external priors this run — either Semantic Scholar had no DOI'd/abstracted hits, or was rate-limited; degrades gracefully either way, see external_retrieval.py)")

    _step("M6b", "batch evidence (spec §3.4) — one candidate design")
    batch_resp = client.post(
        "/sciencerag/priors/batch_evidence",
        json={
            "candidates": [
                {
                    "candidate_id": "cand_short_legs",
                    "description": "short thermoelectric leg length maximizes COP",
                    "design_parameters": {"leg_length": 0.5},
                }
            ]
        },
    )
    batch_resp.raise_for_status()
    batch_body = batch_resp.json()
    result = batch_body["results"][0]
    print(f"coverage: {result['coverage']}")
    print(f"supporting: {len(result['supporting'])}  refuting: {len(result['refuting'])}  neutral: {len(result['neutral'])}")
    for e in result["supporting"]:
        print(f"  SUPPORTS ({e['doi']}): {e['text'][:150]}")
    for e in result["refuting"]:
        print(f"  REFUTES  ({e['doi']}): {e['text'][:150]}")

    _step("DONE", "summary")
    print(
        json.dumps(
            {
                "run_id": RUN_ID,
                "m1_priors": len(priors_body["priors"]),
                "m2a_verdict": validate_a_body["evaluation"]["verdict"],
                "m2b_ood_severity": next(a["severity"] for a in validate_b_body["anomalies"] if a["check"] == "ood"),
                "m3_finetune_triggered": surrogate_update is not None,
                "m3_kg_candidates": len(kg_candidates),
                "m4_report_trace_id": report_body["trace_id"],
                "m5_graph_hit_fallback_used": ask_hit_body["fallback_used"],
                "m5_no_coverage_fallback_used": ask_fallback_body["fallback_used"],
                "m6_external_hits": external_body["coverage"]["external_hits"],
                "m6_batch_evidence_coverage": result["coverage"],
                "kg_total_triples_for_subject": len(query_kg("Bi2Te3 single-stage TEC")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
