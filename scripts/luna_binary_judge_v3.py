"""k/relevance/confidence plan Part 2 — cross-validate Phase C's judge pick
on real production data.

Phase C's judge shootout picked gpt-5.6-luna as the best KEEP/DROP judge,
but only tested it against a 20-item HAND-LABELED synthetic exam set
(data/exam_set_v1.json) — never on real pipeline output. Separately,
candidate D (llm_confidence_score.py) had luna assign a continuous
confidence score to the same 22 v3 prior samples judge_results_v3.json has
GPT-4o KEEP/DROP verdicts for, and both of D's misses turned out to be
USEFULNESS/off-topic failures (an off-target TEG-not-TEC evidence match, a
too-vague scaling_relationship prior) — exactly the failure mode
threshold_judge.py's PRIOR_RUBRIC (SUPPORTED + USEFUL) was built to catch
as a binary check, not something a corroboration-strength score was ever
meant to.

This script runs luna as a BINARY judge — threshold_judge.py's own
PRIOR_RUBRIC, completely unmodified, just a different model — over those
same 22 samples, and reports agreement against the existing GPT-4o
verdicts. Answers: does Phase C's synthetic-exam-set win actually hold up
against a real, independent judge (GPT-4o) on real production priors?

Real API calls, single completions (not agentic), same cheap scale as
llm_confidence_score.py. Resumable.

    uv run python scripts/luna_binary_judge_v3.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import litellm

from threshold_judge import PRIOR_RUBRIC, _strip_code_fences  # noqa: E402

COLLECTION_PATH = Path("data/threshold_collection_v3_post_numeric_gate.json")
JUDGE_PATH = Path("data/judge_results_v3.json")
OUT_PATH = Path("data/luna_judge_results_v3.json")

MODEL = "gpt-5.6-luna"


def _judge_one(query: str, prior: dict, evidence_text: str) -> dict:
    prior_desc = (
        f"kind: {prior['kind']}\nfield: {prior['field']}\n"
        f"related_fields: {prior.get('related_fields', [])}\n"
        f"value: {json.dumps(prior['value'])}\nnotes: {prior.get('notes')}"
    )
    messages = [
        {"role": "system", "content": PRIOR_RUBRIC},
        {
            "role": "user",
            "content": (
                f"Query: {query}\n\nExtracted prior:\n{prior_desc}\n\n"
                f"Cited evidence text:\n{evidence_text}"
            ),
        },
    ]
    try:
        response = litellm.completion(model=MODEL, messages=messages, temperature=0)
    except litellm.BadRequestError as e:
        if "temperature" not in str(e):
            raise
        response = litellm.completion(model=MODEL, messages=messages)
    raw = response.choices[0].message.content
    return json.loads(_strip_code_fences(raw))


def main() -> None:
    collection = json.loads(COLLECTION_PATH.read_text())
    judged = json.loads(JUDGE_PATH.read_text())

    prior_by_key = {}
    for entry in collection:
        for prior in entry["all_priors"]:
            prior_by_key[(prior["prior_id"], entry["query"])] = (entry, prior)

    prior_judgments = [r for r in judged if r["side"] == "prior"]

    results = []
    if OUT_PATH.exists():
        results = json.loads(OUT_PATH.read_text())
    done = {(r["prior_id"], r["query"]) for r in results}

    for i, r in enumerate(prior_judgments, 1):
        key = (r["prior_id"], r["query"])
        if key in done or key not in prior_by_key:
            continue
        entry, prior = prior_by_key[key]
        evidence_by_key = {
            (ev.get("doi") or "", ev.get("span") or ""): ev for ev in entry["all_evidence"]
        }
        cited_texts = [
            evidence_by_key[(src.get("doi") or "", src.get("span") or "")]["text"]
            for src in prior.get("sources", [])
            if (src.get("doi") or "", src.get("span") or "") in evidence_by_key
        ]
        evidence_text = "\n\n".join(cited_texts)

        print(f"[{i}/{len(prior_judgments)}] {prior['prior_id']} ({r['query'][:50]})")
        try:
            parsed = _judge_one(r["query"], prior, evidence_text)
            print(f"  -> luna={parsed['verdict']} (gpt4o={r['verdict']}) {parsed.get('reason')}")
        except Exception as e:  # noqa: BLE001 - keep going past a single bad sample
            print(f"  -> FAILED: {type(e).__name__}: {e}")
            continue

        results.append(
            {
                "prior_id": prior["prior_id"],
                "query": r["query"],
                "gpt4o_verdict": r["verdict"],
                "luna_verdict": parsed["verdict"],
                "luna_failed_check": parsed.get("failed_check"),
                "luna_reason": parsed.get("reason"),
            }
        )
        OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"\nDone. {len(results)} judged, saved to {OUT_PATH}")

    agree = sum(1 for r in results if r["gpt4o_verdict"] == r["luna_verdict"])
    print(f"\nAgreement: {agree}/{len(results)} ({agree / len(results):.1%})")
    from collections import Counter

    confusion = Counter((r["gpt4o_verdict"], r["luna_verdict"]) for r in results)
    print("Confusion (gpt4o, luna) -> count:")
    for k, v in confusion.items():
        print(f"  {k} -> {v}")


if __name__ == "__main__":
    main()
