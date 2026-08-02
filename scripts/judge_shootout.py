"""Phase C Step C2: four-way judge shootout for the second-round semantic verifier.

Runs every item in data/exam_set_v1.json (20 hand-labeled GOOD/BAD priors,
Step C1) through four candidates. Three are real applicants for a cheap
second-round faithfulness check, sharing one word-for-word prompt so the
comparison isn't confounded by prompt differences:

  1. deepseek_self_check — the production extraction model itself (same
     model = same blind spots risk, worth measuring directly)
  2. gpt5_6_luna          — an independent model, same generation as the
     ceiling (#4) but a different codename variant (luna vs. sol) —
     confirmed via the OpenAI API to be a real, distinct model, but its
     actual price/capability tier relative to sol is NOT confirmed (the
     API exposes no metadata distinguishing gpt-5.6's luna/sol/terra
     variants; sol was picked as the flagship on the user's own say-so, not
     from anything queryable). Treat "applicant" here as "cheaper than the
     ceiling by assumption," not by verified pricing.
  3. local_nli            — cross-encoder/nli-deberta-v3-large, zero
     marginal API cost; entailment probability > 0.5 -> SUPPORTED.
     natural-format items ONLY (an NLI model expects prose, not a
     kind/field/value dump) — this candidate is skipped for fielded items.

A fourth, NON-applicant ceiling model sanity-checks the exam set itself
and quantifies what the cheap options give up:

  4. gpt5_6_sol_ceiling — gpt-5.6-sol, confirmed with the user as the
     flagship variant among gpt-5.6's three unlabeled codenames. Not scored
     for "which judge should we use" — see scripts/judge_shootout_report.py
     (Step C3) for how its results get used instead.

Every applicant + the ceiling model gets BOTH the `fielded` and `natural`
presentation of each item (Step C3 wants to know if presentation format
changes judge accuracy); local_nli only ever sees `natural`.

Real API calls for judges 1/2/4 (temperature=0, one short completion per
item, not agentic — cheap and fast, same cost class as
scripts/threshold_judge.py). 20 items x 3 API judges x 2 formats = 120
completions total. local_nli is a one-time ~600MB local model download,
zero further API cost.

Resumable: skips (item_id, judge, format) triples already present in
data/judge_shootout_results.json (fixed filename, not date-stamped — see
collect_threshold_data.py's docstring for why that convention exists).

    uv run python scripts/judge_shootout.py
"""

import json
from pathlib import Path

import litellm
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

EXAM_SET_PATH = Path("data/exam_set_v1.json")
OUT_PATH = Path("data/judge_shootout_results.json")

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-large"
NLI_MODEL_REVISION = "bab4bc7178836f731dcfd18c06ca9def0a137712"

# The 3 applicants share this exact prompt (verbatim, per the task spec) —
# only the model name differs between them. gpt5_6_sol_ceiling reuses the
# same prompt too, purely so its verdicts are comparable to the applicants'.
SHOOTOUT_PROMPT = """You are a strict verification judge for structured scientific claims
extracted from thermoelectric cooler (TEC) literature.

You will be given:
- QUERY: the research question the extraction was answering
- PRIOR: one extracted claim (may be a fielded record or a natural sentence)
- EVIDENCE: the full text of the evidence snippets this prior cites

Your ONLY job is faithfulness: decide whether EVERY statement and EVERY
number in PRIOR is directly stated in, or follows necessarily from, the
EVIDENCE text.

Rules:
- Judge NOT_SUPPORTED if the prior contains ANY of:
  - a number that does not appear in the evidence (allowing trivial
    rounding, e.g. 1.83 -> 1.8)
  - a direction or trend the evidence does not state (e.g. prior says
    "increases" but evidence only says "affects")
  - a recommendation, mechanism, or conclusion the evidence does not state
  - a claim attributed to this evidence that actually isn't in it
- Do NOT use your own domain knowledge to fill gaps. A claim that is
  physically plausible but absent from the evidence is NOT_SUPPORTED.
- Do NOT judge usefulness, importance, or writing quality. A vague but
  faithful prior is SUPPORTED.
- Evidence snippets that merely list references or acknowledgments do not
  support technical claims.

Output JSON only, no markdown fences:
{"verdict": "SUPPORTED" | "NOT_SUPPORTED", "reason": "<one sentence>"}"""

API_JUDGES = {
    # NOTE: this is intentionally read at import time from the same
    # production config (get_llm_model()) used by sciencerag/priors/extract.py,
    # not hardcoded — "self-check" only means something if it's actually
    # today's production model, not whatever it happened to be when this
    # script was written.
    "gpt5_6_luna": "gpt-5.6-luna",
    "gpt5_6_sol_ceiling": "gpt-5.6-sol",
}
FORMATS = ["fielded", "natural"]


def _strip_code_fences(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    return cleaned.strip()


def _fielded_desc(item: dict) -> str:
    return (
        f"kind: {item['kind']}\n"
        f"field: {item['field']}\n"
        f"related_fields: {item['related_fields']}\n"
        f"value: {json.dumps(item['fielded'])}\n"
        # `notes` is supplementary context, not a claim to verify — in
        # production (sciencerag/priors/extract.py's _to_prior) it falls
        # back to the source paper's TITLE whenever the LLM didn't supply
        # its own clarifying note, so it routinely contains attribution
        # text no evidence snippet restates. Labeled here so the judge
        # applies faithfulness checking to `value` (the actual claim), not
        # to this label — found via real shootout results: without this,
        # every prior whose notes happened to carry a paper title got
        # flagged NOT_SUPPORTED regardless of whether the claim itself
        # was correct.
        f"notes (context only, not a claim to verify): {item['notes']}"
    )


def _prior_content(item: dict, fmt: str) -> str:
    return _fielded_desc(item) if fmt == "fielded" else item["natural"]


def _judge_with_api(model: str, item: dict, fmt: str) -> dict:
    user_prompt = (
        f"QUERY: {item['query']}\n\n"
        f"PRIOR ({fmt}):\n{_prior_content(item, fmt)}\n\n"
        f"EVIDENCE:\n{item['evidence_text']}"
    )
    messages = [
        {"role": "system", "content": SHOOTOUT_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        response = litellm.completion(model=model, messages=messages, temperature=0)
    except litellm.BadRequestError as e:
        # gpt-5.6's family (like OpenAI's o-series reasoning models before
        # it) rejects a custom temperature outright — "Only the default (1)
        # value is supported." Not the same failure as a bad prompt/API
        # key, so fall back to the model's default rather than treating it
        # as this item's judgment failing.
        if "temperature" not in str(e):
            raise
        response = litellm.completion(model=model, messages=messages)
    raw = response.choices[0].message.content
    parsed = json.loads(_strip_code_fences(raw))
    return {
        "verdict": parsed["verdict"],
        "reason": parsed.get("reason"),
        "raw_output": raw,
    }


def _load_nli():
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME, revision=NLI_MODEL_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        NLI_MODEL_NAME, revision=NLI_MODEL_REVISION
    )
    model.eval()
    entail_idx = next(i for i, label in model.config.id2label.items() if label == "entailment")
    return tokenizer, model, entail_idx


def _judge_with_nli(tokenizer, model, entail_idx: int, item: dict) -> dict:
    premise = item["evidence_text"]
    hypothesis = item["natural"]
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0]
    entail_prob = probs[entail_idx].item()
    verdict = "SUPPORTED" if entail_prob > 0.5 else "NOT_SUPPORTED"
    return {"verdict": verdict, "entailment_prob": round(entail_prob, 4)}


def main() -> None:
    from sciencerag.common.config import get_llm_model

    judges = {"deepseek_self_check": get_llm_model(), **API_JUDGES}

    items = json.loads(EXAM_SET_PATH.read_text())
    print(f"{len(items)} exam items, {len(judges)} API judges x {len(FORMATS)} formats + local_nli")

    results = []
    if OUT_PATH.exists():
        results = json.loads(OUT_PATH.read_text())
    done = {(r["item_id"], r["judge"], r["format"]) for r in results}

    total = len(items) * len(judges) * len(FORMATS) + len(items)
    n = 0
    for item in items:
        for judge_name, model in judges.items():
            for fmt in FORMATS:
                n += 1
                key = (item["id"], judge_name, fmt)
                if key in done:
                    continue
                print(f"[{n}/{total}] {item['id']} | {judge_name} | {fmt}")
                try:
                    outcome = _judge_with_api(model, item, fmt)
                except Exception as e:  # noqa: BLE001 - keep going past a single bad call
                    print(f"  -> FAILED: {type(e).__name__}: {e}")
                    continue
                print(f"  -> {outcome['verdict']}")
                results.append(
                    {
                        "item_id": item["id"],
                        "true_label": item["label"],
                        "corruption_type": item["corruption_type"],
                        "judge": judge_name,
                        "format": fmt,
                        **outcome,
                    }
                )
                OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print("\nLoading local NLI model (cross-encoder/nli-deberta-v3-large, pinned revision)...")
    tokenizer, nli_model, entail_idx = _load_nli()
    for item in items:
        n += 1
        key = (item["id"], "local_nli", "natural")
        if key in done:
            continue
        print(f"[{n}/{total}] {item['id']} | local_nli | natural")
        outcome = _judge_with_nli(tokenizer, nli_model, entail_idx, item)
        print(f"  -> {outcome['verdict']} (p_entail={outcome['entailment_prob']})")
        results.append(
            {
                "item_id": item["id"],
                "true_label": item["label"],
                "corruption_type": item["corruption_type"],
                "judge": "local_nli",
                "format": "natural",
                **outcome,
            }
        )
        OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"\nDone. {len(results)} judged rows saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
