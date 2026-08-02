"""M1-18: regression runner for sciencerag.priors.

Runs every fixture in tests/fixtures/priors_regression.json against the
REAL pipeline (real DeepSeek/OpenAI API calls, costs money, ~2-3 minutes
per fixture) and checks two things per spec §8's evaluation guidance:

  1. Schema legality: the response validates against the frozen
     sciencerag/schemas/priors.schema.json contract.
  2. Citation existence + the fixture's other property-based expectations
     (min_priors, must_have_kinds, must_cite_real_doi, allow_zero_priors)
     via sciencerag/priors/regression.py's check_fixture().

Exits non-zero if any fixture fails, so it can gate `make test-m1` (M1-21).
Run after any change to prompts, retrieval params, thresholds, or the
corpus (spec §8: "每次修改提示词、检索参数、阈值或语料库后必须运行").

Fixtures marked `known_flaky` (see RegressionFixture) run up to 3x and pass
on majority instead of a single attempt — real repeated-query testing found
PaperQA2's agent_llm tool-selection loop returns substantially different
evidence/priors run-to-run for a handful of thin-coverage queries, even on
an unchanged corpus and code; a fixed `seed` was tried and did not fix it.
Single-run pass/fail on those specific fixtures was mostly measuring which
way the coin landed, not the pipeline's actual behavior.

    uv run python scripts/run_regression.py
"""

import json
import sys
import time
from pathlib import Path

import jsonschema

from sciencerag.priors.regression import RegressionFixture, check_fixture, load_fixtures
from sciencerag.priors.retrieval import build_priors_response

FIXTURES_PATH = Path("tests/fixtures/priors_regression.json")
SCHEMA_PATH = Path("sciencerag/schemas/priors.schema.json")

# known_flaky fixtures get this many attempts, majority-vote decides pass/fail
# (see RegressionFixture.known_flaky's docstring for why single-run isn't a
# reliable signal for these). Non-flaky fixtures always run exactly once.
FLAKY_ATTEMPTS = 3


def _run_once(
    fixture: RegressionFixture, response_schema: dict
) -> tuple[bool, list[str], int, float]:
    """One real pipeline call + check. Returns (passed, violations, n_priors, elapsed)."""
    t0 = time.time()
    try:
        response, _filtered_material_count = build_priors_response(fixture.query)
    except Exception as e:  # noqa: BLE001 - a fixture failure shouldn't kill the whole run
        return False, [f"{type(e).__name__}: {e}"], 0, time.time() - t0

    elapsed = time.time() - t0
    violations = []
    try:
        jsonschema.validate(instance=response.model_dump(), schema=response_schema)
    except jsonschema.ValidationError as e:
        violations.append(f"schema validation failed: {e.message}")
    violations.extend(check_fixture(fixture, response))
    return not violations, violations, len(response.priors), elapsed


def main() -> None:
    fixtures = load_fixtures(FIXTURES_PATH)
    response_schema = json.loads(SCHEMA_PATH.read_text())["PriorsResponse"]

    results = []
    for i, fixture in enumerate(fixtures, 1):
        print(f"\n[{i}/{len(fixtures)}] {fixture.id}: {fixture.query}")
        n_attempts = FLAKY_ATTEMPTS if fixture.known_flaky else 1

        attempts = []
        for attempt_num in range(1, n_attempts + 1):
            passed, violations, n_priors, elapsed = _run_once(fixture, response_schema)
            label = f"attempt {attempt_num}/{n_attempts}" if n_attempts > 1 else "  ->"
            status = "PASS" if passed else "FAIL"
            print(f"  {label} {status} ({n_priors} priors, {elapsed:.0f}s)")
            for v in violations:
                print(f"     - {v}")
            attempts.append((passed, violations, elapsed))

        n_ok = sum(1 for passed, _, _ in attempts if passed)
        passed = n_ok > n_attempts / 2  # majority; n_attempts=1 -> needs the one attempt to pass
        if n_attempts > 1:
            print(f"  -> {'PASS' if passed else 'FAIL'} (majority: {n_ok}/{n_attempts} attempts passed)")
        # Report the first failing attempt's violations for the summary (or
        # empty if it passed) — every attempt's own violations already
        # printed above, this is just what surfaces in FAILED fixtures below.
        violations = next((v for ok, v, _ in attempts if not ok), [])
        elapsed_total = sum(e for _, _, e in attempts)
        results.append((fixture.id, passed, violations, elapsed_total))

    n_passed = sum(1 for _, passed, _, _ in results if passed)
    print(f"\n\n{n_passed}/{len(results)} fixtures passed.")
    if n_passed < len(results):
        print("FAILED fixtures:")
        for fid, passed, violations, _ in results:
            if not passed:
                print(f"  - {fid}: {violations}")
        sys.exit(1)


if __name__ == "__main__":
    main()
