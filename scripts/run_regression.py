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

    uv run python scripts/run_regression.py
"""

import json
import sys
import time
from pathlib import Path

import jsonschema

from sciencerag.priors.regression import check_fixture, load_fixtures
from sciencerag.priors.retrieval import build_priors_response

FIXTURES_PATH = Path("tests/fixtures/priors_regression.json")
SCHEMA_PATH = Path("sciencerag/schemas/priors.schema.json")


def main() -> None:
    fixtures = load_fixtures(FIXTURES_PATH)
    response_schema = json.loads(SCHEMA_PATH.read_text())["PriorsResponse"]

    results = []
    for i, fixture in enumerate(fixtures, 1):
        print(f"\n[{i}/{len(fixtures)}] {fixture.id}: {fixture.query}")
        t0 = time.time()
        try:
            response = build_priors_response(fixture.query)
        except Exception as e:  # noqa: BLE001 - a fixture failure shouldn't kill the whole run
            elapsed = time.time() - t0
            print(f"  -> HARD FAILURE: {type(e).__name__}: {e}")
            results.append((fixture.id, False, [f"{type(e).__name__}: {e}"], elapsed))
            continue

        elapsed = time.time() - t0
        violations = []
        try:
            jsonschema.validate(instance=response.model_dump(), schema=response_schema)
        except jsonschema.ValidationError as e:
            violations.append(f"schema validation failed: {e.message}")

        violations.extend(check_fixture(fixture, response))

        passed = not violations
        status = "PASS" if passed else "FAIL"
        print(f"  -> {status} ({len(response.priors)} priors, {elapsed:.0f}s)")
        for v in violations:
            print(f"     - {v}")
        results.append((fixture.id, passed, violations, elapsed))

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
