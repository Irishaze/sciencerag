"""Manual M1-10 verification: build the PaperQA2 index over corpus/papers and
run one real, cited query against it.

This makes real DeepSeek + OpenAI API calls and costs a small amount of
money — it is a manual check, not part of the automated (free, offline)
pytest suite.

    uv run python scripts/verify_paperqa.py
"""

from sciencerag.priors.retrieval import run_query

QUERY = (
    "What design parameters or operating conditions affect the coefficient "
    "of performance (COP) of a thermoelectric cooler?"
)


def main() -> None:
    response = run_query(QUERY)
    session = response.session

    print("=" * 80)
    print("QUERY:", QUERY)
    print("=" * 80)
    print(session.formatted_answer)
    print("=" * 80)
    print(f"cost: ${session.cost:.4f}")
    print(f"contexts used: {len(session.contexts)}")


if __name__ == "__main__":
    main()
