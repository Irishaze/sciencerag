.PHONY: test-m1 test-m1-fast

# Free/fast: the deterministic unit + schema + smoke test suite (no real
# API calls). Run this on every change.
test-m1-fast:
	uv run pytest -q

# Full M1 acceptance gate (spec §8: "每次修改提示词、检索参数、阈值或语料库后
# 必须运行"): the fast suite, plus the real regression fixtures against the
# live pipeline (scripts/run_regression.py — real API calls, costs money).
test-m1: test-m1-fast
	uv run python scripts/run_regression.py
