# Changelog

## M1 — `sciencerag.priors` (internal literature only)

Real PaperQA2-backed retrieval over `corpus/papers/`, LLM-based structured
extraction into the 5-kind `Prior` schema, and a spec-compliant error/audit
envelope. Full contract documented in [README.md](README.md#sciencerag-priors-契约m1).

Key pieces:

- **Schema & contract**: `PriorsRequest`/`PriorsResponse`/`ErrorResponse`
  frozen to `sciencerag/schemas/priors.schema.json`; `Prior.sources`
  requires at least one citation (no uncited claims); `max_priors` requires
  a positive value.
- **Retrieval**: real PaperQA2 index over the internal corpus (135 papers,
  116 unique after dedup), DeepSeek for extraction/summarization, OpenAI
  embeddings.
- **Extraction**: LLM-based structured pipeline (`sciencerag/priors/extract.py`)
  — evidence → prompt → validated JSON with retry → evidence labels
  resolved to real DOIs by our own code (never LLM-generated) → confidence
  from a deterministic formula, not LLM-scored.
- **Quality gates**: evidence below `MIN_EVIDENCE_RELEVANCE` never reaches
  the LLM; priors below `CONFIDENCE_THRESHOLD` are excluded from `priors[]`
  and surfaced transparently in `coverage.gaps` instead.
- **KG-priority stub** (`sciencerag/priors/kg.py`): query-priority-order
  structure in place per spec §3.2, always returns zero hits until M2+
  wires up real graph storage.
- **`allow_external`**: explicit, tested no-op for M1 (spec §9 OQ#1) —
  requesting it adds a transparency note to `coverage.gaps` rather than
  silently doing nothing; real external retrieval deferred to M6.
- **Soft latency guard**: 30s target (spec §9); a slow response still
  returns normally, only logs a warning and records `elapsed_s` in the
  audit log.
- **Audit log**: append-only JSONL, every call fully reconstructable by
  `trace_id` (request, evidence, output, model config, elapsed time),
  verified for both success and error paths.
- **Regression harness**: `tests/fixtures/priors_regression.json` (7
  property-based fixtures spanning all 5 `kind`s + a verified zero-coverage
  edge case) + `scripts/run_regression.py`, gated behind `make test-m1`.

Not in scope for M1 (explicitly deferred): real external retrieval (M6),
real knowledge graph (M2+), `validate`/`report`/`ask` endpoints (M2-M5).
