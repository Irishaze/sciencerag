# ScienceRAG — Feature Delivery + Adversarial Security Review

**Date:** 2026-08-14/15
**Scope:** Full ScienceRAG service (`sciencerag.priors`, `sciencerag.validate`, `sciencerag.report`, `sciencerag.ask`, `sciencerag.kg_approval`) and its Docker deployment.
**Commits:** [`37378f8`](../../commit/37378f8)…[`ffd00d4`](../../commit/ffd00d4) (9 commits, see [Commit log](#commit-log))

**Note:** this document covers the security-angle review specifically. A third pass — real, twice-repeated end-to-end calls against the live deployment for `priors`/`validate`/`ask` (Chinese and English), plus deeper knowledge-graph rendering checks — found and fixed two more real, non-security functional bugs (a production regression that silently dropped ranking annotations, and a PDF-rendering glyph-corruption issue for Chinese text). See `DELIVERY_REPORT.md` §4 for that round.

---

## 1. Summary

This round of work had two parts:

1. **Feature delivery** — knowledge-graph entity identity + AI-generated ontology, Chinese-language query support, real value-based ranking for comparison questions, a web approval panel for knowledge candidates, and confidence scores calibrated against real leave-one-out model-error data instead of hand-picked constants.
2. **Adversarial security review** — two rounds, systematically attacking the whole service across six angles (abnormal input, boundary conditions, network failure, concurrency, security, user misuse). **3 real, reproduced bugs found and fixed**, each with a live end-to-end repro against a running server, a regression test, and a re-verification against the rebuilt Docker deployment.

All 419 tests pass. The service is redeployed and the fixes have been re-attacked against the live container to confirm they hold in production, not just in local testing.

---

## 2. Feature work delivered

### 2.1 Knowledge-graph entity identity + AI-generated ontology
Previously, a triple's `subject` string did double duty as both "what type of thing is this" and "which specific thing is this" — two different designs with the same subject string collapsed onto one graph node, and the graph rendered as disconnected clusters with no real relationships between designs, materials, and simulation runs.

Fixed by splitting `entity_id` (a deterministic hash of subject + conditions — identifies *which* thing) from `entity_type` (classified against an AI-generated ontology built from the simulation contract — identifies *what kind* of thing). Added entity-to-entity link triples (`SimulationRun → TECDesign`, `SimulationRun → Material`) so the graph now renders as an actually-connected structure.

### 2.2 Chinese-language query support
The knowledge graph's keyword-overlap search only tokenized ASCII alphanumeric runs — a pure-Chinese question tokenized to an empty set and silently matched nothing, even when the graph had directly relevant data. Fixed with jieba segmentation for CJK text, and by including the AI-generated `relation_description` (the only Chinese-language field on a triple) in the text actually being matched against.

Separately, PaperQA2's own literature-search agent was found to burn 2–3 extra full search rounds (several minutes) reformulating a Chinese query into English itself, since the internal corpus is all-English papers. A translation step now runs before the literature search call, cutting this to one round.

### 2.3 Real ranking for comparison questions
A question like "which design has the highest ΔT" used to be treated as a plain keyword match — every candidate design scored similar relevance and the LLM was left to guess which one was "best" from an unordered list, with no real comparison ever computed. `rank_kg_entities()` now detects superlative language and genuinely sorts candidates by the actual field value, in both `/ask` and the M1 priors pipeline.

### 2.4 Web approval panel + literature seeding
Added a full web UI (`/app/approval`) for reviewing and approving knowledge-graph candidates, as an alternative to the CLI script — both share one approval implementation so they can't drift apart. Added a literature-seeding pipeline that converts `sciencerag.priors` literature findings into graph candidates, so the graph can be pre-populated before any simulation runs exist. The human-approval gate remains the only write path into the graph in both cases.

### 2.5 Confidence calibration from real data
KG-candidate confidence was previously a flat 0.7 ("consistent") / 0.4 ("insufficient benchmark") regardless of how close a match actually was. A real leave-one-out cross-validation sweep of the deployed surrogate model (`scripts/loo_scalar_error_sweep.py`) found the underlying 5% benchmark tolerance was badly mismatched per field — some fields generalize to <1.5% error, others have 90th-percentile error as high as 219% against the same training data, meaning "consistent" was effectively unreachable for those fields from a genuine prediction (confirmed separately: every historical "consistent" verdict in the audit log had exactly zero deviation — an echoed benchmark value, never a real prediction). Tolerance is now per-field, calibrated to real error data. Confidence is now a continuous score reflecting how close the actual match was, and the concrete numbers behind it (real deviation %, which benchmark case) travel with the triple instead of being collapsed into an opaque number.

---

## 3. Adversarial security review

**Methodology:** two rounds, systematically attacking `sciencerag.report`, `sciencerag.priors` (including external retrieval — Semantic Scholar/arXiv download), `sciencerag.validate`, `sciencerag.ask`, `sciencerag.kg_approval` (CLI + web), and cross-cutting infrastructure (Docker, static pages, audit logging), from six angles: abnormal input, boundary conditions, network failure, concurrency, security, and user misuse. Every finding below was reproduced against a live running server before being fixed, and re-verified against the rebuilt Docker container after.

### Finding 1 — CRLF/quote injection → filesystem pollution + connection crash
**Severity:** Medium
**Component:** `sciencerag.report`, shared by `kg_candidate_store`/`kg_approval`

**Repro:** `POST /sciencerag/report` with `run_id: 'inject"x\r\nX-Injected-Header: pwned'`.

**Impact:** The path-safety validator only blocked `/`, `\`, and null bytes. The request succeeded (HTTP 200) and wrote two real files to disk with literal `\r\n` and `"` bytes in their filenames (`data/reports/`, which is bind-mounted to the host — not just inside the container). Fetching that report's `/pdf` endpoint then reached `Content-Disposition: inline; filename="{stem}.pdf"` with the same raw string — the underlying HTTP library (h11) correctly refused to emit the resulting malformed header (so real HTTP response-splitting was **not** actually achievable), but did so by raising an exception past this project's own error handling, so the client got a dropped connection instead of a clean error.

**Fix:** Widened the shared path-safety validator's blocklist to also reject `"`, `\r`, `\n`. One fix covers report generation, KG-candidate file storage, and the approval panel's stem parameter, since all three share this validator.

**Verified:** Live repro against the local server, then re-attacked against the rebuilt Docker container — now returns a clean `422` with no files created.

### Finding 2 — SSRF via redirect
**Severity:** Medium-High
**Component:** `sciencerag.priors` external retrieval (Semantic Scholar/arXiv PDF download)

**Repro (mocked, real-world trigger is a compromised/spoofed API response):** a `pdf_url` that itself resolves to a public IP (passing the existing SSRF check) responds with a redirect to `http://169.254.169.254/...` (a cloud metadata address).

**Impact:** The SSRF guard validated the initial URL, then the download used `follow_redirects=True`, which transparently follows any redirect chain with no re-validation of the destination. Since `pdf_url` comes from a public API response the code's own comments already flag as adversarially-influenceable, a malicious or compromised response could redirect the fetch to an internal/metadata address, and the response would be saved to disk and fed to the extraction LLM as if it were trusted paper text.

**Fix:** Redirects are now followed manually (capped at 5 hops), re-validating the SSRF check at every hop. Normal public-to-public redirects (e.g. a DOI resolver) still work correctly.

**Verified:** Two new regression tests — one confirming a redirect to an internal address is refused and never even requested, one confirming a redirect between two public addresses still succeeds.

### Finding 3 — Unbounded batch-evidence candidate count (cost/DoS)
**Severity:** Low-Medium
**Component:** `sciencerag.priors.batch_evidence`

**Impact:** `POST /sciencerag/priors/batch_evidence` runs one real literature-retrieval query (embedding + LLM calls) per candidate, sequentially, with no concurrency limit — and the request's `candidates` list had no upper bound at all. A single request with a very large candidate list would fan out into an unbounded number of billed API calls and could tie up the request indefinitely.

**Fix:** Capped at 20 candidates, matching the feature's intended scale (comparing a pool of candidate designs, per spec §3.4 — not an unbounded batch).

**Verified:** Regression test confirms a request over the cap is rejected before any real work starts (the retrieval function is asserted never-called).

### Areas checked and confirmed solid (no changes needed)
- PDF report rendering's existing HTML-escaping (SSRF/tag-injection guard) — tested directly against `<img>`/`<iframe>`/`<svg><image>` payloads, all neutralized correctly.
- External-PDF download's existing hardening (size cap, PDF-magic-byte verification, atomic write) — solid.
- Concurrent knowledge-graph writes (20 threads) and concurrent approval-batch archiving (20 threads) — zero errors, no lost writes, no corrupted files.
- `sciencerag.validate`'s out-of-distribution check — already handles the small-sample (n=31) statistics honestly and rejects dimension-mismatched/malformed `latent_state` cleanly.
- The CLI approval script (`--file` bypass path) still runs full Pydantic validation via `KGCandidate.model_validate()` — confidence bounds and other constraints aren't bypassable that way.
- No `dangerouslySetInnerHTML` anywhere in the React frontend; the `/demo` static page's own XSS fix (from an earlier pass) re-verified against real payloads.

### Noted, not changed — flagged for a deployment decision
1. **`docker-compose.yml` binds port 8000 to all network interfaces**, and the service has no authentication anywhere. On a trusted local machine this is fine; on a shared/untrusted network, anyone on the same LAN can reach the approval panel (can approve/pollute the knowledge graph) and any endpoint that calls a billed LLM API. If this ever needs to run somewhere other than a personal machine, recommend binding to `127.0.0.1:8000:8000` instead, or adding auth.
2. **`PriorsRequest.max_priors` has no upper bound** (unlike `AskRequest.max_hits`, which caps at 50). No exploitable path was found — it only affects final truncation, not extra computation — but it's a minor API-consistency gap worth closing opportunistically.
3. **Surrogate model reliability for `total_resistance_ohm`/`optimal_current_A`/`max_heat_dissipation_W`** — the real leave-one-out sweep behind the confidence calibration (§2.5) found these three fields have 90th-percentile prediction error of 89–219% against the current 31-sample training set. This isn't something this review fixes; it's a signal that the surrogate model itself may need more training data or different features for these fields specifically.

---

## 4. Verification

- **419/419 tests pass** (`uv run pytest`), including all new regression tests added by this review.
- All fixes reproduced against a real running server *before* the fix, and re-attacked with the identical payload against the rebuilt Docker container *after* — confirmed the fix holds in the actual deployed environment, not just local test isolation.
- No secrets or `.env` values are committed or baked into the Docker image (`.dockerignore` excludes `.env*`; `data/` — which holds the real knowledge graph, reports, and audit log — is bind-mounted, not baked in).

## 5. Commit log

| Commit | Summary |
|---|---|
| `37378f8` | KG entity-identity graph, AI-generated ontology, approval web panel, Chinese query support |
| `fec65c9` | Per-field validate confidence calibration from real leave-one-out data |
| `04e0a3a` | Adversarial-review fix: CRLF/quote injection, SSRF-via-redirect bypass |
| `fc5cd9d` | Adversarial-review fix: unbounded batch_evidence candidate count |
| `4b0bbf1` | Frontend: extract shared icon components |
| `ed6a615` | Fix: PDF report renderer dropped every Chinese label (tofu boxes) |
| `a615053` | Docs: update README and spec for Chinese/ranking support, confidence calibration |
| `294e875` | Fix: M1 priors endpoint silently dropped ranking notes (real regression, found via real-call verification) |
| `ffd00d4` | Fix: CJK PDF bullet/separator glyph corruption; confidence prior_comparison investigation |
