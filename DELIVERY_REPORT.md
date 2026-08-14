# ScienceRAG — Project Delivery Report

**Date:** 2026-08-15
**Repository:** [github.com/Irishaze/sciencerag](https://github.com/Irishaze/sciencerag)

---

## 1. What ScienceRAG is

ScienceRAG is the scientific-literature and knowledge-graph service inside a thermoelectric-cooler (TEC) simulation closed loop. It gives the **Hermes** simulation agent evidence-backed priors before an experiment, validates a simulation's results against literature and known benchmark cases after one, generates cited reports, and answers knowledge-graph questions — so Hermes's design decisions are grounded in retrievable, cited evidence rather than model intuition alone.

Full design spec: `docs/spec/sciencerag_spec_zh.md`.

## 2. Architecture at a glance

A single FastAPI process, one router per endpoint family, 13 real HTTP routes, ~6,600 lines of Python across the five service modules, plus a React/Vite frontend and a static-page fallback:

| Endpoint | Purpose |
|---|---|
| `POST /sciencerag/priors` | Literature retrieval → LLM-extracted, evidence-traceable structured priors |
| `POST /sciencerag/priors/batch_evidence` | Supporting/contradicting evidence per candidate design, for an external ranking mechanism |
| `POST /sciencerag/validate` | Post-simulation result checking: out-of-distribution detection, benchmark/literature comparison, fine-tune suggestions, knowledge-candidate extraction |
| `POST /sciencerag/report` | Assembles a cited Markdown/PDF report from a run's design, validation, and priors |
| `POST /sciencerag/ask` | Knowledge-graph Q&A, falling back to literature retrieval when the graph has no answer |
| `GET /sciencerag/graph` | Full knowledge-graph dump for the visualization UI |
| `/sciencerag/kg_candidates/*` | Web approval panel for knowledge-graph candidates (read pending, approve/reject) |

**Storage:** literature corpus (171 papers) indexed via PaperQA2; knowledge graph as a flat, file-locked JSON store (`data/kg/graph.json`) — no database dependency; reports and audit log likewise on disk. All runtime state is bind-mounted in Docker, not baked into the image.

**LLM backend:** DeepSeek via LiteLLM for extraction/synthesis, OpenAI embeddings for retrieval, both swappable via environment variables — a provider change is a config change, not a code change.

## 3. What's new in this delivery

### 3.1 Knowledge graph: real entity identity + AI-generated ontology
Graph nodes previously conflated "what type of thing is this" with "which specific thing is this" — distinct TEC designs collapsed onto one node, and the graph rendered as disconnected clusters with no visible relationships. Now: `entity_id` (a deterministic hash of subject + geometry) identifies *which* design/run/material a node is; `entity_type` is classified against an ontology the AI generates once from the simulation contract itself, not hardcoded. Real entity-to-entity links (`SimulationRun → TECDesign`, `SimulationRun → Material`) mean the graph now renders as an actually-connected structure — every simulation run visibly links to the design it evaluated and the material it used.

### 3.2 Chinese-language support
The knowledge graph's search previously only matched ASCII text — a pure-Chinese question could silently match nothing even against directly relevant data. Fixed with Chinese word segmentation and by including the graph's Chinese-language labels in the searchable text. Literature retrieval also had a real, measured problem: because the internal corpus is entirely English papers, a Chinese question forced PaperQA2's own search agent into 2–3 extra English-reformulation rounds, adding several minutes per query — a translation step now runs first, cutting this back to one round.

### 3.3 Real comparison/ranking queries
"Which design has the highest ΔT" used to be answered by keyword matching alone — every candidate scored similar relevance and the model had to guess which one was actually best from an unordered list, with no real comparison ever computed. The system now detects comparative language and genuinely sorts candidates by the real field value, both in `/ask` and in the priors pipeline.

### 3.4 Web approval panel
Knowledge-graph candidates (from simulation runs or literature seeding) can now be reviewed and approved through a web UI (`/app` → 知识候选审批), not just the command-line script — both share one approval implementation so they can never drift apart on what "approve" actually does. The human-approval gate remains the **only** write path into the graph in both.

### 3.5 Confidence calibrated against real data, not fixed constants
Knowledge-graph confidence scores were a flat 0.7/0.4 regardless of how good a match actually was. A real leave-one-out cross-validation sweep of the deployed surrogate model — not an assumption — found the tolerance used to judge "consistent" was badly mismatched per field: some scalar outputs generalize to under 1.5% error, others to as much as 219% (90th percentile) against the same training data. Confidence is now a continuous, per-field score calibrated to that real error data, and the concrete numbers behind a given score (actual deviation, which benchmark case) now travel with the record instead of being collapsed into one opaque number.

## 4. Quality and security

**419 automated tests**, covering unit, route, schema, and regression fixture level for every endpoint; run on every change.

A **two-round adversarial security review** was carried out across the whole service — abnormal input, boundary conditions, network failure, concurrency, security, and user misuse — attacking every endpoint with real, reproduced payloads rather than a code-reading pass alone. **Three real bugs were found and fixed**, each reproduced live before the fix and re-attacked against the rebuilt Docker deployment after:

1. A crafted run ID could pollute the filesystem with control characters and crash a connection instead of returning a clean error (medium severity).
2. A compromised or spoofed literature-search API response could redirect an internal file download to an internal/cloud-metadata address (SSRF-via-redirect, medium-high severity).
3. A single request could fan out into an unbounded number of billed API calls with no cap (low-medium severity, cost/availability risk).

All three are fixed, tested, and verified against the live deployment. Full write-up, including confirmed-safe areas and items flagged for a deployment decision (network exposure, one remaining unbounded field, surrogate-model reliability for three scalar outputs): **`ADVERSARIAL_REVIEW_REPORT.md`**.

**Real-call verification** (not just unit tests) was carried out for every endpoint after all fixes landed — priors, validate, and ask were each run twice against the live deployed container with real LLM calls, in both Chinese and English. This surfaced two more real, fixed issues that unit tests alone had missed:

- **A production regression in the ranking feature**: `/sciencerag/priors` had silently stopped attaching rank annotations ("rank 1 of 5", etc.) to knowledge-graph results for comparison questions ("what's the optimal current" and similar). The bug was in the endpoint's actual production code path, not the ranking logic itself — existing unit tests only ever exercised the ranking function directly, never through the real request pipeline, so the regression shipped invisibly. Found via a real Chinese-language call, fixed, and a new test now exercises the real entry point specifically so this class of drift can't hide again.
- **PDF rendering of Chinese text**: an earlier fix for Chinese labels rendering as blank boxes in generated PDF reports had a follow-on issue — the bullet-point and separator characters used elsewhere in the same document rendered as an unrelated, incorrect character under the new font (not just missing — actively wrong). Fixed by switching those specific characters to plain-ASCII equivalents confirmed safe under the same font.

Both are fixed, tested, and confirmed against the live deployment.

## 5. Deployment

Two-stage Docker build (Node builds the frontend; the final image is Python-only, no Node runtime shipped). Runtime state — knowledge graph, reports, audit log, retrieval index — persists via bind mounts/volumes, not baked into the image; no secrets are committed or baked in (`.env` is git- and docker-ignored).

```bash
docker compose up --build
```

Currently running and verified end-to-end against the rebuilt image with this delivery's changes.

## 6. Known limitations (for a deployment decision, not blocking)

- **No authentication, bound to all network interfaces.** Fine on a trusted personal machine; if this ever runs on a shared network, recommend binding to localhost only or adding auth.
- **Surrogate-model reliability** for three scalar outputs (total resistance, optimal current, max heat dissipation) is real but weak (up to 219% p90 error against the current 31-sample training set) — a signal for the modeling side, not something this service can fix on its own.
- **Corpus** currently 171 papers (6 committed to git as seed data, the rest managed locally) — coverage will keep growing as more designs get validated and seeded.

## 7. Repository map

```
sciencerag/          # backend: priors, validate, report, ask, kg_approval
frontend/            # React/Vite UI (/app)
tec_surrogate/       # the trained ML surrogate model this service validates against
corpus/papers/       # literature corpus PaperQA2 indexes
data/                # runtime state (knowledge graph, reports, audit log) — not in git
tests/               # 419 tests
docs/spec/           # full design spec
scripts/             # ontology generation, KG seeding, approval CLI, calibration sweeps
ADVERSARIAL_REVIEW_REPORT.md   # full security review write-up
```

## 8. Commit log

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
