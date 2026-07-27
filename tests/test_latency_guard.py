"""M1-19: soft latency guard for /sciencerag/priors (spec §9: priors ≤ 30s).

Soft means: a slow response still returns 200 with the real result — we
never fail a request just because it took too long. We only log a warning
and record elapsed_s in the audit log. Mocks out real retrieval and fakes
elapsed time via monkeypatched time.monotonic() so this stays fast/free.
"""

import logging

from fastapi.testclient import TestClient

from sciencerag.app import app
from sciencerag.priors import router as priors_router
from sciencerag.priors.models import Coverage, Prior, PriorsResponse, SourcePaper

client = TestClient(app)


def _fake_priors_response(query: str, allow_external: bool = False) -> PriorsResponse:
    return PriorsResponse(
        priors=[
            Prior(
                prior_id="pr_fake_0001",
                kind="parameter_range",
                field="general_finding",
                value={"summary": "fake"},
                confidence=0.8,
                sources=[SourcePaper(doi="10.0000/fake", span="pages 1-2")],
            )
        ],
        coverage=Coverage(internal_hits=1, external_hits=0, gaps=[]),
        trace_id="tr_fake_latency_test",
    )


def _fake_clock(times: list[float]):
    it = iter(times)
    return lambda: next(it)


def test_slow_response_still_succeeds_and_logs_warning(monkeypatch, caplog):
    monkeypatch.setattr(priors_router.retrieval, "build_priors_response", _fake_priors_response)
    # t0=0.0, t1=35.0 -> elapsed 35s, over the 30s target. Patches router's
    # own _now() seam, NOT the global time module (see router.py's comment
    # on _now — httpx/starlette internals also call time.monotonic()).
    monkeypatch.setattr(priors_router, "_now", _fake_clock([0.0, 35.0]))

    with caplog.at_level(logging.WARNING, logger="sciencerag.priors.router"):
        resp = client.post("/sciencerag/priors", json={"query": "slow query"})

    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert any("latency target" in r.message for r in caplog.records)


def test_fast_response_does_not_log_warning(monkeypatch, caplog):
    monkeypatch.setattr(priors_router.retrieval, "build_priors_response", _fake_priors_response)
    # t0=0.0, t1=2.0 -> elapsed 2s, well under the 30s target.
    monkeypatch.setattr(priors_router, "_now", _fake_clock([0.0, 2.0]))

    with caplog.at_level(logging.WARNING, logger="sciencerag.priors.router"):
        resp = client.post("/sciencerag/priors", json={"query": "fast query"})

    assert resp.status_code == 200
    assert not any("latency target" in r.message for r in caplog.records)
