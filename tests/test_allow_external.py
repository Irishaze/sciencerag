"""M1-15 / spec §9 OQ#1: allow_external is an explicit, documented no-op in
M1 — external retrieval is deferred to M6. Pure-function tests against
_add_external_note; no PaperQA2/API calls.
"""

from sciencerag.priors.models import Coverage, PriorsResponse
from sciencerag.priors.retrieval import _add_external_note


def _response(gaps: list[str] | None = None) -> PriorsResponse:
    return PriorsResponse(
        priors=[],
        coverage=Coverage(internal_hits=0, external_hits=0, gaps=gaps or []),
        trace_id="tr_test",
    )


def test_allow_external_true_appends_not_implemented_note():
    response = _add_external_note(_response(), allow_external=True)
    assert len(response.coverage.gaps) == 1
    assert "allow_external" in response.coverage.gaps[0]
    assert "M6" in response.coverage.gaps[0]


def test_allow_external_false_leaves_gaps_unchanged():
    response = _add_external_note(_response(gaps=["some other gap"]), allow_external=False)
    assert response.coverage.gaps == ["some other gap"]


def test_allow_external_never_changes_external_hits():
    """external_hits must stay 0 in M1 regardless of the flag — it's a
    request-acknowledgment note, not a behavior change."""
    response = _add_external_note(_response(), allow_external=True)
    assert response.coverage.external_hits == 0
