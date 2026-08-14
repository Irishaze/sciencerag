"""Shared Pydantic field validators (spec-wide, not tied to one endpoint)."""

from __future__ import annotations

import math

_UNSAFE_PATH_CHARS = ("/", "\\", "\x00", '"', "\r", "\n")


def reject_path_unsafe_id(value: str) -> str:
    """For any `run_id`-shaped field that ends up embedded in a filename
    (sciencerag/report/store.py, sciencerag/validate/kg_candidate_store.py
    both do `f"{run_id}_{timestamp}"` with zero sanitization). A run_id
    containing "/" or "\\" lets the resulting path escape the intended
    directory — confirmed via a real adversarial test: run_id=
    "../kg/marker" made POST /sciencerag/report write a file onto the host
    filesystem outside data/reports/ entirely (data/ is bind-mounted in
    docker-compose.yml, so this reaches real host paths, not just the
    container). Rejecting outright (422) rather than silently
    stripping/sanitizing, since a mangled run_id could still collide with
    an unrelated real run's files.

    "\\x00" is here for a different reason than "/"/"\\": it can't escape a
    directory (it isn't a path separator), but confirmed via a real
    adversarial test that sciencerag.kg_approval's `stem` path parameter
    (validated with this same function, called directly rather than as a
    Pydantic field_validator) reaches pathlib with a null byte still in it,
    and pathlib raises ValueError("embedded null byte") from deep inside
    load_pending — uncaught, so it surfaces as a raw 500 instead of the
    router's typed 404/400. Reject early instead.

    '"'/"\\r"/"\\n" added 2026-08-15 after a full adversarial-review pass:
    a run_id of 'inject"x\\r\\nX-Injected-Header: pwned' passed the old
    check (none of those three chars were blocked), got written into two
    real filenames on disk verbatim (data/reports/*.json and *.md — control
    characters and a literal quote, live in the filesystem), and later,
    fetching that report's /pdf endpoint reached
    report/router.py's `Content-Disposition: inline; filename="{stem}.pdf"`
    with the same raw stem — h11 (uvicorn's HTTP layer) correctly refused
    to emit the resulting malformed header (so real HTTP response-splitting
    was never actually reachable), but it did so by raising
    LocalProtocolError deep inside the ASGI send path, after this project's
    own try/except in get_report_pdf had already run and returned
    successfully — the connection then dropped with an ungraceful "empty
    reply", not the typed 4xx/5xx JSON error response spec §8 requires
    ("always return typed, schema-valid JSON"). Rejecting these three
    characters at the same API boundary as the others closes both the
    filename pollution and the ungraceful-crash paths at once, the same
    "reject early, not deep in a library" principle as \\x00 above."""
    if not value:
        raise ValueError("run_id must not be empty")
    if any(c in value for c in _UNSAFE_PATH_CHARS):
        raise ValueError(f"run_id must not contain {_UNSAFE_PATH_CHARS!r}: {value!r}")
    return value


def reject_non_finite_values(values: dict[str, float]) -> dict[str, float]:
    """Pydantic's `float` type accepts NaN/Infinity as valid IEEE754 floats
    by default — confirmed via a real adversarial test: a NaN scalar_result
    flowed unblocked through /sciencerag/validate into a KG candidate, got
    auto-queued for approval, and `add_triple` happily persisted it as a
    literal (non-standard, RFC 8259-violating) `NaN` token in
    data/kg/graph.json — a value the frontend's `.toFixed()` calls and any
    strict JSON consumer would then choke on. Reject these at the API
    boundary instead of letting a non-finite number travel all the way to
    permanent storage."""
    bad = {k: v for k, v in values.items() if not math.isfinite(v)}
    if bad:
        raise ValueError(f"non-finite values are not allowed: {bad}")
    return values


def reject_non_finite_list(values: list[float] | None) -> list[float] | None:
    """List-shaped counterpart to reject_non_finite_values, for fields like
    ValidateRequest.latent_state that aren't a dict. Confirmed via a real
    adversarial test: a NaN slipped into latent_state didn't crash or
    produce invalid JSON (FastAPI/Pydantic serialize NaN/Infinity as
    `null`) — it silently produced a *wrong, confident* answer instead.
    check_ood's np.searchsorted(training_scores, mahalanobis) treats a NaN
    distance as larger than every real training score, so the request came
    back as severity="blocking" with training_distance_percentile=100.0 —
    indistinguishable from a genuinely, validly extreme design, when the
    true situation is "this input is malformed, we can't score it at all"
    (exactly the case the dimension-mismatch branch right above already
    handles honestly, by reporting an error instead of a fabricated
    score). Reject at the API boundary instead of silently computing a
    plausible-looking wrong severity."""
    if values is None:
        return values
    bad = [v for v in values if not math.isfinite(v)]
    if bad:
        raise ValueError(f"non-finite values are not allowed: {bad}")
    return values
