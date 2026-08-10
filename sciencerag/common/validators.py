"""Shared Pydantic field validators (spec-wide, not tied to one endpoint)."""

from __future__ import annotations

import math

_UNSAFE_PATH_CHARS = ("/", "\\")


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
    an unrelated real run's files."""
    if not value:
        raise ValueError("run_id must not be empty")
    if any(c in value for c in _UNSAFE_PATH_CHARS):
        raise ValueError(f"run_id must not contain {'/'.join(_UNSAFE_PATH_CHARS)!r}: {value!r}")
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
