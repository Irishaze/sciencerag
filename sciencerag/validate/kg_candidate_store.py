"""On-disk queue for KG candidates awaiting human approval.

/sciencerag/validate returns update_package.kg_candidates in its response
body only — nothing persists them. Previously the only way to run
scripts/approve_kg_candidates.py was to manually copy that JSON out of the
response into a file yourself. This module closes that gap by writing every
non-empty kg_candidates batch to data/kg_candidates/pending/ as soon as
validate produces it, so approval is just picking a file that's already
there. The approval step itself is untouched (spec §6.3: candidate ->
approval -> registration, still human-gated, still CLI-only per spec §7)."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from sciencerag.validate.models import KGCandidate

PENDING_DIR = Path("data/kg_candidates/pending")
ARCHIVE_DIR = Path("data/kg_candidates/archive")


def _safe_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace(":", "").replace("+00:00", "Z")


def store_pending_candidates(run_id: str, candidates: list[KGCandidate]) -> Path | None:
    if not candidates:
        return None
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    path = PENDING_DIR / f"{run_id}_{_safe_timestamp()}.json"
    # Write-to-temp-then-rename rather than a direct write_text(): confirmed
    # live that a reader (load_pending, called by
    # scripts/approve_kg_candidates.py) hitting a file mid-write sees a
    # truncated JSON document and raises an uncaught JSONDecodeError —
    # list_pending() already tolerates this (wrapped in try/except), but
    # load_pending() didn't. os.replace() is atomic on POSIX, so a reader
    # always sees either the fully-old (nonexistent) or fully-new file,
    # matching the same pattern already used for data/kg/graph.json and
    # data/reports/*.json.
    tmp_path = path.with_suffix(f".{os.getpid()}.tmp")
    tmp_path.write_text(
        json.dumps([c.model_dump() for c in candidates], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp_path, path)
    return path


def list_pending() -> list[dict]:
    """Newest first, each with a candidate count so --list-pending doesn't
    need to open every file to be useful."""
    if not PENDING_DIR.exists():
        return []
    entries = []
    for path in sorted(PENDING_DIR.glob("*.json"), reverse=True):
        try:
            count = len(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            count = 0
        entries.append({"stem": path.stem, "count": count})
    return entries


def load_pending(stem: str) -> list[KGCandidate]:
    path = PENDING_DIR / f"{stem}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [KGCandidate.model_validate(item) for item in data]


def archive_pending(stem: str) -> None:
    """Moves a processed pending file out of the pending queue so
    --list-pending doesn't keep offering it back up after it's been acted
    on. Candidates the operator chose to skip aren't lost — they're just no
    longer in the default queue; the archived file still has them.

    Idempotent by design, not just by intent: sciencerag.kg_approval's web
    panel calls this at the end of every approve request regardless of
    which indices were approved, so two concurrent approvals against the
    same stem (two operators, or a double-click) both reach this. The
    `src.exists()` check followed by `src.rename()` is not atomic — a real
    concurrency test (2 threads x 2000 trials) confirmed the second caller's
    rename() raising an uncaught FileNotFoundError 99%+ of the time once the
    first caller's rename had already completed. That crash reached the
    caller as a raw 500 even though their own candidate approvals had
    already succeeded (add_triple already committed before archive_pending
    runs) — so the response looked like a failure for a request that had
    actually already done its real work. Treat "already archived by someone
    else" as success, not an error."""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    src = PENDING_DIR / f"{stem}.json"
    try:
        src.rename(ARCHIVE_DIR / f"{stem}.json")
    except FileNotFoundError:
        pass
