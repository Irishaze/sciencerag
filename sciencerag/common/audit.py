"""Append-only JSONL audit log shared by every sciencerag endpoint (spec §3.5/§8).

Each call logs trace_id + endpoint + request + evidence + output, so any
response can be reconstructed and audited later. The log path is a mutable
module attribute (not read once at import time) so tests can point it at a
tmp file via monkeypatch.
"""

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

AUDIT_LOG_PATH = Path("logs/audit.jsonl")

# Every sciencerag route function is a plain `def`, so FastAPI/Starlette runs
# each request in a real worker thread (not cooperatively) — concurrent
# requests across ANY endpoint call this at the same time. A single f.write()
# of a large entry (this app embeds the full request+output dumps; nothing
# caps ValidateRequest.priors/design_parameters/scalar_results sizes) is only
# guaranteed atomic by the OS up to a platform-specific limit; past that, two
# interleaved writes can splice two entries into one line that no longer
# parses as JSON, corrupting logs/audit.jsonl for every downstream reader
# (regression tooling, loo_scalar_error_sweep.py-style sweeps, a human
# tailing it). This only serializes writers within one process — a
# multi-worker deployment (uvicorn --workers > 1) would still need a real
# file lock (fcntl/flock), not covered here since nothing in this repo's
# deployment config runs more than one worker today.
_LOG_LOCK = threading.Lock()


def log_audit_entry(
    *,
    trace_id: str,
    endpoint: str,
    request: dict[str, Any],
    evidence: Any,
    output: dict[str, Any],
    model_config: dict[str, Any] | None = None,
    elapsed_s: float | None = None,
    filtered_material_count: int = 0,
) -> None:
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "trace_id": trace_id,
        "endpoint": endpoint,
        "timestamp": datetime.now(UTC).isoformat(),
        "request": request,
        "evidence": evidence,
        "output": output,
        "model_config": model_config,
        # spec §9 latency target (soft guard, see router.py's LATENCY_TARGET_SECONDS):
        # recorded for every call so latency regressions are auditable after
        # the fact, not just when a warning happens to be watched live.
        "elapsed_s": elapsed_s,
        # spec §3.6: material is fixed (Bi2Te3, prior_target=false) — if the
        # LLM emitted a material_property finding anyway, it's dropped
        # before becoming a Prior (never shown in coverage.gaps, since it's
        # not a coverage shortfall) but logged here so it stays auditable.
        "filtered_material_count": filtered_material_count,
    }
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    with _LOG_LOCK, AUDIT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line)
