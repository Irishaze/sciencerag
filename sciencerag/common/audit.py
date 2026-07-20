"""Append-only JSONL audit log shared by every sciencerag endpoint (spec §3.5/§8).

Each call logs trace_id + endpoint + request + evidence + output, so any
response can be reconstructed and audited later. The log path is a mutable
module attribute (not read once at import time) so tests can point it at a
tmp file via monkeypatch.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

AUDIT_LOG_PATH = Path("logs/audit.jsonl")


def log_audit_entry(
    *,
    trace_id: str,
    endpoint: str,
    request: dict[str, Any],
    evidence: Any,
    output: dict[str, Any],
    model_config: dict[str, Any] | None = None,
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
    }
    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
