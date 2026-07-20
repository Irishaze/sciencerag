"""trace_id generation shared by every sciencerag endpoint (spec §8)."""

import uuid
from datetime import UTC, datetime


def new_trace_id(prefix: str = "tr") -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"{prefix}_{timestamp}_{suffix}"
