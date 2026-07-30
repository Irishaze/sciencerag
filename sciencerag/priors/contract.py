"""Loader for sim_params.json — the simulation-side parameter contract
(spec: docs/spec/sync_to_claude_code.md).

sim_params.json is the single source of truth for which parameter names a
Prior's `field`/`related_fields` may reference: the COMSOL simulation side
owns these names, not ScienceRAG. Only `geometry_free` is `prior_target:
true` — material/operating_condition/derived/numerical_setting are fixed
or out of scope for priors (see the contract's own `rules` section).
"""

import json
from pathlib import Path
from typing import Any

CONTRACT_PATH = Path(__file__).resolve().parent / "sim_params.json"


def _load_contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text())


CONTRACT: dict[str, Any] = _load_contract()
GEOMETRY_FREE_PARAMS: list[dict[str, str]] = CONTRACT["geometry_free"]
GEOMETRY_FREE_NAMES: frozenset[str] = frozenset(p["name"] for p in GEOMETRY_FREE_PARAMS)
