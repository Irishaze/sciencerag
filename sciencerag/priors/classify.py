"""Heuristic classification of evidence text into the 5 `kind` values (spec §3.3).

Deterministic keyword rules, not an LLM call — keeps this cheap and testable
(same reasoning as picking agent_type="fake" over an LLM-driven loop: M1's
corpus is small and the regression tests need stable, repeatable output).
Order matters: the first matching rule wins.
"""

import re

Kind = str  # Literal is defined in models.py; kept loose here to avoid a cycle.

# (pattern, kind, field slug) — checked in order, first match wins.
_RULES: list[tuple[str, Kind, str]] = [
    (r"seebeck coefficient", "material_property", "seebeck_coefficient"),
    (r"electrical resistivity", "material_property", "electrical_resistivity"),
    (r"thermal conductivity", "material_property", "thermal_conductivity"),
    (r"peltier coefficient", "material_property", "peltier_coefficient"),
    (r"\bconvex function\b|\bproportional to\b|\bfunction of\b|\bcorrelat", "scaling_relationship", "cop_relationship"),
    (
        r"\bhowever\b|\blimitation\b|\bdrops to zero\b|\bshould be noted\b|\btrade-?off\b|\bcaution\b",
        "caution",
        "operating_caveat",
    ),
    (
        r"\bpulse-frequency modulation\b|\bduty cycle\b|\bdriving method\b|\bconfiguration\b|\bgeometry\b",
        "candidate_config",
        "design_configuration",
    ),
    (
        r"\boptimal (voltage|value|current)\b|\brange\b|\bbetween \d|\d+(\.\d+)?\s*(v|w|ω|ohm|k|°c)\b",
        "parameter_range",
        "operating_parameter",
    ),
]

DEFAULT_KIND: Kind = "parameter_range"
DEFAULT_FIELD = "general_finding"


def classify(text: str) -> tuple[Kind, str]:
    """Return (kind, field) for a piece of evidence text."""
    lowered = text.lower()
    for pattern, kind, field in _RULES:
        if re.search(pattern, lowered):
            return kind, field
    return DEFAULT_KIND, DEFAULT_FIELD
