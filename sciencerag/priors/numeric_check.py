"""Phase B1: numeric extraction for prior/evidence groundedness checking.

Two extractors feed the matching step (Phase B2/B3):
  - extract_numbers(prior): every number a Prior actually asserts, pulled
    from its value's numeric fields (never from free-text fields other than
    `notes` — see the per-kind breakdown below).
  - extract_numbers_from_text(text): every number mentioned in a block of
    evidence text, handling the messy ways numbers show up in real prose
    (thousands separators, scientific notation, percentages, ranges, ±).

Deliberately NOT unit-aware (v1 scope, see the numeric_check plan): a number
is a number regardless of what unit it's attached to. Unit consistency is a
separate, harder problem left for later.
"""

import re

from sciencerag.priors.models import (
    CandidateConfigValue,
    CautionValue,
    MaterialPropertyValue,
    ParameterRangeValue,
    Prior,
    ScalingRelationshipValue,
)

# A single numeric token: either thousands-comma-grouped (1,200 or 1,200.5)
# or a plain/scientific number (20, 0.06, -40, 1.5e-3). Comma form is tried
# first so "1,200" isn't split into "1" + range-like leftovers.
_NUM = r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"

# Compound patterns are tried before the plain-number fallback (and before
# each other, most-specific-consuming-most-text first) so e.g. "20-200" is
# read as a range, not as "20" followed by a stray "-200".
_COMBINED_RE = re.compile(
    rf"(?P<plusminus>(?P<pm_base>{_NUM})\s*(?:±|\+/-|\+-)\s*(?P<pm_dev>{_NUM}))"
    rf"|(?P<range>(?P<rg_lo>{_NUM})\s*(?:-|–|—|~|to)\s*(?P<rg_hi>{_NUM}))"
    rf"|(?P<percent>(?P<pct>{_NUM})\s*%)"
    rf"|(?P<plain>{_NUM})"
)


def _to_float(raw: str) -> float:
    return float(raw.replace(",", ""))


def extract_numbers_from_text(text: str) -> list[float]:
    """Every number mentioned in `text`, including numbers implied by
    compound expressions:
      - "1,200" -> 1200.0 (thousands separator)
      - "1.5e-3" -> 0.0015 (scientific notation)
      - "15%" -> 15.0 AND 0.15 (both forms recorded — the prior might state
        either, and we don't know which one until we try to match)
      - "20-200" / "20–200" / "20 to 200" -> 20.0 and 200.0 (range,
        both endpoints)
      - "60±5" / "60+/-5" -> 60.0 and 5.0 (base value and deviation)
    """
    numbers: list[float] = []
    for m in _COMBINED_RE.finditer(text):
        if m.group("plusminus"):
            numbers.append(_to_float(m.group("pm_base")))
            numbers.append(_to_float(m.group("pm_dev")))
        elif m.group("range"):
            numbers.append(_to_float(m.group("rg_lo")))
            numbers.append(_to_float(m.group("rg_hi")))
        elif m.group("percent"):
            pct = _to_float(m.group("pct"))
            numbers.append(pct)
            numbers.append(pct / 100)
        else:
            numbers.append(_to_float(m.group("plain")))
    return numbers


def _numeric_dict_values(d: dict) -> list[float]:
    return [float(v) for v in d.values() if isinstance(v, int | float) and not isinstance(v, bool)]


def extract_numbers(prior: Prior) -> list[float]:
    """Every number `prior` actually asserts, per its kind:
      - parameter_range: min/max/typical + any numeric `conditions` values
      - material_property: magnitude + any numeric `conditions` values
      - scaling_relationship: none (x/y/direction/functional_form/
        validity_range are all names/labels/prose, not asserted numbers)
      - candidate_config: numeric `parameters` + `reported_performance` values
      - caution: none (statement/applicability_scope are prose)

    Plus every number in `prior.notes`, parsed the same way evidence text
    is (free text, so it needs the full extract_numbers_from_text handling
    rather than a direct field read).
    """
    numbers: list[float] = []
    value = prior.value

    if isinstance(value, ParameterRangeValue):
        numbers.extend(v for v in (value.min, value.max, value.typical) if v is not None)
        numbers.extend(_numeric_dict_values(value.conditions))
    elif isinstance(value, MaterialPropertyValue):
        if value.magnitude is not None:
            numbers.append(value.magnitude)
        numbers.extend(_numeric_dict_values(value.conditions))
    elif isinstance(value, ScalingRelationshipValue):
        pass
    elif isinstance(value, CandidateConfigValue):
        numbers.extend(_numeric_dict_values(value.parameters))
        numbers.extend(_numeric_dict_values(value.reported_performance))
    elif isinstance(value, CautionValue):
        pass

    if prior.notes:
        numbers.extend(extract_numbers_from_text(prior.notes))

    return numbers
