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
#
# The leading (?<![A-Za-z]) blocks a match starting right after a letter —
# this TEC corpus's evidence text is full of "Bi2Te3"/"Sb2Te3"/"ZT2.5"-style
# chemical-formula and figure-of-merit notation, and without this guard
# those subscript/inline digits (e.g. the "2" and "3" in "Bi2Te3") get
# scooped up as if they were free-floating numbers — silently widening the
# evidence-number pool and letting an unrelated prior number "2" or "3"
# pass the groundedness check just because the material's name happens to
# contain that digit. Deliberately NOT also blocking a letter immediately
# AFTER the number (no trailing lookahead) — "60um"/"200uV" (number
# directly abutting its unit, no space) are common and must still match.
# Known remaining gap: a number written with no space before it either,
# e.g. "T300K" or "ZT2.5" itself as a genuine reported value rather than a
# figure-of-merit label, would also be blocked — accepted as a rarer
# failure mode than the chemical-formula contamination this fixes.
_COMBINED_RE = re.compile(
    r"(?<![A-Za-z])(?:"
    rf"(?P<plusminus>(?P<pm_base>{_NUM})\s*(?:±|\+/-|\+-)\s*(?P<pm_dev>{_NUM}))"
    rf"|(?P<range>(?P<rg_lo>{_NUM})\s*(?:-|–|—|~|to)\s*(?P<rg_hi>{_NUM}))"
    rf"|(?P<percent>(?P<pct>{_NUM})\s*%)"
    rf"|(?P<plain>{_NUM})"
    r")"
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
      - scaling_relationship: x/y/direction are names/labels, not numbers,
        but `functional_form`/`validity_range` are free text and can smuggle
        in a fabricated equation or bound (e.g. "valid for L > 5 mm") same as
        caution's prose fields below — scanned the same way.
      - candidate_config: numeric `parameters` + `reported_performance` values
      - caution: `statement`/`applicability_scope` ARE prose, but prose can
        still smuggle in a specific fabricated number (e.g. a `statement`
        asserting "insufficient for cooling requiring ~90 K ΔT" with no 90
        anywhere in the cited evidence) — found via a real production
        sample where exactly this happened and slipped past this check
        because it was originally treated as pure prose with nothing to
        verify. Parsed with extract_numbers_from_text like `prior.notes`,
        not read as fixed fields, since there's no structured numeric slot
        to read directly.

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
        if value.functional_form:
            numbers.extend(extract_numbers_from_text(value.functional_form))
        if value.validity_range:
            numbers.extend(extract_numbers_from_text(value.validity_range))
    elif isinstance(value, CandidateConfigValue):
        numbers.extend(_numeric_dict_values(value.parameters))
        numbers.extend(_numeric_dict_values(value.reported_performance))
    elif isinstance(value, CautionValue):
        numbers.extend(extract_numbers_from_text(value.statement))
        if value.applicability_scope:
            numbers.extend(extract_numbers_from_text(value.applicability_scope))

    if prior.notes:
        numbers.extend(extract_numbers_from_text(prior.notes))

    return numbers


# Widened from a literal 1% because the rounding case this is meant to
# tolerate — evidence states 1.83, the prior (reasonably) rounds it to
# 1.8 — is itself a ~1.64% relative difference: |1.83-1.8|/1.83 ≈ 0.0164.
# A strict 1% would reject that exact case. 2% keeps the same "tolerate
# rounding, not real numeric drift" intent with enough headroom for it.
NUMERIC_MATCH_RELATIVE_TOLERANCE = 0.02


def numbers_match(a: float, b: float) -> bool:
    """True if `a` (a number the prior asserts) matches `b` (a number found
    in evidence text): exact equality, or within NUMERIC_MATCH_RELATIVE_TOLERANCE
    relative error — tolerates the prior rounding a number when copying it
    out of evidence (e.g. evidence's 1.83 -> prior's 1.8), not real drift.

    Deliberately NOT unit-aware (v1 scope — see module docstring): 20 and
    0.02 never match on their own, even when one is "20 µm" and the other
    is "0.02 mm" (the same physical length). Unit conversion is a
    deliberately deferred non-goal, not an oversight — see extract.py's
    prompt rule to preserve evidence's original units instead.
    """
    if a == b:
        return True
    if b == 0:
        return a == 0
    return abs(a - b) / abs(b) <= NUMERIC_MATCH_RELATIVE_TOLERANCE


def find_unmatched_numbers(prior_numbers: list[float], evidence_numbers: list[float]) -> list[float]:
    """Which numbers from `prior_numbers` have no match anywhere in
    `evidence_numbers`. Empty result means every number the prior asserts
    is grounded somewhere in the evidence."""
    return [n for n in prior_numbers if not any(numbers_match(n, e) for e in evidence_numbers)]
