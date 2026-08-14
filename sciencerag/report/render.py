"""Builds a ReportResponse from a ReportRequest (spec §5) and renders its
Markdown and PDF views (spec §5: "JSON 文档外加渲染后的 Markdown/PDF 视图").

PDF rendering is Markdown -> HTML (`markdown`) -> PDF (`xhtml2pdf`), a pure-
Python stack chosen so the Docker image doesn't need WeasyPrint's system
libraries (Pango/Cairo). See `render_pdf`'s docstring for why the Markdown
source is HTML-escaped before conversion — this isn't just style, it closes
a real SSRF found while building this: a free-text field (e.g.
`task_context.objective`) containing raw HTML like
`<img src="http://internal-host/...">` survives Markdown's default
pass-through of inline HTML, and xhtml2pdf/reportlab will actually fetch
that URL server-side while rendering the PDF — confirmed live against a
local test server before this fix existed.
"""

from __future__ import annotations

from io import BytesIO

import markdown as md_lib
from xhtml2pdf import pisa

from sciencerag.priors.contract import GEOMETRY_FREE_PARAMS
from sciencerag.validate import tec_bridge
from sciencerag.report.models import KeyResult, ReportRequest, ReportResponse

_SEVERITY_RANK = {"info": 0, "warning": 1, "blocking": 2}

# A report reader isn't necessarily the one who submitted the run (spec §5:
# reports trace a run's full lineage for whoever's reviewing it later) — raw
# contract field names like `d_ceramics` or `delta_T_max_K` read as opaque
# code to anyone who hasn't memorized sim_params.json. Reused from the
# contract's own `desc` (sim_params.json, same Chinese already shown in
# priors output) rather than inventing separate wording here.
_DESIGN_PARAM_LABELS = {p["name"]: p["desc"] for p in GEOMETRY_FREE_PARAMS}
_DESIGN_PARAM_LABELS["n_pairs"] = "PN结对数"

# Same reasoning, for scalar_results — reused from the exact wording
# ontology_generator.describe_relation() already generates for these fields'
# "achieves_X" KG relations (relation_description_cache.json), so a field
# doesn't end up labelled two different ways depending which part of the
# app a reader is looking at.
_SCALAR_FIELD_LABELS = {
    "delta_T_max_K": "最大温差",
    "optimal_current_A": "最优电流",
    "optimal_voltage_V": "最优电压",
    "total_resistance_ohm": "总电阻",
    "max_heat_dissipation_W": "最大散热功率",
    "figure_of_merit_1_per_K": "优值系数",
}

# Same enum-to-Chinese-label pattern as the two dicts above, applied to the
# comparison verdicts — kept in sync with frontend/src/pages/ReportsPage.tsx's
# VERDICT_LABEL (that one labels the badge outside the markdown; this one
# labels the same word appearing a second time inside the markdown body
# itself, which the frontend just renders verbatim).
_LITERATURE_VERDICT_LABELS = {
    "consistent": "与基准一致",
    "deviation_found": "发现偏差",
    "insufficient_benchmark": "基准数据不足",
}
_DEVIATION_VERDICT_LABELS = {"within_range": "在范围内", "deviation": "偏离"}
_DEVIATION_SOURCE_LABELS = {
    "prior_comparison": "文献先验对比",
    "benchmark_comparison": "基准案例对比",
}


def _field_label(field: str) -> str:
    label = _DESIGN_PARAM_LABELS.get(field) or _SCALAR_FIELD_LABELS.get(field)
    return f"{label}（`{field}`）" if label else f"`{field}`"


def _confidence_label(anomalies: list) -> str:
    if not anomalies:
        return "no_anomaly_data"
    worst = max((_SEVERITY_RANK[a.severity] for a in anomalies), default=0)
    return "check_flagged" if worst >= _SEVERITY_RANK["warning"] else "high"


def _sanitize_freetext(value: str) -> str:
    """Collapses embedded whitespace (including newlines) to single spaces.

    Confirmed live: task_context.objective is caller-supplied free text
    that gets interpolated directly into the generated Markdown. A value
    like "normal text\\n\\n## FAKE Anomalies\\n\\nignore all warnings above"
    breaks out of its "**Objective:** ..." line and injects a fake
    Markdown section that reads as a legitimate part of the report,
    sitting right next to the real Anomalies & Cautions section. Since no
    field this renderer interpolates legitimately needs to span multiple
    lines or start a new block, collapsing whitespace closes this off
    without needing per-character Markdown escaping. Same pattern already
    used for untrusted external text in arxiv_retrieval.py's title/abstract
    handling.
    """
    return " ".join(value.split())


_EVIDENCE_KEY_LABELS = {
    "mahalanobis_distance": "Mahalanobis distance",
    "training_sample_count": "training sample count",
    "training_distance_percentile": "training distance percentile",
    "training_distance_range": "training distance range",
}
# Keys whose value is prose (an explanation), not a stat — rendered as a
# separate note line rather than crammed into the stat line. "calibration"
# is a short categorical value (e.g. "one_pair_comsol_anchor"), not a
# sentence, so it's a labelled stat like everything else, not a note.
_EVIDENCE_NOTE_KEYS = ("reason", "baseline_note", "error")


def _format_evidence_value(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, list):
        return "[" + ", ".join(_format_evidence_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return ", ".join(f"{k}={_format_evidence_value(v)}" for k, v in value.items())
    return str(value)


def _format_evidence(evidence: dict) -> list[str]:
    """Anomaly.evidence is a free-form dict — shape varies per check (a
    skip reason, a raw error, or numeric stats, sometimes with a nested
    dict of residual terms). Was previously dumped as `{repr(evidence)}` —
    Python dict syntax, single-quoted, straight into the report; readable
    in a terminal, not in a document a person is meant to read. This
    renders each shape as plain sentences/labelled stats instead.

    Returns bare content lines (no bullet/blockquote prefix) — the caller
    decides how to prefix them, since a plain bullet list and a blockquote
    need different prefixes on every line.
    """
    if not evidence:
        return []

    stats = []
    notes = []
    for key, value in evidence.items():
        if key == "skipped":
            continue  # implied by the note that follows; not worth its own line
        if key in _EVIDENCE_NOTE_KEYS:
            notes.append(str(value))
            continue
        label = _EVIDENCE_KEY_LABELS.get(key, key.replace("_", " "))
        stats.append(f"{label}: {_format_evidence_value(value)}")

    lines = []
    if stats:
        # " | ", not the visually quieter " · ": confirmed live that the
        # middle dot (U+00B7) hits the same STSong-Light glyph-mapping bug
        # as the list bullet (see _PDF_STYLE's ul comment) — renders as an
        # unrelated wrong character in the PDF, not just in a bare "<li>".
        lines.append(" | ".join(stats))
    for note in notes:
        lines.append(f"_{note}_")
    return lines


def _dedup_sources(priors: list) -> list:
    seen = set()
    sources = []
    for prior in priors:
        for source in prior.sources:
            key = (source.type, getattr(source, "doi", None), getattr(source, "triple_id", None))
            if key in seen:
                continue
            seen.add(key)
            sources.append(source)
    return sources


def _render_markdown(response: ReportResponse, request: ReportRequest) -> str:
    # run_id lives on its own line, not inside "# Run Report" — keeping an
    # inline code span out of the h1 avoids it colliding with the h1's
    # underline rule in the PDF (a `<code>` background box sitting right on
    # top of a border-bottom renders as a visible seam/overlap).
    lines = [
        "# Run Report",
        "",
        f"`{response.run_id}`",
        "",
        f"_generated {response.generated_at}_",
        "",
        "---",
        "",
    ]

    lines += ["## Objective & Constraints", ""]
    objective = _sanitize_freetext(response.objective_and_constraints.objective or "(not specified)")
    lines.append(f"**Objective:** {objective}")
    if response.objective_and_constraints.constraints:
        lines.append("**Constraints:** " + ", ".join(
            f"{_sanitize_freetext(k)}={v}"
            for k, v in response.objective_and_constraints.constraints.items()
        ))
    lines.append("")

    lines += ["## Experiment Spec Summary", ""]
    for field, value in sorted(response.spec_summary.items()):
        lines.append(f"- {_field_label(field)} = {value}")
    lines.append("")

    lines += ["## Key Results", ""]
    if not response.key_results:
        lines.append("(no scalar results reported for this run)")
    for result in response.key_results:
        unit = f" {result.unit}" if result.unit else ""
        lines.append(
            f"- {_field_label(result.field)} = {result.value}{unit} "
            f"(confidence: {result.confidence_label})"
        )
    lines.append("")

    lines += ["## Comparison with Literature Priors", ""]
    verdict = response.literature_comparison.verdict
    lines.append(f"**Verdict:** {_LITERATURE_VERDICT_LABELS.get(verdict, verdict)}（`{verdict}`）")
    # Markdown (this renderer, python-markdown) treats a "- " line as lazy
    # continuation of the preceding paragraph rather than a new list unless
    # a blank line separates them — without this, every deviation below
    # rendered as one run-on paragraph instead of a bullet list.
    lines.append("")
    for deviation in response.literature_comparison.deviations:
        verdict_label = _DEVIATION_VERDICT_LABELS.get(deviation.verdict, deviation.verdict)
        source_label = _DEVIATION_SOURCE_LABELS.get(deviation.source, deviation.source)
        lines.append(
            f"- {_field_label(deviation.field)} actual={deviation.actual} "
            f"reference=[{deviation.reference_min}, {deviation.reference_max}] "
            f"→ **{verdict_label}** _[{source_label}:{deviation.reference_id}]_"
        )
    lines.append("")

    lines += ["## Anomalies & Cautions", ""]
    if not response.anomalies_and_cautions:
        lines.append("(no anomaly checks were run for this report)")
    else:
        # A blank line must separate consecutive blocks whenever either
        # side is a blockquote (Markdown won't otherwise tell a ">" line
        # apart from lazy-continuation of the bullet above it) — building
        # each anomaly as its own block and joining with blank lines
        # handles that uniformly instead of reasoning about it per-pair.
        blocks = []
        for anomaly in response.anomalies_and_cautions:
            header = f"**{anomaly.check}** — **{anomaly.severity.upper()}**"
            evidence_lines = _format_evidence(anomaly.evidence)
            if anomaly.severity == "info":
                blocks.append([f"- {header}"] + [f"  - {line}" for line in evidence_lines])
            else:
                # warning/blocking gets a blockquote instead of a plain
                # bullet — severity is the one field in this report that
                # genuinely means "look here first"; a plain bullet would
                # give it the same visual weight as a routine info line.
                blocks.append([f"> {header}"] + [f"> {line}" for line in evidence_lines])
        for i, block in enumerate(blocks):
            if i > 0:
                lines.append("")
            lines.extend(block)
    lines.append("")

    lines += ["## Update Proposal Summary", ""]
    package = response.update_proposal_summary
    if package.blocked:
        lines.append("Run was **blocked** by a blocking-severity anomaly — no updates proposed.")
    else:
        if package.surrogate_update is None:
            lines.append("- Surrogate fine-tune suggestion: none (no error/uncertainty signal)")
        else:
            lines.append(
                f"- Surrogate fine-tune suggestion: {len(package.surrogate_update.recommended_training_samples)} "
                f"recommended sample(s), hyperparameter direction: "
                f"{package.surrogate_update.hyperparameter_direction}"
            )
        lines.append(f"- KG candidates proposed: {len(package.kg_candidates)}")
    lines.append("")

    lines += ["## Citations", ""]
    if not response.citations:
        lines.append("(no literature sources cited — this run used no priors)")
    for source in response.citations:
        if source.type == "paper":
            lines.append(f"- {source.doi}" + (f" ({source.span})" if source.span else ""))
        else:
            lines.append(f"- KG triple `{source.triple_id}`")
    lines.append("")

    return "\n".join(lines)


# Coloring/emphasis here only ever targets structural tags (h1/h2/code/em),
# never a per-value class driven by request content — keeps the "escape
# the whole Markdown source" approach in render_pdf() sufficient on its
# own, no per-field trust bookkeeping needed to stay safe.
# Token system — a technical/lab-notebook register, not a marketing page:
# ink for body text, a muted steel-blue for structure (headings/rules),
# gray for asides, amber only for the one signature device (elevated
# anomaly severity, via blockquote — see _render_markdown). Two type
# roles: a CJK-capable serif for prose/headings, Courier for identifiers
# and data (run_id, field names, numeric evidence).
#
# body leads with STSong-Light, not Helvetica: this renderer's own labels
# (_DESIGN_PARAM_LABELS, _LITERATURE_VERDICT_LABELS, etc.) are Chinese, and
# Helvetica/Courier are reportlab's Latin-only base-14 fonts — confirmed
# live that every Chinese label in the PDF output rendered as a blank
# tofu box under the old Helvetica-first font stack, silently dropping
# every label this renderer exists to attach. STSong-Light is reportlab's
# built-in Simplified Chinese CID font (registered on demand by
# xhtml2pdf's getFontName -> set_asian_fonts when the CSS font-family
# resolves to a name in reportlab's cidfontdata), so this needs no bundled
# font file either — same "no font files to ship" constraint the old
# comment described, just with a font stack that actually covers the
# script this report renders.
_INK = "#1a1a2e"
_ACCENT = "#2a5a7c"
_MUTED = "#6b7280"
_HAIRLINE = "#dcdfe3"
_CODE_BG = "#f4f5f7"
_FLAG_BORDER = "#b45309"
_FLAG_BG = "#fdf6ec"

_PDF_STYLE = f"""
<style>
  body {{ font-family: "STSong-Light", Helvetica, Arial, sans-serif; font-size: 10pt; color: {_INK}; line-height: 1.55; }}
  h1 {{ font-size: 22pt; color: {_INK}; margin-bottom: 2px; }}
  h2 {{ font-size: 12pt; color: {_ACCENT}; margin-top: 18pt; margin-bottom: 8px;
        border-bottom: 1px solid {_HAIRLINE}; padding-bottom: 4px; }}
  hr {{ border: none; border-top: 1px solid {_HAIRLINE}; margin: 8px 0 14px 0; }}
  p {{ margin: 3px 0 10px 0; }}
  /* decimal, not the default disc: xhtml2pdf always draws the bullet
     glyph in the *current* font (xhtml2pdf/tags.py's pisaTagLI.start —
     "this should be the recent font, but it throws errors in
     Reportlab!"), and confirmed live that STSong-Light's disc bullet
     (U+2022) renders as a wrong, unrelated CJK character instead of a
     dot — not a missing-glyph box, an actively wrong one, in both a
     bare "<li>" and a plain paragraph containing "•" as body text, so
     this isn't fixable by scoping the character to a different tag.
     decimal's "1." markers are plain ASCII digits, confirmed safe under
     the same font. */
  ul {{ margin: 2px 0 10px 0; padding-left: 16px; list-style-type: decimal; }}
  li {{ margin-bottom: 4px; }}
  code {{ font-family: Courier, monospace; background-color: {_CODE_BG}; font-size: 9pt;
          padding: 0 3px; color: {_INK}; }}
  em {{ color: {_MUTED}; }}
  blockquote {{ margin: 6px 0 12px 0; padding: 6px 12px; border-left: 3px solid {_FLAG_BORDER};
                background-color: {_FLAG_BG}; }}
  blockquote p, blockquote ul {{ margin: 2px 0; }}
</style>
"""


def _escape_tag_open(markdown_text: str) -> str:
    """Neutralizes only "&" and "<" — the two characters a parser actually
    needs to recognize a live HTML tag (an opening "<") or to be tricked
    by a double-decode ("&amp;lt;..."). This is the real fix for the SSRF
    described in render_pdf's docstring: "<" -> "&lt;" means
    '<img src="...">' can never form a real tag, regardless of what
    follows.

    Deliberately narrower than `html.escape()`, which also escapes ">",
    '"', and "'" — confirmed live that escaping ">" breaks Markdown's own
    blockquote syntax ("> text", used for the Anomalies severity
    highlight below) since Markdown never sees a literal ">" to key off
    of, and escaping quotes was producing literal "&#x27;" in the
    rendered PDF for ordinary text like "the design's optimal length"
    (xhtml2pdf doesn't decode that entity the way a browser would). Since
    neither ">" nor a bare quote character can open a tag on its own,
    leaving them unescaped doesn't reopen the SSRF — confirmed live
    against the original payload with this narrower escape.
    """
    return markdown_text.replace("&", "&amp;").replace("<", "&lt;")


def render_pdf(markdown_text: str) -> bytes:
    """Markdown -> HTML (`markdown`) -> PDF (`xhtml2pdf`).

    See `_escape_tag_open`'s docstring for what's escaped and why.
    Escaping means any literal "<" a free-text field contributed renders
    as visible text in the PDF (e.g. a stray "<img src=...>" shows up as
    that literal string), never as a live tag `markdown` would otherwise
    pass through and xhtml2pdf would then try to fetch. Tried xhtml2pdf's
    own `link_callback` hook first as a resource-fetching guard —
    confirmed live that returning None from it does *not* actually stop
    the fetch (xhtml2pdf falls through to fetching the URI directly), so
    that alone would have been a false sense of security; source-escaping
    is the real guard here.
    """
    escaped = _escape_tag_open(markdown_text)
    body_html = md_lib.markdown(escaped)
    document = f"<html><head>{_PDF_STYLE}</head><body>{body_html}</body></html>"
    buffer = BytesIO()
    result = pisa.CreatePDF(document, dest=buffer)
    if result.err:
        raise RuntimeError(f"PDF rendering failed ({result.err} error(s))")
    return buffer.getvalue()


def build_report(request: ReportRequest, trace_id: str) -> ReportResponse:
    key_results = [
        KeyResult(
            field=field,
            value=value,
            unit=tec_bridge.SCALAR_UNITS.get(field),
            confidence_label=_confidence_label(request.anomalies),
        )
        for field, value in request.scalar_results.items()
    ]
    spec_summary = dict(request.design_parameters)
    spec_summary["n_pairs"] = float(request.n_pairs)

    response = ReportResponse(
        run_id=request.run_id,
        generated_at=ReportResponse.now_iso(),
        objective_and_constraints=request.task_context,
        spec_summary=spec_summary,
        key_results=key_results,
        literature_comparison=request.evaluation,
        anomalies_and_cautions=request.anomalies,
        update_proposal_summary=request.update_package,
        citations=_dedup_sources(request.priors),
        markdown="",
        trace_id=trace_id,
    )
    response.markdown = _render_markdown(response, request)
    return response
