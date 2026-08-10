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

import html
from io import BytesIO

import markdown as md_lib
from xhtml2pdf import pisa

from sciencerag.validate import tec_bridge
from sciencerag.report.models import KeyResult, ReportRequest, ReportResponse

_SEVERITY_RANK = {"info": 0, "warning": 1, "blocking": 2}


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
    lines = [f"# Run Report — `{response.run_id}`", "", f"_generated {response.generated_at}_", ""]

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
        lines.append(f"- `{field}` = {value}  _[run:{response.run_id}]_")
    lines.append("")

    lines += ["## Key Results", ""]
    if not response.key_results:
        lines.append("(no scalar results reported for this run)")
    for result in response.key_results:
        unit = f" {result.unit}" if result.unit else ""
        lines.append(
            f"- **{result.field}** = {result.value}{unit} "
            f"(confidence: {result.confidence_label}) _[run:{response.run_id}]_"
        )
    lines.append("")

    lines += ["## Comparison with Literature Priors", ""]
    lines.append(f"**Verdict:** {response.literature_comparison.verdict}")
    for deviation in response.literature_comparison.deviations:
        lines.append(
            f"- `{deviation.field}` actual={deviation.actual} "
            f"reference=[{deviation.reference_min}, {deviation.reference_max}] "
            f"→ **{deviation.verdict}** _[{deviation.source}:{deviation.reference_id}]_"
        )
    lines.append("")

    lines += ["## Anomalies & Cautions", ""]
    if not response.anomalies_and_cautions:
        lines.append("(no anomaly checks were run for this report)")
    for anomaly in response.anomalies_and_cautions:
        lines.append(f"- **{anomaly.check}** — {anomaly.severity}: `{anomaly.evidence}`")
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


_PDF_STYLE = """
<style>
  body { font-family: Helvetica, Arial, sans-serif; font-size: 10pt; }
  h1 { font-size: 16pt; }
  h2 { font-size: 13pt; margin-top: 14pt; }
  code { font-family: Courier, monospace; background-color: #f0f0f0; }
</style>
"""


def render_pdf(markdown_text: str) -> bytes:
    """Markdown -> HTML (`markdown`) -> PDF (`xhtml2pdf`).

    The Markdown source is HTML-escaped *before* conversion — this is the
    actual fix for the SSRF described in the module docstring, not a
    style choice. Escaping means any literal "<"/">" a free-text field
    contributed renders as visible text in the PDF (e.g. a stray
    "<img src=...>" shows up as that literal string), never as a live
    tag `markdown` would otherwise pass through and xhtml2pdf would then
    try to fetch. Tried xhtml2pdf's own `link_callback` hook first as a
    resource-fetching guard — confirmed live that returning None from it
    does *not* actually stop the fetch (xhtml2pdf falls through to
    fetching the URI directly), so that alone would have been a false
    sense of security; source-escaping is the real guard here.
    """
    escaped = html.escape(markdown_text)
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
