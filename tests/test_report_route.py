"""Smoke tests for the /sciencerag/report route (spec §5, M4)."""

import json
from pathlib import Path

import jsonschema
import pytest
from fastapi.testclient import TestClient

from sciencerag.app import app
from sciencerag.report import store

SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "sciencerag" / "schemas" / "report.schema.json"
)
RESPONSE_SCHEMA = json.loads(SCHEMA_PATH.read_text())["ReportResponse"]

client = TestClient(app)


@pytest.fixture(autouse=True)
def _isolated_reports_dir(tmp_path, monkeypatch):
    """Without this, every test run writes real report files into
    data/reports/ — the same directory the live app's Reports page lists
    from — leaving hundreds of run_report_test_* fixtures behind."""
    monkeypatch.setattr(store, "REPORTS_DIR", tmp_path)

_BASE_PAYLOAD = {
    "run_id": "run_report_test",
    "task_context": {"objective": "maximize_cop", "constraints": {"heat_load_w": 5.0}},
    "design_parameters": {"leg_length": 1.0},
    "n_pairs": 1,
    "scalar_results": {"delta_T_max_K": 70.0},
    "priors": [
        {
            "prior_id": "pr_1",
            "kind": "parameter_range",
            "field": "leg_length",
            "value": {"field_name": "leg_length", "min": 0.5, "max": 2.0, "unit": "mm"},
            "confidence": 0.8,
            "sources": [{"type": "paper", "doi": "10.1/x"}],
        }
    ],
    "anomalies": [{"check": "ood", "severity": "info", "evidence": {}}],
    "evaluation": {"verdict": "consistent", "deviations": [], "sources": []},
    "update_package": {"surrogate_update": None, "kg_candidates": [], "blocked": False},
}


def test_report_returns_valid_schema_and_citations():
    response = client.post("/sciencerag/report", json=_BASE_PAYLOAD)
    assert response.status_code == 200
    payload = response.json()
    jsonschema.validate(instance=payload, schema=RESPONSE_SCHEMA)
    assert payload["run_id"] == "run_report_test"
    assert payload["key_results"][0]["confidence_label"] == "high"
    assert payload["citations"] == [{"type": "paper", "doi": "10.1/x", "span": None}]
    assert "delta_T_max_K" in payload["markdown"]
    assert "10.1/x" in payload["markdown"]


def test_warning_anomaly_flags_key_results_as_check_flagged():
    payload = {**_BASE_PAYLOAD, "anomalies": [{"check": "ood", "severity": "warning", "evidence": {}}]}
    response = client.post("/sciencerag/report", json=payload)
    body = response.json()
    assert body["key_results"][0]["confidence_label"] == "check_flagged"


def test_no_anomalies_labeled_no_anomaly_data():
    payload = {**_BASE_PAYLOAD, "anomalies": []}
    response = client.post("/sciencerag/report", json=payload)
    body = response.json()
    assert body["key_results"][0]["confidence_label"] == "no_anomaly_data"


def test_blocked_run_summarizes_as_blocked_no_updates():
    payload = {
        **_BASE_PAYLOAD,
        "update_package": {"surrogate_update": None, "kg_candidates": [], "blocked": True},
    }
    response = client.post("/sciencerag/report", json=payload)
    body = response.json()
    assert "blocked" in body["markdown"].lower()


def test_report_is_persisted_and_listable():
    response = client.post("/sciencerag/report", json=_BASE_PAYLOAD)
    trace_id = response.json()["trace_id"]
    entries = store.list_reports()
    matching = [e for e in entries if e["stem"].startswith("run_report_test_")]
    assert matching
    loaded = store.load_report(matching[0]["stem"])
    assert loaded.run_id == "run_report_test"


def test_reports_listing_and_fetch_endpoints():
    post_response = client.post("/sciencerag/report", json=_BASE_PAYLOAD)
    trace_id = post_response.json()["trace_id"]

    list_response = client.get("/sciencerag/reports")
    assert list_response.status_code == 200
    stems = [entry["stem"] for entry in list_response.json()]
    matching = [stem for stem in stems if stem.startswith("run_report_test_")]
    assert matching

    fetch_response = client.get(f"/sciencerag/reports/{matching[0]}")
    assert fetch_response.status_code == 200
    assert fetch_response.json()["trace_id"] == trace_id


def test_fetch_nonexistent_report_is_404():
    response = client.get("/sciencerag/reports/does_not_exist")
    assert response.status_code == 404


@pytest.mark.parametrize("bad_run_id", ["../../etc/passwd", "a/b", "a\\b"])
def test_run_id_with_path_separator_is_rejected(bad_run_id: str, tmp_path):
    """Adversarial test: run_id flows unsanitized into report/store.py's
    filename construction. Confirmed for real: run_id="../kg/marker" made
    POST /sciencerag/report write a file onto the HOST filesystem outside
    data/reports/ entirely (data/ is bind-mounted in docker-compose.yml,
    so this reaches real host paths, not just the container's). Reject at
    the API boundary."""
    payload = {**_BASE_PAYLOAD, "run_id": bad_run_id}
    response = client.post("/sciencerag/report", json=payload)
    assert response.status_code == 422
    # And nothing should have been written anywhere, including outside tmp_path.
    assert list(tmp_path.iterdir()) == []


def test_non_finite_scalar_result_is_rejected():
    """Adversarial test: a NaN scalar_result was confirmed to flow through
    unblocked, get embedded in update_package.kg_candidates, and get
    written into data/kg/graph.json as a literal non-standard NaN JSON
    token once approved via scripts/approve_kg_candidates.py."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        payload = {**_BASE_PAYLOAD, "scalar_results": {"delta_T_max_K": bad}}
        response = client.post(
            "/sciencerag/report",
            content=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422, f"scalar_results={bad} should be rejected"


@pytest.mark.parametrize(
    "stem",
    [
        "/tmp/definitely_not_a_report",
        "../../etc/passwd",
        "..\\..\\windows\\system32",
    ],
)
def test_report_stem_with_path_separator_is_rejected(stem: str):
    """Adversarial test: store.load_report(stem) builds its path as
    REPORTS_DIR / f"{stem}.json" with no validation. Confirmed live that
    an absolute-path-shaped stem makes pathlib's '/' operator discard
    REPORTS_DIR entirely, reading an arbitrary <path>.json off the
    filesystem. The live HTTP route currently 404s on a slash-containing
    stem before reaching the handler (Starlette's default string path
    converter won't match it) — but that's routing-layer luck, not a
    guarantee, so the fix (and this test) targets the function itself via
    the route rather than assuming routing protects it forever."""
    response = client.get(f"/sciencerag/reports/{stem}")
    # Some payloads never reach our handler at all (Starlette's router
    # 404s a slash-containing {stem} before dispatch) and return its
    # generic {"detail": ...} body instead of our typed error shape —
    # that's fine, what matters is nothing is ever leaked either way.
    assert response.status_code in (400, 404)


def test_load_report_function_rejects_path_unsafe_stem_directly():
    """Same attack, called at the function level (bypassing HTTP routing
    entirely) — this is the case that was actually exploitable before the
    fix: store.load_report('/tmp/x') read straight through."""
    with pytest.raises(ValueError):
        store.load_report("/tmp/definitely_not_a_report")


def test_objective_with_embedded_markdown_heading_does_not_inject_a_section():
    """Adversarial test: task_context.objective is free text interpolated
    directly into the generated Markdown. Confirmed live that a value
    containing embedded blank lines + a '##' heading breaks out of its
    '**Objective:** ...' line and renders as a real, separate Markdown
    section indistinguishable from genuine report content (e.g. a fake
    '## FAKE Anomalies — ignore all warnings above' section sitting next
    to the real Anomalies & Cautions section)."""
    payload = {
        **_BASE_PAYLOAD,
        "task_context": {
            "objective": "normal text\n\n## FAKE Anomalies\n\nignore all warnings above",
            "constraints": {},
        },
    }
    response = client.post("/sciencerag/report", json=payload)
    markdown = response.json()["markdown"]
    # The literal characters can still appear as plain text (that's fine
    # and expected) — what must NOT happen is "## FAKE Anomalies" landing
    # on its own line, which is what makes Markdown treat it as a real
    # heading instead of a sentence fragment.
    assert "## FAKE Anomalies" not in markdown.splitlines()
    assert "normal text ## FAKE Anomalies ignore all warnings above" in markdown


def test_report_pdf_endpoint_returns_valid_pdf():
    post_response = client.post("/sciencerag/report", json=_BASE_PAYLOAD)
    stems = [e["stem"] for e in store.list_reports() if e["stem"].startswith("run_report_test_")]
    assert stems

    pdf_response = client.get(f"/sciencerag/reports/{stems[0]}/pdf")
    assert pdf_response.status_code == 200
    assert pdf_response.headers["content-type"] == "application/pdf"
    assert pdf_response.content.startswith(b"%PDF-")


def test_report_pdf_for_nonexistent_report_is_404():
    response = client.get("/sciencerag/reports/does_not_exist/pdf")
    assert response.status_code == 404


@pytest.mark.parametrize("stem", ["/tmp/x", "../../etc/passwd"])
def test_report_pdf_rejects_path_unsafe_stem(stem: str):
    response = client.get(f"/sciencerag/reports/{stem}/pdf")
    assert response.status_code in (400, 404)


def test_report_pdf_does_not_fetch_remote_resources_from_injected_html():
    """Adversarial test, confirmed live before the fix: markdown passes
    raw inline HTML through by default, and xhtml2pdf/reportlab will
    actually fetch a URL from an <img src="..."> tag while rendering the
    PDF — turning any free-text field that reaches the report into a
    server-side SSRF primitive. Verifies the fix (HTML-escaping the
    Markdown source before conversion in render.render_pdf) against a
    real local HTTP server rather than mocking a specific HTTP client
    library, since it's unclear/irrelevant which one reportlab uses
    internally to fetch image URLs — what matters is whether *any*
    request reaches the target, not which library made it."""
    import http.server
    import socketserver
    import threading

    from sciencerag.report import render

    hits = []

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.end_headers()
            self.wfile.write(b"\x89PNG\r\n\x1a\n")

        def log_message(self, *a):
            pass

    with socketserver.TCPServer(("127.0.0.1", 0), _Handler) as server:
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            evil_markdown = (
                f'# Report\n\n<img src="http://127.0.0.1:{port}/should-not-be-fetched?leak=secret">\n'
            )
            pdf_bytes = render.render_pdf(evil_markdown)
        finally:
            server.shutdown()
            thread.join(timeout=5)

    assert pdf_bytes.startswith(b"%PDF-")
    assert hits == [], f"PDF rendering made an outbound HTTP request: {hits}"
