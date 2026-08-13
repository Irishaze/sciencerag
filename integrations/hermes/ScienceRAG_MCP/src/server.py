"""MCP adapter that lets Hermes call the ScienceRAG FastAPI service."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP


SCIENCERAG_HOME = Path(os.environ.get("SCIENCERAG_HOME", r"D:\comsol\ScienceRAG"))
SCIENCERAG_UV_EXE = os.environ.get("SCIENCERAG_UV_EXE", "uv")
SCIENCERAG_PORT = os.environ.get("SCIENCERAG_PORT", "8000")
SCIENCERAG_API_BASE_URL = os.environ.get("SCIENCERAG_API_BASE_URL", "auto").rstrip("/")
DEFAULT_TIMEOUT_SECONDS = float(os.environ.get("SCIENCERAG_API_TIMEOUT_SECONDS", "180"))

mcp = FastMCP("ScienceRAG")


def _url(path: str) -> str:
    return f"{_api_base_url()}/{path.lstrip('/')}"


def _api_base_url() -> str:
    if SCIENCERAG_API_BASE_URL and SCIENCERAG_API_BASE_URL.lower() != "auto":
        return SCIENCERAG_API_BASE_URL
    return f"http://127.0.0.1:{SCIENCERAG_PORT}"


def _request(method: str, path: str, payload: Any | None = None, timeout: float | None = None) -> Any:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(_url(path), data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=timeout or DEFAULT_TIMEOUT_SECONDS) as response:
            body = response.read()
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                return json.loads(body.decode("utf-8"))
            return {
                "ok": True,
                "contentType": content_type,
                "contentLength": len(body),
                "bodyPreview": body[:500].decode("utf-8", errors="replace"),
            }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ScienceRAG API HTTP {exc.code} {path}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"ScienceRAG API is not reachable at {_api_base_url()}: {exc.reason}") from exc


def _download(path: str, payload: dict[str, Any] | None, output_path: str | None, timeout: float | None = None) -> dict[str, Any]:
    req = urllib.request.Request(_url(path), headers={"Accept": "application/pdf"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout or DEFAULT_TIMEOUT_SECONDS) as response:
            body = response.read()
            disposition = response.headers.get("content-disposition", "")
            filename = _filename_from_disposition(disposition) or "sciencerag-report.pdf"
            target = Path(output_path) if output_path else Path(SCIENCERAG_HOME) / "artifacts" / "reports" / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
            return {
                "ok": True,
                "path": str(target),
                "contentType": response.headers.get("content-type", ""),
                "bytes": len(body),
            }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ScienceRAG report export failed HTTP {exc.code}: {body}") from exc


def _filename_from_disposition(disposition: str) -> str | None:
    if not disposition:
        return None
    for part in disposition.split(";"):
        key, _, value = part.strip().partition("=")
        if key.lower() == "filename*":
            _, _, encoded = value.partition("''")
            return urllib.parse.unquote(encoded.strip('"'))
        if key.lower() == "filename":
            return value.strip('"')
    return None


def _is_ready() -> bool:
    try:
        _request("GET", "/sciencerag/reports", timeout=2)
        return True
    except Exception:
        return False


def _start_server_if_needed() -> bool:
    if _is_ready():
        return True
    if not SCIENCERAG_HOME.exists():
        return False
    subprocess.Popen(
        [SCIENCERAG_UV_EXE, "run", "uvicorn", "sciencerag.app:app", "--port", SCIENCERAG_PORT],
        cwd=str(SCIENCERAG_HOME),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    return True


@mcp.tool()
def sciencerag_launch(wait_seconds: int = 60) -> dict[str, Any]:
    """Start the ScienceRAG API (uv run uvicorn) if it isn't already running, and wait for it to become ready."""
    if _is_ready():
        return {"ok": True, "apiBaseUrl": _api_base_url(), "alreadyRunning": True}
    if not SCIENCERAG_HOME.exists():
        return {"ok": False, "error": f"ScienceRAG repo not found: {SCIENCERAG_HOME}"}
    _start_server_if_needed()
    deadline = time.time() + max(0, wait_seconds)
    while time.time() < deadline:
        if _is_ready():
            return {"ok": True, "apiBaseUrl": _api_base_url(), "alreadyRunning": False}
        time.sleep(1)
    return {"ok": False, "apiBaseUrl": _api_base_url(), "error": "ScienceRAG launched, but API did not become ready in time."}


@mcp.tool()
def sciencerag_status() -> dict[str, Any]:
    """Report whether the ScienceRAG API is reachable."""
    ready = _is_ready()
    return {"ok": ready, "apiBaseUrl": _api_base_url(), "home": str(SCIENCERAG_HOME)}


@mcp.tool()
def sciencerag_ask(question: str, max_hits: int = 10) -> dict[str, Any]:
    """Ask a question over the ScienceRAG knowledge graph + literature corpus."""
    _start_server_if_needed()
    return _request("POST", "/sciencerag/ask", {"question": question, "max_hits": max_hits})


@mcp.tool()
def sciencerag_graph() -> dict[str, Any]:
    """Return the full knowledge-graph subgraph (nodes + edges)."""
    _start_server_if_needed()
    return _request("GET", "/sciencerag/graph")


@mcp.tool()
def sciencerag_priors(
    query: str,
    max_priors: int = 5,
    allow_external: bool = False,
    task_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Retrieve literature-grounded priors for a design query."""
    _start_server_if_needed()
    payload: dict[str, Any] = {"query": query, "max_priors": max_priors, "allow_external": allow_external}
    if task_context is not None:
        payload["task_context"] = task_context
    return _request("POST", "/sciencerag/priors", payload)


@mcp.tool()
def sciencerag_priors_batch_evidence(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Get supporting/refuting literature evidence for a batch of candidate designs."""
    _start_server_if_needed()
    return _request("POST", "/sciencerag/priors/batch_evidence", {"candidates": candidates})


@mcp.tool()
def sciencerag_validate(
    run_id: str,
    design_parameters: dict[str, float] | None = None,
    n_pairs: int = 1,
    scalar_results: dict[str, float] | None = None,
    latent_state: list[float] | None = None,
    prior_ids: list[str] | None = None,
    priors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run anomaly checks + benchmark evaluation on a simulation/experiment run."""
    _start_server_if_needed()
    payload: dict[str, Any] = {
        "run_id": run_id,
        "design_parameters": design_parameters or {},
        "n_pairs": n_pairs,
        "scalar_results": scalar_results or {},
        "prior_ids": prior_ids or [],
        "priors": priors or [],
    }
    if latent_state is not None:
        payload["latent_state"] = latent_state
    return _request("POST", "/sciencerag/validate", payload)


@mcp.tool()
def sciencerag_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Build and store a report. payload follows the ScienceRAG /sciencerag/report request schema
    (run_id, design_parameters, scalar_results, priors, anomalies, evaluation, update_package —
    the evaluation/update_package fields are normally the output of a prior sciencerag_validate call)."""
    _start_server_if_needed()
    return _request("POST", "/sciencerag/report", payload)


@mcp.tool()
def sciencerag_list_reports() -> list[dict[str, Any]]:
    """List stored reports."""
    _start_server_if_needed()
    return _request("GET", "/sciencerag/reports")


@mcp.tool()
def sciencerag_get_report(stem: str) -> dict[str, Any]:
    """Fetch a stored report by its stem id."""
    _start_server_if_needed()
    return _request("GET", f"/sciencerag/reports/{urllib.parse.quote(stem)}")


@mcp.tool()
def sciencerag_export_report_pdf(stem: str, output_path: str | None = None) -> dict[str, Any]:
    """Render a stored report as PDF and save it to disk."""
    _start_server_if_needed()
    return _download(f"/sciencerag/reports/{urllib.parse.quote(stem)}/pdf", None, output_path)


@mcp.tool()
def sciencerag_kg_candidates_pending() -> list[dict[str, Any]]:
    """List pending knowledge-graph candidate batches awaiting human approval."""
    _start_server_if_needed()
    return _request("GET", "/sciencerag/kg_candidates/pending")


@mcp.tool()
def sciencerag_kg_candidate_detail(stem: str) -> dict[str, Any]:
    """Get the candidates in one pending KG batch."""
    _start_server_if_needed()
    return _request("GET", f"/sciencerag/kg_candidates/pending/{urllib.parse.quote(stem)}")


@mcp.tool()
def sciencerag_approve_kg_candidate(
    stem: str,
    operator: str = "hermes",
    reason: str = "",
    approve_all: bool = False,
    indices: list[int] | None = None,
) -> dict[str, Any]:
    """Approve some or all candidates in a pending KG batch (human-in-the-loop gate — only call
    this after the human operator using Hermes has actually reviewed the candidates)."""
    _start_server_if_needed()
    payload = {
        "approve_all": approve_all,
        "indices": indices or [],
        "operator": operator,
        "reason": reason,
    }
    return _request("POST", f"/sciencerag/kg_candidates/pending/{urllib.parse.quote(stem)}/approve", payload)


def main() -> None:
    if os.environ.get("SCIENCERAG_AUTO_LAUNCH", "0").lower() in {"1", "true", "yes"}:
        _start_server_if_needed()
    mcp.run()


if __name__ == "__main__":
    main()
