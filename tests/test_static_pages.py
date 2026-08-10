"""Smoke test for the static web layer routes (spec §7, M5)."""

import pytest
from fastapi.testclient import TestClient

from sciencerag.app import FRONTEND_DIST, app

client = TestClient(app)


def test_workbench_page_serves():
    response = client.get("/workbench")
    assert response.status_code == 200
    assert "ScienceRAG Workbench" in response.text


def test_demo_page_still_serves():
    response = client.get("/demo")
    assert response.status_code == 200


# The real spec §7 Vite/React frontend (frontend/) is a separate build step
# (`npm run build`) — these are skipped, not failed, when frontend/dist
# hasn't been built yet (e.g. a fresh checkout before running that step),
# same as the rest of the suite never requires Node/npm to be installed.
_frontend_built = FRONTEND_DIST.exists()


@pytest.mark.skipif(not _frontend_built, reason="frontend/dist not built (run `npm run build` in frontend/)")
def test_root_redirects_to_app():
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/app"


@pytest.mark.skipif(not _frontend_built, reason="frontend/dist not built (run `npm run build` in frontend/)")
def test_app_serves_built_frontend_with_correct_asset_base():
    response = client.get("/app/", follow_redirects=False)
    assert response.status_code == 200
    assert '"/app/assets/' in response.text or "'/app/assets/" in response.text


@pytest.mark.skipif(not _frontend_built, reason="frontend/dist not built (run `npm run build` in frontend/)")
@pytest.mark.parametrize("client_side_route", ["/app/ask", "/app/graph", "/app/reports"])
def test_client_side_routes_serve_app_shell_not_404(client_side_route):
    """react-router's routes (/ask, /graph, /reports) only exist in the
    browser — a real HTTP request for one (page refresh, typed/bookmarked/
    shared URL) has no matching file on disk. Regression test for a real
    bug: plain StaticFiles 404s here instead of falling back to
    index.html, since Starlette raises HTTPException(404) rather than
    returning a 404 response in the not-found case (see
    sciencerag/app.py's SPAStaticFiles override and its docstring)."""
    response = client.get(client_side_route, follow_redirects=False)
    assert response.status_code == 200
    assert "<div id=\"root\">" in response.text


def test_unrelated_404_is_not_swallowed_by_spa_fallback():
    """The SPA fallback is scoped to the /app mount — a real 404 outside
    it (e.g. a typo'd API path) must stay a real 404, not silently become
    the frontend shell."""
    response = client.get("/sciencerag/this-route-does-not-exist")
    assert response.status_code == 404
