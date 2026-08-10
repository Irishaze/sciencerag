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
