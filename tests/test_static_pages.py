"""Smoke test for the static web layer routes (spec §7, M5)."""

from fastapi.testclient import TestClient

from sciencerag.app import app

client = TestClient(app)


def test_workbench_page_serves():
    response = client.get("/workbench")
    assert response.status_code == 200
    assert "ScienceRAG Workbench" in response.text


def test_demo_page_still_serves():
    response = client.get("/demo")
    assert response.status_code == 200
