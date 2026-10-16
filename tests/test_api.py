"""HTTP API auth and analyze smoke tests."""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import create_app


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("GAVEL_API_KEYS", "test-key")
    return TestClient(create_app({"test-key"}))


def _png_bytes(w: int = 64, h: int = 64) -> bytes:
    img = np.full((h, w, 3), 120, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_health_no_key(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_analyze_without_key_401(client):
    r = client.post(
        "/v1/analyze",
        files={"file": ("x.png", io.BytesIO(_png_bytes()), "image/png")},
    )
    assert r.status_code == 401


def test_analyze_with_key_200(client):
    r = client.post(
        "/v1/analyze",
        headers={"X-API-Key": "test-key"},
        files={"file": ("x.png", io.BytesIO(_png_bytes()), "image/png")},
        data={"include_overlay": "false"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "decision" in body
    assert "fusion" in body
    assert "protocol" in body


def test_analyze_bearer_auth(client):
    r = client.post(
        "/v1/analyze",
        headers={"Authorization": "Bearer test-key"},
        files={"file": ("x.png", io.BytesIO(_png_bytes()), "image/png")},
    )
    assert r.status_code == 200


def test_create_app_requires_keys(monkeypatch):
    monkeypatch.delenv("GAVEL_API_KEYS", raising=False)
    with pytest.raises(RuntimeError, match="GAVEL_API_KEYS"):
        create_app()
