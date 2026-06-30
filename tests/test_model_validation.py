from fastapi.testclient import TestClient

import app.main as main
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_embeddings_requires_inbound_api_key():
    # No SSUAI_SERVICE_API_KEY configured (CI default) => gate fails closed with 401,
    # never falling through to an unauthenticated upstream call.
    response = client.post("/v1/embeddings", json={"text": "숭실대학교 정보 검색 테스트"})
    assert response.status_code == 401


def test_embeddings_authed_but_upstream_unconfigured(monkeypatch):
    # Inbound auth passes (matching key) but no upstream credential => 503, and the
    # response carries a generic message, not provider internals.
    monkeypatch.setattr(main, "SERVICE_API_KEY", "test-inbound-key")
    monkeypatch.setattr(main, "GEMINI_API_KEY", "")
    response = client.post(
        "/v1/embeddings",
        json={"text": "test"},
        headers={"X-API-Key": "test-inbound-key"},
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "embedding upstream not configured"


def test_embeddings_wrong_key_rejected(monkeypatch):
    monkeypatch.setattr(main, "SERVICE_API_KEY", "correct-key")
    response = client.post(
        "/v1/embeddings",
        json={"text": "test"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 401
