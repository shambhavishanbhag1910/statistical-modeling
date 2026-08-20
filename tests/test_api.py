from fastapi.testclient import TestClient

from failure_risk.api import app


def test_health_endpoint_exists():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in {"healthy", "model_not_loaded"}
