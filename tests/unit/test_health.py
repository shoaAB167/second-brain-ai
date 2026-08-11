from fastapi.testclient import TestClient

from personal_ai.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    """Test GET /api/v1/health returns HTTP 200 with expected schema."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "app_name" in data
    assert "environment" in data
