from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from personal_ai.llm import LLMClient, get_llm_client
from personal_ai.llm.models import LLMResponse
from personal_ai.main import app

client = TestClient(app)


def test_post_chat_endpoint_success() -> None:
    """Verify POST /api/v1/chat returns ChatResponse using mock LLMClient."""
    mock_llm_client = MagicMock(spec=LLMClient)
    mock_llm_client.generate_response = AsyncMock(
        return_value=LLMResponse(
            content="API test response",
            provider="openai",
            model="gpt-4o-mini",
            latency_ms=100.0,
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )
    )

    app.dependency_overrides[get_llm_client] = lambda: mock_llm_client

    try:
        payload = {
            "message": "What is the capital of France?",
            "system_prompt": "Answer in one word.",
        }
        response = client.post("/api/v1/chat", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "API test response"
        assert data["provider"] == "openai"
        assert data["model"] == "gpt-4o-mini"
        assert data["latency_ms"] == 100.0
        assert data["prompt_tokens"] == 10
        assert data["completion_tokens"] == 20
        assert data["total_tokens"] == 30
    finally:
        app.dependency_overrides.clear()


def test_post_chat_validation_error() -> None:
    """Verify POST /api/v1/chat returns 422 Unprocessable Entity on empty message."""
    payload = {"message": ""}
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 422
