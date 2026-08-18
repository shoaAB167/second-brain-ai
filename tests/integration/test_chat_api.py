import asyncio
from unittest.mock import AsyncMock, MagicMock
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from personal_ai.core.auth import create_access_token
from personal_ai.db.models import Base, User
from personal_ai.db.session import get_db_session
from personal_ai.llm import LLMClient, LLMStreamChunk, get_llm_client
from personal_ai.llm.models import LLMResponse
from personal_ai.main import app

client = TestClient(app)
test_engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
test_session_factory = async_sessionmaker(bind=test_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture(autouse=True)
def setup_test_db_and_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setup async in-memory SQLite database session override for integration tests."""
    async def init_tables() -> None:
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(init_tables())

    async def override_get_db_session() -> AsyncSession:
        async with test_session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db_session
    yield
    app.dependency_overrides.clear()
    asyncio.run(test_engine.dispose())


def create_real_test_user(email: str = "realuser@example.com") -> tuple[uuid.UUID, str]:
    """Helper to create a real User row in test DB and return (user_id, jwt_token)."""
    async def _create() -> User:
        async with test_session_factory() as session:
            user = User(email=email, password_hash="hashedpass123")
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user

    user = asyncio.run(_create())
    token = create_access_token(user_id=user.id)
    return user.id, token


def test_unauthenticated_chat_request_is_rejected() -> None:
    """15. Verify unauthenticated chat request is rejected with HTTP 401."""
    payload = {"message": "Hello"}
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 401

    stream_response = client.post("/api/v1/chat/stream", json=payload)
    assert stream_response.status_code == 401


def test_jwt_for_nonexistent_user_returns_401() -> None:
    """Requirement E: Verify valid JWT for nonexistent/deleted user is rejected with HTTP 401."""
    fake_user_id = uuid.uuid4()
    fake_token = create_access_token(user_id=fake_user_id)
    headers = {"Authorization": f"Bearer {fake_token}"}

    payload = {"message": "Hello"}
    response = client.post("/api/v1/chat", json=payload, headers=headers)
    assert response.status_code == 401
    assert "invalid authentication credentials" in response.text.lower()


def test_post_chat_endpoint_creates_conversation_and_returns_response() -> None:
    """Requirement D & F: Verify authenticated POST /api/v1/chat returns ChatResponse with real DB User."""
    _, token = create_real_test_user(email="chattest@example.com")
    headers = {"Authorization": f"Bearer {token}"}

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

    payload = {
        "message": "What is the capital of France?",
        "system_prompt": "Answer in one word.",
    }
    response = client.post("/api/v1/chat", json=payload, headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert "conversation_id" in data
    assert data["response"] == "API test response"

    # Second call reusing conversation_id by same authenticated user
    conv_id = data["conversation_id"]
    payload2 = {
        "message": "And what is its population?",
        "conversation_id": conv_id,
    }
    response2 = client.post("/api/v1/chat", json=payload2, headers=headers)
    assert response2.status_code == 200
    assert response2.json()["conversation_id"] == conv_id


def test_user_a_cannot_access_user_b_conversation() -> None:
    """Requirement F: Verify User A cannot access or continue User B's conversation."""
    _, token_a = create_real_test_user(email="usera@example.com")
    _, token_b = create_real_test_user(email="userb@example.com")

    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}

    mock_llm_client = MagicMock(spec=LLMClient)
    mock_llm_client.generate_response = AsyncMock(
        return_value=LLMResponse(content="Response", provider="openai", model="gpt-4o-mini", latency_ms=10.0)
    )
    app.dependency_overrides[get_llm_client] = lambda: mock_llm_client

    # User A creates a conversation
    resp_a = client.post("/api/v1/chat", json={"message": "Hello from A"}, headers=headers_a)
    assert resp_a.status_code == 200
    conv_id = resp_a.json()["conversation_id"]

    # User B attempts to send message to User A's conversation_id -> rejected with 404 (do not leak existence)
    resp_b = client.post(
        "/api/v1/chat",
        json={"message": "Hi from B trying to access A's chat", "conversation_id": conv_id},
        headers=headers_b,
    )
    assert resp_b.status_code == 404
    assert "not found" in resp_b.text.lower()


def test_post_chat_stream_returns_text_event_stream() -> None:
    """Requirement B: Verify streaming chat returns text/event-stream without waiting for classification."""
    _, token = create_real_test_user(email="streamuser@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    async def mock_stream_gen(*args, **kwargs):
        yield LLMStreamChunk(content="Hello")
        yield LLMStreamChunk(content=" streaming")

    mock_llm_client = MagicMock(spec=LLMClient)
    mock_llm_client.stream_response = MagicMock(side_effect=mock_stream_gen)
    mock_llm_client.generate_response = AsyncMock(
        return_value=LLMResponse(
            content='{"is_experience": false, "type": null, "importance": 0.0, "confidence": 0.0}',
            provider="openai",
            model="gpt-4o-mini",
            latency_ms=10.0,
        )
    )
    app.dependency_overrides[get_llm_client] = lambda: mock_llm_client

    payload = {"message": "Stream hello"}
    response = client.post("/api/v1/chat/stream", json=payload, headers=headers)

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    body = response.text
    assert 'data: {"type":"token","content":"Hello"}' in body
    assert 'data: {"type":"token","content":" streaming"}' in body
    assert '"type":"done"' in body
    assert '"conversation_id"' in body
