from unittest.mock import AsyncMock, MagicMock
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from personal_ai.db.models import Base
from personal_ai.db.session import get_db_session
from personal_ai.llm import LLMClient, LLMStreamChunk, get_llm_client
from personal_ai.llm.models import LLMResponse
from personal_ai.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_db_and_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setup async in-memory SQLite database session override for integration tests."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

    import asyncio
    async def init_tables() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(init_tables())

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    async def override_get_db_session() -> AsyncSession:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db_session
    yield
    app.dependency_overrides.clear()
    asyncio.run(engine.dispose())


def test_post_chat_endpoint_creates_conversation_and_returns_response() -> None:
    """Verify POST /api/v1/chat returns ChatResponse and conversation_id."""
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
    response = client.post("/api/v1/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "conversation_id" in data
    assert data["response"] == "API test response"
    assert data["provider"] == "openai"
    assert data["model"] == "gpt-4o-mini"
    assert data["latency_ms"] == 100.0
    assert data["prompt_tokens"] == 10
    assert data["completion_tokens"] == 20
    assert data["total_tokens"] == 30

    # Second call reusing conversation_id
    conv_id = data["conversation_id"]
    payload2 = {
        "message": "And what is its population?",
        "conversation_id": conv_id,
    }
    response2 = client.post("/api/v1/chat", json=payload2)
    assert response2.status_code == 200
    assert response2.json()["conversation_id"] == conv_id


def test_post_chat_validation_error() -> None:
    """Verify POST /api/v1/chat returns 422 Unprocessable Entity on empty message."""
    payload = {"message": ""}
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 422


def test_post_chat_stream_returns_text_event_stream() -> None:
    """Verify POST /api/v1/chat/stream returns text/event-stream with SSE token and done events."""

    async def mock_stream_gen(*args, **kwargs):
        yield LLMStreamChunk(content="Hello")
        yield LLMStreamChunk(content=" streaming")

    mock_llm_client = MagicMock(spec=LLMClient)
    mock_llm_client.stream_response = MagicMock(side_effect=mock_stream_gen)
    app.dependency_overrides[get_llm_client] = lambda: mock_llm_client

    payload = {"message": "Stream hello"}
    response = client.post("/api/v1/chat/stream", json=payload)

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    body = response.text
    assert 'data: {"type":"token","content":"Hello"}' in body
    assert 'data: {"type":"token","content":" streaming"}' in body
    assert 'data: {"type":"done"}' in body
