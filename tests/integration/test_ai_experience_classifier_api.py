import asyncio
import json
import uuid
from unittest.mock import AsyncMock, MagicMock
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from personal_ai.db.models import Base, ExperienceClassificationModel, ExperienceModel, Message
from personal_ai.db.session import get_db_session
from personal_ai.llm import LLMClient, get_llm_client
from personal_ai.llm.exceptions import LLMConnectionException
from personal_ai.llm.models import LLMResponse
from personal_ai.main import app

client = TestClient(app)
test_engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
test_session_factory = async_sessionmaker(bind=test_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture(autouse=True)
def setup_ai_classifier_db(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_chat_triggers_ai_classifier_and_persists_classification_and_experience() -> None:
    """Verify chat request uses AI Experience Classifier and persists classification & experience."""
    mock_llm_client = MagicMock(spec=LLMClient)

    # Return structured classifier JSON on first call, chat response on second call
    classifier_json = json.dumps({
        "is_experience": True,
        "type": "GOAL",
        "importance": 0.90,
        "confidence": 0.95,
    })

    async def mock_generate_response(messages, **kwargs):
        # Check if messages contain classifier prompt
        is_classifier = any("Personal Experience Classifier" in msg.content for msg in messages)
        if is_classifier:
            return LLMResponse(content=classifier_json, provider="openai", model="gpt-4o-mini", latency_ms=10.0)
        return LLMResponse(content="That is an exciting goal!", provider="openai", model="gpt-4o-mini", latency_ms=10.0)

    mock_llm_client.generate_response = AsyncMock(side_effect=mock_generate_response)
    app.dependency_overrides[get_llm_client] = lambda: mock_llm_client

    user_prompt = "I've decided to focus my career on AI engineering."
    payload = {"message": user_prompt}

    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200

    # Verify database persistence of Message, ExperienceModel, and ExperienceClassificationModel
    async def verify_db() -> None:
        async with test_session_factory() as session:
            # Query user message
            msg_stmt = select(Message).where(Message.content == user_prompt)
            msg_res = await session.execute(msg_stmt)
            user_msg = msg_res.scalar_one_or_none()
            assert user_msg is not None

            # Query classification provenance record
            class_stmt = select(ExperienceClassificationModel).where(
                ExperienceClassificationModel.source_message_id == user_msg.id
            )
            class_res = await session.execute(class_stmt)
            class_model = class_res.scalar_one_or_none()

            assert class_model is not None
            assert class_model.is_experience is True
            assert class_model.type == "GOAL"
            assert class_model.importance == 0.90
            assert class_model.confidence == 0.95

            # Query promoted experience
            exp_stmt = select(ExperienceModel).where(ExperienceModel.source_message_id == user_msg.id)
            exp_res = await session.execute(exp_stmt)
            exp_model = exp_res.scalar_one_or_none()

            assert exp_model is not None
            assert exp_model.content == user_prompt
            assert exp_model.source == "CHAT"
            assert exp_model.status == "RECEIVED"
            assert exp_model.source_message_id == user_msg.id

    asyncio.run(verify_db())


def test_classifier_failure_does_not_break_chat() -> None:
    """Verify chat request succeeds even if classifier encounters LLM exception."""
    mock_llm_client = MagicMock(spec=LLMClient)

    async def mock_generate_response(messages, **kwargs):
        is_classifier = any("Personal Experience Classifier" in msg.content for msg in messages)
        if is_classifier:
            raise LLMConnectionException("Connection timeout during classification")
        return LLMResponse(content="Normal chat response", provider="openai", model="gpt-4o-mini", latency_ms=10.0)

    mock_llm_client.generate_response = AsyncMock(side_effect=mock_generate_response)
    app.dependency_overrides[get_llm_client] = lambda: mock_llm_client

    payload = {"message": "I feel tired today."}
    response = client.post("/api/v1/chat", json=payload)

    # Chat MUST succeed 200 OK
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Normal chat response"
