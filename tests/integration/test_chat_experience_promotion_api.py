import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from personal_ai.application.experience import (
    ExperiencePromotionService,
    PromotionStrategy,
    RecordExperience,
)
from personal_ai.db.models import Base, ExperienceModel, Message
from personal_ai.db.repositories import SQLAlchemyExperienceRepository
from personal_ai.db.session import get_db_session
from personal_ai.llm import LLMClient, get_llm_client
from personal_ai.llm.models import LLMResponse
from personal_ai.main import app

client = TestClient(app)
test_engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
test_session_factory = async_sessionmaker(bind=test_engine, class_=AsyncSession, expire_on_commit=False)


class AlwaysPromoteUserStrategy(PromotionStrategy):
    """Test strategy promoting all user messages unconditionally."""

    def evaluate(self, message: Message, explicit_signal: bool = False) -> bool:
        role_str = message.role.value if hasattr(message.role, "value") else str(message.role)
        return role_str.lower() == "user"


@pytest.fixture(autouse=True)
def setup_chat_promotion_db(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_chat_promotion_creates_experience_linked_to_user_message() -> None:
    """Verify chat request creates user Message and promoted Experience linked via source_message_id."""
    mock_llm_client = MagicMock(spec=LLMClient)
    mock_llm_client.generate_response = AsyncMock(
        return_value=LLMResponse(
            content="I noted that.",
            provider="openai",
            model="gpt-4o-mini",
            latency_ms=50.0,
        )
    )
    app.dependency_overrides[get_llm_client] = lambda: mock_llm_client

    # Inject promotion service with AlwaysPromoteUserStrategy into ChatService via dependency override
    from personal_ai.api.routers.chat import get_experience_promotion_service

    async def override_chat_promotion_service() -> ExperiencePromotionService:
        async with test_session_factory() as session:
            repo = SQLAlchemyExperienceRepository(session=session)
            record_exp = RecordExperience(repository=repo)
            return ExperiencePromotionService(
                record_experience=record_exp,
                strategy=AlwaysPromoteUserStrategy(),
            )

    app.dependency_overrides[get_experience_promotion_service] = override_chat_promotion_service

    user_prompt = "I started learning FastAPI today."
    payload = {"message": user_prompt}

    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200

    # Verify that both Message and Experience were persisted and linked via source_message_id
    async def verify_db() -> None:
        async with test_session_factory() as session:
            # Query user message
            msg_stmt = select(Message).where(Message.content == user_prompt)
            msg_res = await session.execute(msg_stmt)
            user_msg = msg_res.scalar_one_or_none()

            assert user_msg is not None

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
