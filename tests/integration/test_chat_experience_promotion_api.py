import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from personal_ai.application.experience import (
    ExperiencePromotionService,
    PromotionStrategy,
    RecordExperience,
)
from personal_ai.core.auth import create_access_token
from personal_ai.db.models import Base, ExperienceModel, Message, User
from personal_ai.db.repositories import SQLAlchemyExperienceRepository
from personal_ai.db.session import get_db_session
from personal_ai.infrastructure.experience import SQLAlchemyBackgroundExperienceProcessor
from personal_ai.llm import LLMClient, get_llm_client
from personal_ai.llm.models import LLMResponse
from personal_ai.main import app

client = TestClient(app)
test_engine = create_async_engine("sqlite+aiosqlite:///:memory:", poolclass=StaticPool, echo=False)
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


def create_real_test_user(email: str = "promouser@example.com") -> tuple[uuid.UUID, dict[str, str]]:
    """Helper to create a real User row in test DB and return (user_id, headers)."""
    async def _create() -> User:
        async with test_session_factory() as session:
            user = User(email=email, password_hash="hashedpass123")
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user

    user = asyncio.run(_create())
    token = create_access_token(user_id=user.id)
    return user.id, {"Authorization": f"Bearer {token}"}


def test_chat_promotion_creates_experience_linked_to_user_message() -> None:
    """Requirement G: Verify authenticated chat request creates user Message and promoted Experience linked via source_message_id."""
    user_id, headers = create_real_test_user("promotest@example.com")
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

    from personal_ai.api.routers.chat import get_background_experience_processor
    app.dependency_overrides[get_background_experience_processor] = lambda: SQLAlchemyBackgroundExperienceProcessor(
        session_factory=test_session_factory,
        llm_client=mock_llm_client,
        strategy=AlwaysPromoteUserStrategy(),
    )

    user_prompt = "I started learning FastAPI today."
    payload = {"message": user_prompt}

    response = client.post("/api/v1/chat", json=payload, headers=headers)
    assert response.status_code == 200

    # Give background promotion task time to run
    asyncio.run(asyncio.sleep(0.15))

    # Verify that both Message and Experience were persisted and linked via source_message_id
    async def verify_db() -> None:
        async with test_session_factory() as session:
            msg_stmt = select(Message).where(Message.content == user_prompt)
            msg_res = await session.execute(msg_stmt)
            user_msg = msg_res.scalar_one_or_none()

            assert user_msg is not None

            exp_stmt = select(ExperienceModel).where(ExperienceModel.source_message_id == user_msg.id)
            exp_res = await session.execute(exp_stmt)
            exp_model = exp_res.scalar_one_or_none()

            assert exp_model is not None
            assert exp_model.content == user_prompt
            assert exp_model.source == "CHAT"
            assert exp_model.status == "RECEIVED"
            assert exp_model.source_message_id == user_msg.id
            assert exp_model.user_id == user_id

    asyncio.run(verify_db())


def test_duplicate_concurrent_promotion_race_handled_gracefully() -> None:
    """Requirement I: Verify duplicate concurrent promotion on same source_message_id is safely handled."""
    user_id, _ = create_real_test_user("racetest@example.com")

    async def run_race() -> None:
        async with test_session_factory() as session:
            msg = Message(conversation_id=uuid.uuid4(), role="user", content="Race condition message")
            session.add(msg)
            await session.commit()
            await session.refresh(msg)

            repo = SQLAlchemyExperienceRepository(session=session)
            record_exp = RecordExperience(repository=repo)
            service = ExperiencePromotionService(
                record_experience=record_exp,
                strategy=AlwaysPromoteUserStrategy(),
                experience_repo=repo,
            )

            # First promotion succeeds
            res1 = await service.promote_message(message=msg, user_id=user_id)
            assert res1.promoted is True
            assert res1.experience_id is not None

            # Second promotion attempt on same source_message_id is gracefully handled without error
            res2 = await service.promote_message(message=msg, user_id=user_id)
            assert res2.promoted is False
            assert res2.experience_id == res1.experience_id

    asyncio.run(run_race())
