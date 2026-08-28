import asyncio
import json
from pathlib import Path
import tempfile
import uuid
from unittest.mock import AsyncMock, MagicMock
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from personal_ai.core.auth import create_access_token
from personal_ai.db.models import Base, ExperienceClassificationModel, ExperienceModel, Message, User
from personal_ai.db.session import get_db_session
from personal_ai.infrastructure.embedding import MockEmbeddingProvider
from personal_ai.infrastructure.experience import SQLAlchemyBackgroundExperienceProcessor
from personal_ai.llm import LLMClient, get_llm_client
from personal_ai.llm.exceptions import LLMConnectionException
from personal_ai.llm.models import LLMResponse
from personal_ai.main import app

client = TestClient(app)

tmp_db_path = Path(tempfile.gettempdir()) / f"test_ai_classifier_{uuid.uuid4().hex}.db"
test_engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_db_path.as_posix()}", echo=False)
test_session_factory = async_sessionmaker(bind=test_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture(autouse=True)
def setup_ai_classifier_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setup async SQLite database session override for integration tests."""
    async def init_tables() -> None:
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(init_tables())

    async def override_get_db_session() -> AsyncSession:
        async with test_session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db_session

    from personal_ai.api.routers.chat import get_background_experience_processor
    app.dependency_overrides[get_background_experience_processor] = lambda: SQLAlchemyBackgroundExperienceProcessor(
        session_factory=test_session_factory,
        llm_client=app.dependency_overrides.get(get_llm_client, lambda: MagicMock())(),
        embedding_provider=MockEmbeddingProvider(),
    )

    yield
    app.dependency_overrides.clear()


@pytest.fixture(scope="module", autouse=True)
def cleanup_tmp_db() -> None:
    """Cleanup temporary database file after test module completes."""
    yield
    asyncio.run(test_engine.dispose())
    tmp_db_path.unlink(missing_ok=True)


def create_real_test_user(email: str = "classifieruser@example.com") -> tuple[uuid.UUID, dict[str, str]]:
    """Helper to create real User in DB and return (user_id, headers)."""
    async def _create() -> User:
        async with test_session_factory() as session:
            user = User(email=email, password_hash="hashedpass")
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user

    user = asyncio.run(_create())
    token = create_access_token(user_id=user.id)
    return user.id, {"Authorization": f"Bearer {token}"}


def test_chat_triggers_ai_classifier_and_extractor_and_persists_experience_and_embedding() -> None:
    """Requirement 18 & PR #10: Verify full pipeline: Message -> Classifier -> Extractor -> Promotion -> Experience DB -> Background Embedding."""
    user_id, headers = create_real_test_user("aipipeline@example.com")
    mock_llm_client = MagicMock(spec=LLMClient)

    classifier_json = json.dumps({
        "is_experience": True,
        "type": "GOAL",
        "importance": 0.90,
        "confidence": 0.95,
        "reasoning": "User states salary and role career target.",
    })

    extractor_json = json.dumps({
        "content": "Reach a 30 LPA backend engineering role",
        "domain": "career",
        "status": "active",
        "confidence": 0.94,
        "reasoning": "Extracted user goal concisely.",
    })

    async def mock_generate_response(messages, **kwargs):
        sys_prompt = messages[0].content if messages else ""
        if "Personal Experience Classifier" in sys_prompt:
            return LLMResponse(content=classifier_json, provider="openai", model="gpt-4o-mini", latency_ms=10.0)
        elif "Personal Experience Extractor" in sys_prompt:
            return LLMResponse(content=extractor_json, provider="openai", model="gpt-4o-mini", latency_ms=10.0)
        return LLMResponse(content="That is an exciting goal!", provider="openai", model="gpt-4o-mini", latency_ms=10.0)

    mock_llm_client.generate_response = AsyncMock(side_effect=mock_generate_response)
    app.dependency_overrides[get_llm_client] = lambda: mock_llm_client

    bg_processor = SQLAlchemyBackgroundExperienceProcessor(
        session_factory=test_session_factory,
        llm_client=mock_llm_client,
        embedding_provider=MockEmbeddingProvider(),
    )
    from personal_ai.api.routers.chat import get_background_experience_processor
    app.dependency_overrides[get_background_experience_processor] = lambda: bg_processor

    user_prompt = "I want to reach 30 LPA as a backend engineer."
    payload = {"message": user_prompt}

    response = client.post("/api/v1/chat", json=payload, headers=headers)
    assert response.status_code == 200

    # Execute background promotion and verify DB state deterministically
    async def run_promotion_and_verify() -> None:
        async with test_session_factory() as session:
            msg_stmt = select(Message).where(Message.content == user_prompt)
            msg_res = await session.execute(msg_stmt)
            user_msg = msg_res.scalar_one_or_none()
            assert user_msg is not None

            # Directly invoke background promotion for deterministic test execution
            await bg_processor.process_background_promotion(user_msg, user_id=user_id)

            exp_stmt = select(ExperienceModel).where(ExperienceModel.source_message_id == user_msg.id)
            exp_res = await session.execute(exp_stmt)
            exp_model = exp_res.scalar_one_or_none()

            class_stmt = select(ExperienceClassificationModel).where(
                ExperienceClassificationModel.source_message_id == user_msg.id
            )
            class_res = await session.execute(class_stmt)
            class_model = class_res.scalar_one_or_none()

            assert exp_model is not None
            assert exp_model.content == "Reach a 30 LPA backend engineering role"
            assert exp_model.type == "GOAL"
            assert exp_model.domain == "career"
            assert exp_model.extraction_confidence == 0.94
            assert exp_model.source == "CHAT"
            assert exp_model.status == "RECEIVED"
            assert exp_model.user_id == user_id
            assert class_model is not None
            assert class_model.is_experience is True
            assert class_model.type == "GOAL"
            assert class_model.experience_id == exp_model.id

            # PR #10: Verify embedding fields persisted in ExperienceModel
            assert exp_model.embedding_status == "COMPLETED"
            assert exp_model.embedding_model == "text-embedding-3-small"
            assert exp_model.embedding is not None
            assert len(exp_model.embedding) == 1536

    asyncio.run(run_promotion_and_verify())


def test_general_technical_question_is_not_promoted_to_experience() -> None:
    """Requirement 18: General technical question -> classifier is_experience=False -> Extractor NOT called -> NO Experience created in DB."""
    user_id, headers = create_real_test_user("techquestion@example.com")
    mock_llm_client = MagicMock(spec=LLMClient)

    classifier_json = json.dumps({
        "is_experience": False,
        "type": None,
        "importance": 0.10,
        "confidence": 0.99,
        "reasoning": "General technical knowledge question.",
    })

    async def mock_generate_response(messages, **kwargs):
        sys_prompt = messages[0].content if messages else ""
        if "Personal Experience Classifier" in sys_prompt:
            return LLMResponse(content=classifier_json, provider="openai", model="gpt-4o-mini", latency_ms=10.0)
        elif "Personal Experience Extractor" in sys_prompt:
            pytest.fail("Extractor MUST NOT be called when classifier evaluates is_experience=False")
        return LLMResponse(content="Dependency injection is a software design pattern...", provider="openai", model="gpt-4o-mini", latency_ms=10.0)

    mock_llm_client.generate_response = AsyncMock(side_effect=mock_generate_response)
    app.dependency_overrides[get_llm_client] = lambda: mock_llm_client

    bg_processor = SQLAlchemyBackgroundExperienceProcessor(
        session_factory=test_session_factory,
        llm_client=mock_llm_client,
        embedding_provider=MockEmbeddingProvider(),
    )
    from personal_ai.api.routers.chat import get_background_experience_processor
    app.dependency_overrides[get_background_experience_processor] = lambda: bg_processor

    user_prompt = "What is dependency injection?"
    payload = {"message": user_prompt}

    response = client.post("/api/v1/chat", json=payload, headers=headers)
    assert response.status_code == 200

    async def verify_db() -> None:
        async with test_session_factory() as session:
            msg_stmt = select(Message).where(Message.content == user_prompt)
            msg_res = await session.execute(msg_stmt)
            user_msg = msg_res.scalar_one_or_none()
            assert user_msg is not None

            # Execute background promotion processor
            await bg_processor.process_background_promotion(user_msg, user_id=user_id)

            exp_stmt = select(ExperienceModel).where(ExperienceModel.source_message_id == user_msg.id)
            exp_res = await session.execute(exp_stmt)
            exp_model = exp_res.scalar_one_or_none()

            class_stmt = select(ExperienceClassificationModel).where(
                ExperienceClassificationModel.source_message_id == user_msg.id
            )
            class_res = await session.execute(class_stmt)
            class_model = class_res.scalar_one_or_none()

            # Classification record persisted with is_experience=False
            assert class_model is not None
            assert class_model.is_experience is False
            assert class_model.type is None

            # MUST NOT create an Experience entity in DB
            assert exp_model is None

    asyncio.run(verify_db())


def test_classifier_failure_does_not_break_chat() -> None:
    """Requirement C & 9: Verify chat request succeeds even if classifier encounters LLM exception."""
    _, headers = create_real_test_user("failtest@example.com")
    mock_llm_client = MagicMock(spec=LLMClient)

    async def mock_generate_response(messages, **kwargs):
        is_classifier = any("Personal Experience Classifier" in msg.content for msg in messages)
        if is_classifier:
            raise LLMConnectionException("Connection timeout during classification")
        return LLMResponse(content="Normal chat response", provider="openai", model="gpt-4o-mini", latency_ms=10.0)

    mock_llm_client.generate_response = AsyncMock(side_effect=mock_generate_response)
    app.dependency_overrides[get_llm_client] = lambda: mock_llm_client

    from personal_ai.api.routers.chat import get_background_experience_processor
    app.dependency_overrides[get_background_experience_processor] = lambda: SQLAlchemyBackgroundExperienceProcessor(
        session_factory=test_session_factory,
        llm_client=mock_llm_client,
        embedding_provider=MockEmbeddingProvider(),
    )

    payload = {"message": "I feel tired today."}
    response = client.post("/api/v1/chat", json=payload, headers=headers)

    # Chat MUST succeed 200 OK
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Normal chat response"
