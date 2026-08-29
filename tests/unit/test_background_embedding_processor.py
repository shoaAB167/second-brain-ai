import asyncio
from pathlib import Path
import tempfile
import uuid
from unittest.mock import AsyncMock, MagicMock
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from personal_ai.config.settings import get_settings
from personal_ai.db.models import Base, Conversation, ExperienceModel, Message
from personal_ai.infrastructure.embedding import MockEmbeddingProvider
from personal_ai.infrastructure.experience import SQLAlchemyBackgroundExperienceProcessor
from personal_ai.llm.client import LLMClient
from personal_ai.llm.models import LLMResponse

tmp_db_path = Path(tempfile.gettempdir()) / f"test_bg_embed_{uuid.uuid4().hex}.db"
test_engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_db_path.as_posix()}", echo=False)
test_session_factory = async_sessionmaker(bind=test_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture(autouse=True)
def setup_test_db() -> None:
    """Initialize temporary database tables."""
    async def init_tables() -> None:
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(init_tables())
    yield


@pytest.fixture(scope="module", autouse=True)
def cleanup_test_db() -> None:
    """Cleanup temporary database after module completes."""
    yield
    asyncio.run(test_engine.dispose())
    tmp_db_path.unlink(missing_ok=True)


def create_mock_llm_client() -> MagicMock:
    """Helper creating a mock LLMClient returning valid classification & extraction responses."""
    mock_llm = MagicMock(spec=LLMClient)
    classifier_json = '{"is_experience": true, "type": "GOAL", "importance": 0.9, "confidence": 0.95}'
    extractor_json = '{"content": "Reach 30 LPA backend engineer", "domain": "career", "confidence": 0.95}'

    async def mock_generate(messages, **kwargs):
        sys_prompt = messages[0].content if messages else ""
        if "Personal Experience Classifier" in sys_prompt:
            return LLMResponse(content=classifier_json, provider="gemini", model="gemini-3.5-flash", latency_ms=5.0)
        elif "Personal Experience Extractor" in sys_prompt:
            return LLMResponse(content=extractor_json, provider="gemini", model="gemini-3.5-flash", latency_ms=5.0)
        return LLMResponse(content="Response", provider="gemini", model="gemini-3.5-flash", latency_ms=5.0)

    mock_llm.generate_response = AsyncMock(side_effect=mock_generate)
    return mock_llm


@pytest.mark.asyncio
async def test_5a_background_embedding_succeeds() -> None:
    """Requirement 5A: Experience promoted -> Embedding succeeds -> Experience has embedding, embedding_model, status=COMPLETED, embedded_at."""
    user_id = uuid.uuid4()
    mock_llm = create_mock_llm_client()
    mock_provider = MockEmbeddingProvider(dimensions=1536)

    processor = SQLAlchemyBackgroundExperienceProcessor(
        session_factory=test_session_factory,
        llm_client=mock_llm,
        embedding_provider=mock_provider,
    )

    async with test_session_factory() as session:
        conv = Conversation(id=uuid.uuid4(), user_id=user_id)
        session.add(conv)
        msg = Message(id=uuid.uuid4(), conversation_id=conv.id, role="user", content="I want to reach 30 LPA backend engineer.")
        session.add(msg)
        await session.commit()

        await processor.process_background_promotion(msg, user_id=user_id)

    async with test_session_factory() as verify_session:
        stmt = select(ExperienceModel).where(ExperienceModel.source_message_id == msg.id)
        res = await verify_session.execute(stmt)
        exp = res.scalar_one_or_none()

        assert exp is not None
        assert exp.content == "Reach 30 LPA backend engineer"
        assert exp.embedding_status == "COMPLETED"
        assert exp.embedding_model == "gemini-embedding-001"
        assert exp.embedding is not None
        assert len(exp.embedding) == 1536
        assert exp.embedded_at is not None


@pytest.mark.asyncio
async def test_5b_background_embedding_provider_fails_safely() -> None:
    """Requirement 5B: Experience promoted -> Provider fails -> Experience still exists with embedding_status=FAILED."""
    user_id = uuid.uuid4()
    mock_llm = create_mock_llm_client()
    failing_provider = MockEmbeddingProvider(should_fail=True)

    processor = SQLAlchemyBackgroundExperienceProcessor(
        session_factory=test_session_factory,
        llm_client=mock_llm,
        embedding_provider=failing_provider,
    )

    async with test_session_factory() as session:
        conv = Conversation(id=uuid.uuid4(), user_id=user_id)
        session.add(conv)
        msg = Message(id=uuid.uuid4(), conversation_id=conv.id, role="user", content="I want to reach 30 LPA backend engineer.")
        session.add(msg)
        await session.commit()

        await processor.process_background_promotion(msg, user_id=user_id)

    async with test_session_factory() as verify_session:
        stmt = select(ExperienceModel).where(ExperienceModel.source_message_id == msg.id)
        res = await verify_session.execute(stmt)
        exp = res.scalar_one_or_none()

        # Experience remains persisted!
        assert exp is not None
        assert exp.content == "Reach 30 LPA backend engineer"
        assert exp.embedding_status == "FAILED"
        assert exp.embedding is None


@pytest.mark.asyncio
async def test_5c_background_embedding_dimension_mismatch_fails_safely() -> None:
    """Requirement 5C: Embedding returns incorrect dimension -> Experience remains, vector is NOT persisted, status=FAILED."""
    user_id = uuid.uuid4()
    mock_llm = create_mock_llm_client()

    mock_provider = MagicMock(spec=MockEmbeddingProvider)
    mock_provider.model_name = "gemini-embedding-001"
    mock_provider.dimensions = 1536
    # Provider returns wrong vector length 512
    mock_provider.embed = AsyncMock(return_value=[0.1] * 512)

    processor = SQLAlchemyBackgroundExperienceProcessor(
        session_factory=test_session_factory,
        llm_client=mock_llm,
        embedding_provider=mock_provider,
    )

    async with test_session_factory() as session:
        conv = Conversation(id=uuid.uuid4(), user_id=user_id)
        session.add(conv)
        msg = Message(id=uuid.uuid4(), conversation_id=conv.id, role="user", content="I want to reach 30 LPA backend engineer.")
        session.add(msg)
        await session.commit()

        await processor.process_background_promotion(msg, user_id=user_id)

    async with test_session_factory() as verify_session:
        stmt = select(ExperienceModel).where(ExperienceModel.source_message_id == msg.id)
        res = await verify_session.execute(stmt)
        exp = res.scalar_one_or_none()

        # Experience remains persisted!
        assert exp is not None
        assert exp.embedding_status == "FAILED"
        # Vector is NOT persisted!
        assert exp.embedding is None


@pytest.mark.asyncio
async def test_5d_embedding_disabled_does_not_call_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requirement 5D: Embedding disabled -> Experience is created -> Embedding provider is NOT called."""
    settings = get_settings()
    monkeypatch.setattr(settings, "embedding_enabled", False)

    user_id = uuid.uuid4()
    mock_llm = create_mock_llm_client()

    mock_provider = MagicMock(spec=MockEmbeddingProvider)
    mock_provider.embed = AsyncMock()

    processor = SQLAlchemyBackgroundExperienceProcessor(
        session_factory=test_session_factory,
        llm_client=mock_llm,
        embedding_provider=mock_provider,
    )

    async with test_session_factory() as session:
        conv = Conversation(id=uuid.uuid4(), user_id=user_id)
        session.add(conv)
        msg = Message(id=uuid.uuid4(), conversation_id=conv.id, role="user", content="I want to reach 30 LPA backend engineer.")
        session.add(msg)
        await session.commit()

        await processor.process_background_promotion(msg, user_id=user_id)

    async with test_session_factory() as verify_session:
        stmt = select(ExperienceModel).where(ExperienceModel.source_message_id == msg.id)
        res = await verify_session.execute(stmt)
        exp = res.scalar_one_or_none()

        assert exp is not None
        # Provider MUST NOT be called!
        mock_provider.embed.assert_not_called()
        assert exp.embedding_status == "PENDING"
