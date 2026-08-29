import asyncio
from pathlib import Path
import tempfile
import uuid
from unittest.mock import AsyncMock, MagicMock
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from personal_ai.api.dependencies import get_llm_client
from personal_ai.api.routers.memories import get_embedding_provider
from personal_ai.core.auth import create_access_token
from personal_ai.db.models import Base, ExperienceModel, User
from personal_ai.db.session import get_db_session
from personal_ai.infrastructure.embedding import MockEmbeddingProvider
from personal_ai.llm import LLMClient
from personal_ai.llm.models import LLMMessage, LLMResponse
from personal_ai.main import app

client = TestClient(app)

tmp_db_path = Path(tempfile.gettempdir()) / f"test_mem_chat_api_{uuid.uuid4().hex}.db"
test_engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_db_path.as_posix()}", echo=False)
test_session_factory = async_sessionmaker(bind=test_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture(autouse=True)
def setup_mem_chat_api_db() -> None:
    """Setup async SQLite database session and dependency overrides for integration tests."""
    async def init_tables() -> None:
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(init_tables())

    async def override_get_db_session() -> AsyncSession:
        async with test_session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_embedding_provider] = lambda: MockEmbeddingProvider()

    yield
    app.dependency_overrides.clear()


@pytest.fixture(scope="module", autouse=True)
def cleanup_tmp_db() -> None:
    """Cleanup temporary database file after test module completes."""
    yield
    asyncio.run(test_engine.dispose())
    tmp_db_path.unlink(missing_ok=True)


def create_real_test_user(email: str = "memchatuser@example.com") -> tuple[uuid.UUID, dict[str, str]]:
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


def test_chat_api_augments_llm_with_relevant_user_memories() -> None:
    """Requirement PR #12: POST /api/v1/chat retrieves relevant user memories and augments LLM context."""
    user_id, headers = create_real_test_user("user_mem_aug@example.com")
    provider = MockEmbeddingProvider()

    # 1. Seed user experience in DB with completed embedding
    async def seed_memory() -> None:
        career_vec = await provider.embed("Reach a salary of 30 LPA")
        async with test_session_factory() as session:
            exp = ExperienceModel(
                id=uuid.uuid4(),
                user_id=user_id,
                content="Reach a salary of 30 LPA",
                type="GOAL",
                domain="career",
                source="CHAT",
                status="RECEIVED",
                embedding=career_vec,
                embedding_model="gemini-embedding-001",
                embedding_status="COMPLETED",
            )
            session.add(exp)
            await session.commit()

    asyncio.run(seed_memory())

    # 2. Mock LLMClient to inspect received messages
    captured_messages: list[LLMMessage] = []
    mock_llm = MagicMock(spec=LLMClient)

    async def fake_generate(messages: list[LLMMessage], **kwargs) -> LLMResponse:
        nonlocal captured_messages
        captured_messages = list(messages)
        return LLMResponse(
            content="To achieve 30 LPA, focus on architecture and system design.",
            provider="gemini",
            model="gemini-3.5-flash",
            latency_ms=20.0,
        )

    mock_llm.generate_response = AsyncMock(side_effect=fake_generate)
    app.dependency_overrides[get_llm_client] = lambda: mock_llm

    # 3. Send chat message
    payload = {"message": "Reach a salary of 30 LPA"}
    response = client.post("/api/v1/chat", json=payload, headers=headers)

    assert response.status_code == 200, response.text
    data = response.json()
    assert "30 LPA" in data["response"]
    assert "conversation_id" in data

    # 4. Verify LLM received <user_memory> in the system prompt
    assert len(captured_messages) >= 2
    system_msg = captured_messages[0]
    assert system_msg.role == "system"
    assert "<user_memory>" in system_msg.content
    assert "Reach a salary of 30 LPA" in system_msg.content
    assert "Type: GOAL" in system_msg.content
    assert "Domain: career" in system_msg.content


def test_chat_api_user_isolation_for_memories() -> None:
    """Requirement 17E: User B querying does not receive User A's memories in LLM prompt."""
    user_a_id, _ = create_real_test_user("user_a_mem@example.com")
    user_b_id, headers_b = create_real_test_user("user_b_mem@example.com")
    provider = MockEmbeddingProvider()

    # Seed User A memory only
    async def seed_memory_a() -> None:
        secret_vec = await provider.embed("User A secret financial plan")
        async with test_session_factory() as session:
            exp_a = ExperienceModel(
                id=uuid.uuid4(),
                user_id=user_a_id,
                content="User A secret financial plan",
                type="FACT",
                source="CHAT",
                status="RECEIVED",
                embedding=secret_vec,
                embedding_model="gemini-embedding-001",
                embedding_status="COMPLETED",
            )
            session.add(exp_a)
            await session.commit()

    asyncio.run(seed_memory_a())

    captured_messages: list[LLMMessage] = []
    mock_llm = MagicMock(spec=LLMClient)

    async def fake_generate(messages: list[LLMMessage], **kwargs) -> LLMResponse:
        nonlocal captured_messages
        captured_messages = list(messages)
        return LLMResponse(content="Answer for user B", provider="gemini", model="gemini-3.5-flash", latency_ms=10.0)

    mock_llm.generate_response = AsyncMock(side_effect=fake_generate)
    app.dependency_overrides[get_llm_client] = lambda: mock_llm

    payload = {"message": "User A secret financial plan"}
    response = client.post("/api/v1/chat", json=payload, headers=headers_b)

    assert response.status_code == 200, response.text
    # LLM system prompt for User B MUST NOT contain <user_memory> with User A's data
    for msg in captured_messages:
        if msg.role == "system":
            assert "<user_memory>" not in msg.content
