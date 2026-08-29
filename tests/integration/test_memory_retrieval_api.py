import asyncio
from pathlib import Path
import tempfile
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from personal_ai.api.routers.memories import get_embedding_provider
from personal_ai.core.auth import create_access_token
from personal_ai.db.models import Base, ExperienceModel, User
from personal_ai.db.session import get_db_session
from personal_ai.infrastructure.embedding import MockEmbeddingProvider
from personal_ai.main import app

client = TestClient(app)

tmp_db_path = Path(tempfile.gettempdir()) / f"test_memory_api_{uuid.uuid4().hex}.db"
test_engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_db_path.as_posix()}", echo=False)
test_session_factory = async_sessionmaker(bind=test_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture(autouse=True)
def setup_memory_api_db() -> None:
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


def create_real_test_user(email: str = "memuser@example.com") -> tuple[uuid.UUID, dict[str, str]]:
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


def test_unauthenticated_memory_search_returns_401() -> None:
    """Requirement 18I: Unauthenticated request to /memories/search returns 401 Unauthorized."""
    response = client.get("/api/v1/memories/search?q=career")
    assert response.status_code == 401


def test_authenticated_memory_search_returns_ranked_results() -> None:
    """Requirement 18A & 18B: Authenticated user searches and receives ranked semantic memory results."""
    user_id, headers = create_real_test_user("user_ranked@example.com")
    provider = MockEmbeddingProvider()

    async def seed_experiences() -> None:
        vec_career = await provider.embed("Reach 30 LPA as a backend engineer")
        vec_music = await provider.embed("Plays electric guitar in a rock band")

        async with test_session_factory() as session:
            exp1 = ExperienceModel(
                id=uuid.uuid4(),
                user_id=user_id,
                content="Reach 30 LPA as a backend engineer",
                type="GOAL",
                domain="career",
                source="CHAT",
                status="RECEIVED",
                embedding=vec_career,
                embedding_model="gemini-embedding-001",
                embedding_status="COMPLETED",
            )
            exp2 = ExperienceModel(
                id=uuid.uuid4(),
                user_id=user_id,
                content="Plays electric guitar in a rock band",
                type="PREFERENCE",
                domain="music",
                source="CHAT",
                status="RECEIVED",
                embedding=vec_music,
                embedding_model="gemini-embedding-001",
                embedding_status="COMPLETED",
            )
            session.add_all([exp1, exp2])
            await session.commit()

    asyncio.run(seed_experiences())

    response = client.get("/api/v1/memories/search?q=Reach+30+LPA+as+a+backend+engineer&limit=5", headers=headers)
    assert response.status_code == 200

    data = response.json()
    assert data["query"] == "Reach 30 LPA as a backend engineer"
    assert data["count"] == 2
    assert len(data["results"]) == 2

    top_result = data["results"][0]
    assert top_result["content"] == "Reach 30 LPA as a backend engineer"
    assert top_result["type"] == "GOAL"
    assert top_result["domain"] == "career"
    assert "experienceId" in top_result
    assert "similarity" in top_result
    # Vector embedding MUST NOT be leaked in API response
    assert "embedding" not in top_result


def test_user_isolation_strictly_enforced_in_api() -> None:
    """Requirement 18C & 20: User A cannot retrieve User B's memories under any circumstances."""
    user_a_id, headers_a = create_real_test_user("user_a@example.com")
    user_b_id, headers_b = create_real_test_user("user_b@example.com")
    provider = MockEmbeddingProvider()

    async def seed_isolated_experiences() -> None:
        vec_target = await provider.embed("Top secret financial investment strategy")

        async with test_session_factory() as session:
            exp_a = ExperienceModel(
                id=uuid.uuid4(),
                user_id=user_a_id,
                content="User A public hobby memory",
                type="FACT",
                source="CHAT",
                status="RECEIVED",
                embedding=vec_target,
                embedding_model="gemini-embedding-001",
                embedding_status="COMPLETED",
            )
            exp_b = ExperienceModel(
                id=uuid.uuid4(),
                user_id=user_b_id,
                content="User B Top secret financial investment strategy",
                type="FACT",
                source="CHAT",
                status="RECEIVED",
                embedding=vec_target,
                embedding_model="gemini-embedding-001",
                embedding_status="COMPLETED",
            )
            session.add_all([exp_a, exp_b])
            await session.commit()

    asyncio.run(seed_isolated_experiences())

    # User A queries with User B's secret exact query
    response_a = client.get(
        "/api/v1/memories/search?q=Top+secret+financial+investment+strategy",
        headers=headers_a,
    )
    assert response_a.status_code == 200
    data_a = response_a.json()

    # MUST NOT contain User B's memory!
    for result in data_a["results"]:
        assert "User B" not in result["content"]

    # User B queries and receives User B's memory
    response_b = client.get(
        "/api/v1/memories/search?q=Top+secret+financial+investment+strategy",
        headers=headers_b,
    )
    assert response_b.status_code == 200
    data_b = response_b.json()
    assert len(data_b["results"]) == 1
    assert data_b["results"][0]["content"] == "User B Top secret financial investment strategy"


def test_empty_results_when_no_matching_memories() -> None:
    """Requirement 18F: Returns empty results list when user has no memories."""
    _, headers = create_real_test_user("user_empty@example.com")
    response = client.get("/api/v1/memories/search?q=anything", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 0
    assert data["results"] == []
