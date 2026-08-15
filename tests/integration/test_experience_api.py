import asyncio
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from personal_ai.db.models import Base, ExperienceModel
from personal_ai.db.session import get_db_session
from personal_ai.main import app

client = TestClient(app)
test_engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
test_session_factory = async_sessionmaker(bind=test_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture(autouse=True)
def setup_experience_db(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_post_experience_returns_202_accepted_and_persists() -> None:
    """Verify POST /api/v1/experiences returns HTTP 202 Accepted and persists Experience."""
    raw_content = "I started learning FastAPI today."
    payload = {
        "content": raw_content,
        "source": "CHAT",
    }

    response = client.post("/api/v1/experiences", json=payload)

    assert response.status_code == 202
    data = response.json()

    assert "experienceId" in data
    exp_id = uuid.UUID(data["experienceId"])
    assert data["status"] == "RECEIVED"
    assert data["message"] == "Experience recorded successfully."

    # Verify persistence in database
    async def verify_db() -> None:
        async with test_session_factory() as session:
            stmt = select(ExperienceModel).where(ExperienceModel.id == exp_id)
            res = await session.execute(stmt)
            model = res.scalar_one_or_none()

            assert model is not None
            assert model.content == raw_content
            assert model.source == "CHAT"
            assert model.status == "RECEIVED"
            assert model.created_at is not None

    asyncio.run(verify_db())


def test_post_experience_empty_content_rejected() -> None:
    """Verify POST /api/v1/experiences rejects empty content with 422 Unprocessable Entity."""
    payload = {
        "content": "",
        "source": "CHAT",
    }
    response = client.post("/api/v1/experiences", json=payload)
    assert response.status_code in (400, 422)


def test_post_experience_whitespace_content_rejected() -> None:
    """Verify POST /api/v1/experiences rejects whitespace-only content."""
    payload = {
        "content": "   \n\t ",
        "source": "CHAT",
    }
    response = client.post("/api/v1/experiences", json=payload)
    assert response.status_code in (400, 422)


def test_post_experience_invalid_source_rejected() -> None:
    """Verify POST /api/v1/experiences rejects invalid source."""
    payload = {
        "content": "Valid content",
        "source": "UNSUPPORTED_SOURCE",
    }
    response = client.post("/api/v1/experiences", json=payload)
    assert response.status_code in (400, 422)
