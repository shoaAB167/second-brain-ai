import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from personal_ai.db.models import Base
from personal_ai.db.session import get_db_session
from personal_ai.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setup async in-memory SQLite database session override for auth integration tests."""
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


def test_register_api_success() -> None:
    """Verify POST /api/v1/auth/register creates user and returns 201 Created without password_hash."""
    payload = {
        "email": "registerapi@example.com",
        "password": "securepassword123",
    }
    response = client.post("/api/v1/auth/register", json=payload)

    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["email"] == "registerapi@example.com"
    assert "password" not in data
    assert "password_hash" not in data


def test_register_api_duplicate_email_rejected() -> None:
    """Verify POST /api/v1/auth/register rejects duplicate email with HTTP 400."""
    payload = {
        "email": "dupapi@example.com",
        "password": "securepassword123",
    }
    client.post("/api/v1/auth/register", json=payload)

    response = client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 400
    assert "already exists" in response.text.lower()


def test_login_api_success_returns_jwt() -> None:
    """Verify POST /api/v1/auth/login returns JWT bearer access token."""
    reg_payload = {
        "email": "loginapi@example.com",
        "password": "securepassword123",
    }
    client.post("/api/v1/auth/register", json=reg_payload)

    login_payload = {
        "email": "loginapi@example.com",
        "password": "securepassword123",
    }
    response = client.post("/api/v1/auth/login", json=login_payload)

    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_login_api_invalid_credentials_returns_generic_401() -> None:
    """Verify POST /api/v1/auth/login returns generic 401 Unauthorized for bad credentials."""
    login_payload = {
        "email": "unknown@example.com",
        "password": "wrongpassword",
    }
    response = client.post("/api/v1/auth/login", json=login_payload)

    assert response.status_code == 401
    assert "invalid email or password" in response.text.lower()
