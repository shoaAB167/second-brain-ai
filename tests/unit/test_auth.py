from datetime import timedelta
import uuid

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from personal_ai.application.auth.auth_service import AuthService
from personal_ai.core.auth import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from personal_ai.core.exceptions import AppException
from personal_ai.db.models import Base
from personal_ai.db.repositories.sqlalchemy_user_repository import SQLAlchemyUserRepository
from personal_ai.domain.user.models import LoginRequest, RegisterRequest


@pytest_asyncio.fixture
async def db_session() -> AsyncSession:
    """Fixture providing an in-memory SQLite async session."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    await engine.dispose()


@pytest.mark.asyncio
async def test_user_registration_succeeds(db_session: AsyncSession) -> None:
    """1. Verify user registration succeeds and returns safe user model."""
    repo = SQLAlchemyUserRepository(session=db_session)
    service = AuthService(user_repo=repo)

    request = RegisterRequest(email="testuser@example.com", password="securepassword123")
    response = await service.register_user(request)

    assert response.id is not None
    assert response.email == "testuser@example.com"
    assert response.created_at is not None


@pytest.mark.asyncio
async def test_password_is_hashed_and_plain_never_stored(db_session: AsyncSession) -> None:
    """2 & 3. Verify password is hashed with bcrypt and plaintext password is never stored."""
    repo = SQLAlchemyUserRepository(session=db_session)
    service = AuthService(user_repo=repo)

    raw_password = "securepassword123"
    request = RegisterRequest(email="hashed@example.com", password=raw_password)
    response = await service.register_user(request)

    user_in_db = await repo.get_user_by_id(response.id)
    assert user_in_db is not None
    assert user_in_db.password_hash != raw_password
    assert user_in_db.password_hash.startswith("$2b$") or user_in_db.password_hash.startswith("$2a$")
    assert verify_password(raw_password, user_in_db.password_hash) is True


@pytest.mark.asyncio
async def test_duplicate_email_registration_rejected(db_session: AsyncSession) -> None:
    """4. Verify registration with duplicate email address is rejected with HTTP 400."""
    repo = SQLAlchemyUserRepository(session=db_session)
    service = AuthService(user_repo=repo)

    request = RegisterRequest(email="duplicate@example.com", password="securepassword123")
    await service.register_user(request)

    with pytest.raises(AppException) as exc_info:
        await service.register_user(request)

    assert exc_info.value.status_code == 400
    assert "already exists" in exc_info.value.message


@pytest.mark.asyncio
async def test_too_short_password_rejected(db_session: AsyncSession) -> None:
    """6. Verify password shorter than 8 characters is rejected."""
    repo = SQLAlchemyUserRepository(session=db_session)
    service = AuthService(user_repo=repo)

    with pytest.raises(Exception) as exc_info:
        request = RegisterRequest(email="shortpw@example.com", password="short")
        await service.register_user(request)

    assert "8" in str(exc_info.value) or "at least" in str(exc_info.value)


@pytest.mark.asyncio
async def test_login_with_correct_credentials_returns_jwt(db_session: AsyncSession) -> None:
    """7 & 8. Verify login with correct credentials returns valid JWT token."""
    repo = SQLAlchemyUserRepository(session=db_session)
    service = AuthService(user_repo=repo)

    reg_req = RegisterRequest(email="authlogin@example.com", password="securepassword123")
    user_res = await service.register_user(reg_req)

    login_req = LoginRequest(email="authlogin@example.com", password="securepassword123")
    token_res = await service.login_user(login_req)

    assert token_res.access_token is not None
    assert token_res.token_type == "bearer"

    decoded_id = decode_access_token(token_res.access_token)
    assert decoded_id == user_res.id


@pytest.mark.asyncio
async def test_login_with_incorrect_password_returns_generic_401(db_session: AsyncSession) -> None:
    """9. Verify login with wrong password returns generic 401 Unauthorized."""
    repo = SQLAlchemyUserRepository(session=db_session)
    service = AuthService(user_repo=repo)

    reg_req = RegisterRequest(email="wrongpw@example.com", password="securepassword123")
    await service.register_user(reg_req)

    login_req = LoginRequest(email="wrongpw@example.com", password="wrongpassword")
    with pytest.raises(AppException) as exc_info:
        await service.login_user(login_req)

    assert exc_info.value.status_code == 401
    assert "Invalid email or password" in exc_info.value.message


@pytest.mark.asyncio
async def test_login_with_unknown_email_returns_generic_401(db_session: AsyncSession) -> None:
    """10. Verify login with non-existent email returns generic 401 Unauthorized."""
    repo = SQLAlchemyUserRepository(session=db_session)
    service = AuthService(user_repo=repo)

    login_req = LoginRequest(email="nonexistent@example.com", password="somepassword")
    with pytest.raises(AppException) as exc_info:
        await service.login_user(login_req)

    assert exc_info.value.status_code == 401
    assert "Invalid email or password" in exc_info.value.message


def test_invalid_and_expired_jwt_raises_401() -> None:
    """11 & 12. Verify invalid and expired JWT tokens raise HTTP 401 AppException."""
    user_id = uuid.uuid4()

    # Expired token test
    expired_token = create_access_token(user_id=user_id, expires_delta=timedelta(seconds=-10))
    with pytest.raises(AppException) as exc_expired:
        decode_access_token(expired_token)
    assert exc_expired.value.status_code == 401

    # Malformed token test
    with pytest.raises(AppException) as exc_invalid:
        decode_access_token("invalid.jwt.token")
    assert exc_invalid.value.status_code == 401
