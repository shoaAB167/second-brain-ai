from typing import Optional
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from personal_ai.core.logger import get_logger
from personal_ai.db.models import User
from personal_ai.db.repositories.base import UserRepository

logger = get_logger(__name__)


class SQLAlchemyUserRepository(UserRepository):
    """SQLAlchemy implementation of the UserRepository interface."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with an active AsyncSession."""
        self._session = session

    async def create_user(self, email: str, password_hash: str) -> User:
        """Create and persist a new user identity."""
        user = User(
            email=email.strip().lower(),
            password_hash=password_hash,
        )
        self._session.add(user)
        await self._session.commit()
        await self._session.refresh(user)

        logger.info("Created new user [user_id=%s, email=%s]", user.id, user.email)
        return user

    async def get_user_by_id(self, user_id: uuid.UUID) -> Optional[User]:
        """Fetch user by unique UUID."""
        stmt = select(User).where(User.id == user_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Fetch user by email address (case-insensitive)."""
        normalized_email = email.strip().lower()
        stmt = select(User).where(User.email == normalized_email)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()
