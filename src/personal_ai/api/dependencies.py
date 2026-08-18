from typing import Optional
import uuid

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from personal_ai.application.auth.auth_service import AuthService
from personal_ai.core.auth import decode_access_token
from personal_ai.core.exceptions import AppException
from personal_ai.db.repositories.sqlalchemy_conversation_repository import (
    SQLAlchemyConversationRepository,
)
from personal_ai.db.repositories.sqlalchemy_user_repository import (
    SQLAlchemyUserRepository,
)
from personal_ai.db.session import get_db_session
from personal_ai.llm import LLMClient, get_llm_client
from personal_ai.services.chat_service import ChatService

security = HTTPBearer(auto_error=False)


async def get_user_repository(
    session: AsyncSession = Depends(get_db_session),
) -> SQLAlchemyUserRepository:
    """Dependency providing UserRepository instance."""
    return SQLAlchemyUserRepository(session=session)


async def get_auth_service(
    user_repo: SQLAlchemyUserRepository = Depends(get_user_repository),
) -> AuthService:
    """Dependency providing AuthService instance."""
    return AuthService(user_repo=user_repo)


async def get_chat_service(
    session: AsyncSession = Depends(get_db_session),
    llm_client: LLMClient = Depends(get_llm_client),
) -> ChatService:
    """Dependency providing ChatService with database session and LLM client."""
    conversation_repo = SQLAlchemyConversationRepository(session=session)
    return ChatService(
        llm_client=llm_client,
        conversation_repo=conversation_repo,
    )


async def get_current_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> uuid.UUID:
    """FastAPI dependency to extract and validate authenticated user_id from Bearer JWT.

    Args:
        credentials: Captured Authorization Bearer credentials header.

    Returns:
        uuid.UUID: The authenticated user's unique UUID.

    Raises:
        AppException(401): If authorization header is missing, invalid, or expired.
    """
    if not credentials or not credentials.credentials:
        raise AppException(
            message="Authentication required. Please provide a valid Bearer token.",
            status_code=401,
        )

    return decode_access_token(credentials.credentials)
