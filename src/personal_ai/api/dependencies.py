from typing import Optional
import uuid

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from personal_ai.application.auth.auth_service import AuthService
from personal_ai.application.memory import MemoryRetrievalService
from personal_ai.core.auth import decode_access_token
from personal_ai.core.exceptions import AppException
from personal_ai.db.repositories.base import UserRepository
from personal_ai.db.repositories.sqlalchemy_conversation_repository import (
    SQLAlchemyConversationRepository,
)
from personal_ai.db.repositories.sqlalchemy_experience_repository import (
    SQLAlchemyExperienceRepository,
)
from personal_ai.db.repositories.sqlalchemy_user_repository import (
    SQLAlchemyUserRepository,
)
from personal_ai.db.session import get_db_session
from personal_ai.infrastructure.embedding import (
    EmbeddingProvider,
    get_embedding_provider as create_embedding_provider,
)
from personal_ai.llm import LLMClient, get_llm_client
from personal_ai.services.chat_service import ChatService

security = HTTPBearer(auto_error=False)


async def get_user_repository(
    session: AsyncSession = Depends(get_db_session),
) -> UserRepository:
    """Dependency providing UserRepository instance."""
    return SQLAlchemyUserRepository(session=session)


async def get_auth_service(
    user_repo: UserRepository = Depends(get_user_repository),
) -> AuthService:
    """Dependency providing AuthService instance."""
    return AuthService(user_repo=user_repo)


def get_embedding_provider() -> EmbeddingProvider:
    """Dependency provider constructing configured EmbeddingProvider via factory."""
    return create_embedding_provider()


def get_memory_retrieval_service(
    session: AsyncSession = Depends(get_db_session),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
) -> MemoryRetrievalService:
    """Dependency provider constructing MemoryRetrievalService with session and embedding provider."""
    repo = SQLAlchemyExperienceRepository(session=session)
    return MemoryRetrievalService(
        embedding_provider=embedding_provider,
        experience_repo=repo,
    )


async def get_chat_service(
    session: AsyncSession = Depends(get_db_session),
    llm_client: LLMClient = Depends(get_llm_client),
    retrieval_service: MemoryRetrievalService = Depends(get_memory_retrieval_service),
) -> ChatService:
    """Dependency providing ChatService with database session, LLM client, and memory retrieval service."""
    conversation_repo = SQLAlchemyConversationRepository(session=session)
    return ChatService(
        llm_client=llm_client,
        conversation_repo=conversation_repo,
        retrieval_service=retrieval_service,
    )


async def get_current_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    user_repo: UserRepository = Depends(get_user_repository),
) -> uuid.UUID:
    """FastAPI dependency to extract, validate, and verify authenticated user existence from Bearer JWT.

    Args:
        credentials: Captured Authorization Bearer credentials header.
        user_repo: Injected UserRepository abstraction.

    Returns:
        uuid.UUID: The authenticated user's unique UUID.

    Raises:
        AppException(401): If authorization token is missing, invalid, expired, or user does not exist.
    """
    if not credentials or not credentials.credentials:
        raise AppException(
            message="Authentication required. Please provide a valid Bearer token.",
            status_code=401,
        )

    user_id = decode_access_token(credentials.credentials)
    user = await user_repo.get_user_by_id(user_id)
    if not user:
        raise AppException(
            message="Invalid authentication credentials.",
            status_code=401,
        )

    return user.id
