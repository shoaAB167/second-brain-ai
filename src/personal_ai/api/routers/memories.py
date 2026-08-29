from typing import Optional
import uuid

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from personal_ai.api.dependencies import get_current_user_id, get_db_session
from personal_ai.application.memory import MemoryRetrievalService
from personal_ai.db.repositories import SQLAlchemyExperienceRepository
from personal_ai.infrastructure.embedding import EmbeddingProvider, OpenAIEmbeddingProvider
from personal_ai.models.memory import (
    MemorySearchResponse,
    MemorySearchResultItem,
)

router = APIRouter()


def get_embedding_provider() -> EmbeddingProvider:
    """Dependency provider constructing default OpenAIEmbeddingProvider."""
    return OpenAIEmbeddingProvider()


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


@router.get(
    "/memories/search",
    response_model=MemorySearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic Memory Search",
    description="Perform user-scoped semantic similarity search over indexed experience memories.",
)
async def search_memories(
    q: str = Query(..., min_length=1, max_length=1000, description="Natural language search query text."),
    limit: int = Query(default=5, ge=1, le=20, description="Maximum number of results to return (1-20)."),
    threshold: Optional[float] = Query(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Optional minimum cosine similarity score threshold in [-1.0, 1.0].",
    ),
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    retrieval_service: MemoryRetrievalService = Depends(get_memory_retrieval_service),
) -> MemorySearchResponse:
    """Execute user-scoped semantic memory retrieval query returning ranked matching experiences."""
    results = await retrieval_service.search(
        user_id=current_user_id,
        query=q,
        limit=limit,
        threshold=threshold,
    )

    return MemorySearchResponse(
        query=q,
        count=len(results),
        results=[
            MemorySearchResultItem(
                experience_id=item.experience_id,
                type=item.type,
                content=item.content,
                domain=item.domain,
                status=item.status,
                similarity=item.similarity,
                source_message_id=item.source_message_id,
                created_at=item.created_at,
            )
            for item in results
        ],
    )
