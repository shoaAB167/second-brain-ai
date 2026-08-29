from dataclasses import dataclass
from datetime import datetime
import time
from typing import List, Optional
import uuid

from personal_ai.config.settings import get_settings
from personal_ai.core.exceptions import AppException
from personal_ai.core.logger import get_logger
from personal_ai.domain.experience import ExperienceRepository
from personal_ai.infrastructure.embedding.provider import EmbeddingProvider

logger = get_logger(__name__)


@dataclass
class MemorySearchResult:
    """Clean domain representation of a retrieved Experience memory matching a query."""

    experience_id: uuid.UUID
    type: Optional[str]
    content: str
    domain: Optional[str]
    status: str
    similarity: float
    source_message_id: Optional[uuid.UUID]
    created_at: datetime


class MemoryRetrievalService:
    """Application service for performing user-scoped semantic memory retrieval over indexed Experiences."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        experience_repo: ExperienceRepository,
    ) -> None:
        """Initialize MemoryRetrievalService.

        Args:
            embedding_provider: EmbeddingProvider implementation for query vector generation.
            experience_repo: ExperienceRepository port interface.
        """
        self._provider = embedding_provider
        self._experience_repo = experience_repo

    async def search(
        self,
        user_id: uuid.UUID,
        query: str,
        limit: int = 5,
        threshold: Optional[float] = None,
    ) -> List[MemorySearchResult]:
        """Perform semantic similarity search over the authenticated user's experiences.

        Enforces strict limit validation (1 <= limit <= 20) and model consistency invariants.

        Args:
            user_id: The authenticated user's UUID.
            query: Natural-language search query.
            limit: Maximum results to return (1-20, default 5).
            threshold: Optional minimum cosine similarity threshold in [-1.0, 1.0].

        Returns:
            List[MemorySearchResult]: Ranked list of matching memory search results.

        Raises:
            AppException: If query, limit, threshold, or model compatibility invariants are violated.
        """
        if not query or not query.strip():
            raise AppException(message="Search query cannot be empty.", status_code=400)

        # Requirement 3: Explicit limit validation (no silent clamping)
        if not isinstance(limit, int) or limit < 1 or limit > 20:
            raise AppException(
                message="Result limit must be an integer between 1 and 20.",
                status_code=400,
            )

        if threshold is not None and not (-1.0 <= threshold <= 1.0):
            raise AppException(
                message="Similarity threshold must be between -1.0 and 1.0.",
                status_code=400,
            )

        # Requirement 4: Embedding model consistency invariant validation
        settings = get_settings()
        if self._provider.model_name != settings.embedding_model:
            logger.error(
                "Embedding model mismatch: provider=%s, settings=%s",
                self._provider.model_name,
                settings.embedding_model,
            )
            raise AppException(
                message=f"Query embedding model '{self._provider.model_name}' is incompatible with configured model '{settings.embedding_model}'.",
                status_code=500,
            )

        start_time = time.perf_counter()

        # Step 1: Generate query embedding vector using configured provider
        try:
            query_vector = await self._provider.embed(query.strip())
        except Exception as exc:
            logger.error("Query embedding generation failed: %s", exc)
            raise AppException(
                message=f"Failed to generate query embedding: {str(exc)}",
                status_code=502,
            )

        # Step 2: Validate vector dimension invariant
        if len(query_vector) != self._provider.dimensions or len(query_vector) != settings.embedding_dimensions:
            logger.error(
                "Query vector dimension mismatch [expected=%d, got=%d]",
                self._provider.dimensions,
                len(query_vector),
            )
            raise AppException(
                message="Query vector dimension mismatch with configured embedding model.",
                status_code=500,
            )

        # Step 3: Execute user-scoped database vector similarity search
        scored_experiences = await self._experience_repo.search_by_vector(
            user_id=user_id,
            query_vector=query_vector,
            limit=limit,
            threshold=threshold,
        )

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "Retrieved memories [user_id=%s, count=%d, limit=%d, duration_ms=%.1f]",
            user_id,
            len(scored_experiences),
            limit,
            duration_ms,
        )

        # Step 4: Map to clean application result models
        results: List[MemorySearchResult] = []
        for exp, similarity in scored_experiences:
            results.append(
                MemorySearchResult(
                    experience_id=exp.id,
                    type=exp.type.value if exp.type and hasattr(exp.type, "value") else (str(exp.type) if exp.type else None),
                    content=exp.content,
                    domain=exp.domain,
                    status=exp.status.value if hasattr(exp.status, "value") else str(exp.status),
                    similarity=round(similarity, 4),
                    source_message_id=exp.source_message_id,
                    created_at=exp.created_at,
                )
            )

        return results
