from dataclasses import dataclass
from datetime import datetime, timezone
import time
from typing import List, Optional

from personal_ai.core.logger import get_logger
from personal_ai.domain.experience import Experience, build_experience_embedding_text
from personal_ai.infrastructure.embedding.provider import EmbeddingProvider

logger = get_logger(__name__)


def utc_now() -> datetime:
    """Return current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


@dataclass
class EmbeddingResult:
    """Result container for Experience embedding operations."""

    success: bool
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    status: str = "PENDING"
    embedded_at: Optional[datetime] = None
    error: Optional[str] = None


class ExperienceEmbeddingService:
    """Application service for generating vector embeddings for Experience entities.

    Depends on abstract EmbeddingProvider interface. Enforces idempotency and fail-closed safety.
    Does NOT directly execute HTTP requests or database queries.
    """

    def __init__(self, provider: EmbeddingProvider) -> None:
        """Initialize ExperienceEmbeddingService.

        Args:
            provider: Abstract EmbeddingProvider port interface implementation.
        """
        self._provider = provider

    async def embed_experience(self, experience: Experience) -> EmbeddingResult:
        """Generate vector embedding for a structured Experience entity.

        Enforces idempotency: skips regeneration if already embedded with current provider model.
        Fails safely on provider exceptions without raising unhandled errors.

        Args:
            experience: Target Experience domain entity.

        Returns:
            EmbeddingResult: Container indicating success/failure, vector, model, and metadata.
        """
        if not experience or not experience.content:
            return EmbeddingResult(
                success=False,
                status="FAILED",
                error="Experience or content is empty.",
            )

        # Idempotency check: Skip regeneration if already indexed with current model and valid vector
        if (
            experience.embedding_status == "COMPLETED"
            and experience.embedding_model == self._provider.model_name
            and experience.embedding is not None
            and len(experience.embedding) == self._provider.dimensions
        ):
            logger.info(
                "Skipping vector embedding generation [experience_id=%s]: already indexed with model %s",
                experience.id,
                self._provider.model_name,
            )
            return EmbeddingResult(
                success=True,
                embedding=experience.embedding,
                embedding_model=experience.embedding_model,
                status="COMPLETED",
                embedded_at=experience.embedded_at,
            )

        start_time = time.perf_counter()
        canonical_text = build_experience_embedding_text(experience)

        try:
            vector = await self._provider.embed(canonical_text)
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            logger.info(
                "Vector embedding generated successfully [experience_id=%s, model=%s, dim=%d, duration_ms=%.1f]",
                experience.id,
                self._provider.model_name,
                len(vector),
                duration_ms,
            )
            return EmbeddingResult(
                success=True,
                embedding=vector,
                embedding_model=self._provider.model_name,
                status="COMPLETED",
                embedded_at=utc_now(),
            )
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                "Vector embedding generation failed safely [experience_id=%s, model=%s, duration_ms=%.1f]: %s",
                experience.id,
                self._provider.model_name,
                duration_ms,
                exc,
            )
            return EmbeddingResult(
                success=False,
                status="FAILED",
                error=str(exc),
            )
