from typing import Optional, Union
import uuid

from personal_ai.core.exceptions import AppException
from personal_ai.domain.experience import (
    Experience,
    ExperienceImportance,
    ExperienceLifecycle,
    ExperienceRepository,
    ExperienceSource,
    ExperienceStatus,
    ExperienceType,
)


class RecordExperience:
    """Application use case for recording a structured or raw life Experience observation.

    Orchestrates validation, domain object creation, status assignment (RECEIVED),
    and persistence through the abstract ExperienceRepository interface.

    Does NOT depend on FastAPI, HTTP status codes, SQLAlchemy, or LLM providers.
    """

    def __init__(self, repository: ExperienceRepository) -> None:
        """Initialize use case with abstract ExperienceRepository interface.

        Args:
            repository: Abstract repository implementation for experience persistence.
        """
        self._repository = repository

    async def execute(
        self,
        content: str,
        source: Union[ExperienceSource, str],
        user_id: Optional[str] = None,
        source_message_id: Optional[uuid.UUID] = None,
        type: Optional[Union[ExperienceType, str]] = None,
        domain: Optional[str] = None,
        importance: Optional[Union[ExperienceImportance, str]] = None,
        lifecycle: Optional[Union[ExperienceLifecycle, str]] = None,
        extraction_confidence: Optional[float] = None,
    ) -> Experience:
        """Execute the RecordExperience use case.

        Args:
            content: Raw or extracted user experience text.
            source: Source channel (e.g. CHAT, FILE).
            user_id: Optional user identifier.
            source_message_id: Optional UUID of the originating user Message.
            type: Optional ExperienceType or category string.
            domain: Optional categorical domain string.
            importance: Optional ExperienceImportance (LOW, MEDIUM, HIGH).
            lifecycle: Optional ExperienceLifecycle (STABLE, RECURRING, TEMPORARY, TIME_BOUND).
            extraction_confidence: Optional extractor confidence float score.

        Returns:
            Experience: The created and persisted domain Experience entity.

        Raises:
            AppException: If content or source validation fails.
        """
        if not isinstance(content, str) or not content or not content.strip():
            raise AppException(
                message="Experience content cannot be empty or whitespace-only.",
                status_code=400,
            )

        try:
            exp_source = (
                source
                if isinstance(source, ExperienceSource)
                else ExperienceSource(str(source).upper())
            )
        except ValueError:
            raise AppException(
                message=f"Invalid experience source: '{source}'.",
                status_code=400,
            )

        try:
            experience = Experience(
                content=content,
                source=exp_source,
                user_id=user_id,
                source_message_id=source_message_id,
                type=type,
                domain=domain,
                importance=importance or ExperienceImportance.MEDIUM,
                lifecycle=lifecycle or ExperienceLifecycle.STABLE,
                extraction_confidence=extraction_confidence,
                status=ExperienceStatus.RECEIVED,
            )
        except ValueError as exc:
            raise AppException(message=str(exc), status_code=400) from exc

        return await self._repository.create(experience)
