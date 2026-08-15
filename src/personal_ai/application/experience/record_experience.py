from typing import Optional, Union

from personal_ai.core.exceptions import AppException
from personal_ai.domain.experience import (
    Experience,
    ExperienceRepository,
    ExperienceSource,
    ExperienceStatus,
)


class RecordExperience:
    """Application use case for recording a raw, uninterpreted life Experience observation.

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
    ) -> Experience:
        """Execute the RecordExperience use case.

        Args:
            content: Raw user-provided experience text.
            source: Source channel (e.g. CHAT, FILE).
            user_id: Optional user identifier.

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
                status=ExperienceStatus.RECEIVED,
            )
        except ValueError as exc:
            raise AppException(message=str(exc), status_code=400) from exc

        return await self._repository.create(experience)
