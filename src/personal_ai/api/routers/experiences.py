import uuid
from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from personal_ai.api.dependencies import get_current_user_id, get_db_session
from personal_ai.application.experience import RecordExperience
from personal_ai.db.repositories import SQLAlchemyExperienceRepository
from personal_ai.domain.experience import ExperienceRepository
from personal_ai.models.experience import (
    RecordExperienceRequest,
    RecordExperienceResponse,
)

router = APIRouter()


def get_experience_repository(
    session: AsyncSession = Depends(get_db_session),
) -> ExperienceRepository:
    """Dependency provider constructing concrete SQLAlchemyExperienceRepository as ExperienceRepository abstraction."""
    return SQLAlchemyExperienceRepository(session=session)


def get_record_experience_use_case(
    repository: ExperienceRepository = Depends(get_experience_repository),
) -> RecordExperience:
    """Dependency provider injecting repository abstraction into RecordExperience use case."""
    return RecordExperience(repository=repository)


@router.post(
    "/experiences",
    response_model=RecordExperienceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record Raw Experience",
    description="Records an uninterpreted raw user experience observation for the authenticated user. Returns HTTP 201 Created.",
)
async def record_experience(
    request: RecordExperienceRequest,
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    use_case: RecordExperience = Depends(get_record_experience_use_case),
) -> RecordExperienceResponse:
    """Execute RecordExperience use case returning HTTP 201 Created response."""
    created_experience = await use_case.execute(
        content=request.content,
        source=request.source,
        user_id=str(current_user_id),
    )

    return RecordExperienceResponse(
        experienceId=created_experience.id,
        status=created_experience.status.value,
        message="Experience recorded successfully.",
    )
