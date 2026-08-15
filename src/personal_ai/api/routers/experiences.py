from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from personal_ai.application.experience import RecordExperience
from personal_ai.db.repositories import SQLAlchemyExperienceRepository
from personal_ai.db.session import get_db_session
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
    status_code=status.HTTP_202_ACCEPTED,
    summary="Record Raw Experience",
    description="Records an uninterpreted raw user experience observation. Returns HTTP 202 Accepted.",
)
async def record_experience(
    request: RecordExperienceRequest,
    use_case: RecordExperience = Depends(get_record_experience_use_case),
) -> RecordExperienceResponse:
    """Execute RecordExperience use case returning HTTP 202 Accepted response."""
    created_experience = await use_case.execute(
        content=request.content,
        source=request.source,
    )

    return RecordExperienceResponse(
        experienceId=created_experience.id,
        status=created_experience.status.value,
        message="Experience recorded successfully.",
    )
