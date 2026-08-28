from typing import Optional
import uuid
import pytest

from personal_ai.application.experience import RecordExperience
from personal_ai.core.exceptions import AppException
from personal_ai.domain.experience import (
    Experience,
    ExperienceRepository,
    ExperienceSource,
    ExperienceStatus,
)


class MockExperienceRepository(ExperienceRepository):
    """Mock implementation of ExperienceRepository for unit testing."""

    def __init__(self) -> None:
        self.created_experiences: list[Experience] = []

    async def create(self, experience: Experience) -> Experience:
        self.created_experiences.append(experience)
        return experience

    async def update(self, experience: Experience) -> Experience:
        for i, exp in enumerate(self.created_experiences):
            if exp.id == experience.id:
                self.created_experiences[i] = experience
                return experience
        self.created_experiences.append(experience)
        return experience

    async def get_by_id(self, experience_id: uuid.UUID) -> Optional[Experience]:
        for exp in self.created_experiences:
            if exp.id == experience_id:
                return exp
        return None

    async def get_by_source_message_id(self, source_message_id: uuid.UUID) -> Optional[Experience]:
        for exp in self.created_experiences:
            if exp.source_message_id == source_message_id:
                return exp
        return None


@pytest.mark.asyncio
async def test_record_experience_use_case_success() -> None:
    """Verify RecordExperience use case validates rules and persists Experience with RECEIVED status."""
    repo = MockExperienceRepository()
    use_case = RecordExperience(repository=repo)

    raw_text = "I decided to move to Bangalore."
    result = await use_case.execute(content=raw_text, source="CHAT", user_id="user_123")

    assert isinstance(result, Experience)
    assert result.content == raw_text
    assert result.source == ExperienceSource.CHAT
    assert result.status == ExperienceStatus.RECEIVED
    assert result.user_id == "user_123"

    assert len(repo.created_experiences) == 1
    assert repo.created_experiences[0].id == result.id


@pytest.mark.asyncio
async def test_record_experience_use_case_rejects_empty_content() -> None:
    """Verify use case raises AppException when content is empty."""
    repo = MockExperienceRepository()
    use_case = RecordExperience(repository=repo)

    with pytest.raises(AppException) as exc_info:
        await use_case.execute(content="   ", source="CHAT")

    assert exc_info.value.status_code == 400
    assert "cannot be empty" in exc_info.value.message
    assert len(repo.created_experiences) == 0


@pytest.mark.asyncio
async def test_record_experience_use_case_rejects_invalid_source() -> None:
    """Verify use case raises AppException when source is invalid."""
    repo = MockExperienceRepository()
    use_case = RecordExperience(repository=repo)

    with pytest.raises(AppException) as exc_info:
        await use_case.execute(content="Valid content", source="UNSUPPORTED_SOURCE")

    assert exc_info.value.status_code == 400
    assert "Invalid experience source" in exc_info.value.message
    assert len(repo.created_experiences) == 0
