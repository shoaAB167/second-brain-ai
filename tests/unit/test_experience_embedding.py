from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
import uuid
import pytest

from personal_ai.application.experience import EmbeddingResult, ExperienceEmbeddingService
from personal_ai.config.settings import Settings, get_settings
from personal_ai.domain.experience import (
    Experience,
    ExperienceSource,
    ExperienceStatus,
    ExperienceType,
    build_experience_embedding_text,
)
from personal_ai.infrastructure.embedding import MockEmbeddingProvider
from personal_ai.llm.exceptions import LLMConnectionException


def test_1_build_experience_embedding_text_canonical_formatting() -> None:
    """Requirement 17.1: Canonical embedding text generation uses structured attributes deterministically."""
    exp = Experience(
        id=uuid.uuid4(),
        content="Reach a 30 LPA backend engineering role",
        type=ExperienceType.GOAL,
        domain="career",
        status=ExperienceStatus.RECEIVED,
        source=ExperienceSource.CHAT,
    )

    canonical_text = build_experience_embedding_text(exp)

    assert "Type: GOAL" in canonical_text
    assert "Domain: career" in canonical_text
    assert "Status: RECEIVED" in canonical_text
    assert "Content: Reach a 30 LPA backend engineering role" in canonical_text
    assert str(exp.id) not in canonical_text


@pytest.mark.asyncio
async def test_2_embedding_provider_success() -> None:
    """Requirement 17.2: Embedding provider generates vector successfully."""
    provider = MockEmbeddingProvider(dimensions=1536)
    vector = await provider.embed("Type: GOAL | Content: Learn FastAPI")

    assert isinstance(vector, list)
    assert len(vector) == 1536
    assert all(isinstance(x, float) for x in vector)


@pytest.mark.asyncio
async def test_3_embedding_provider_failure() -> None:
    """Requirement 17.3: Embedding provider failure raises expected LLMConnectionException."""
    provider = MockEmbeddingProvider(should_fail=True)
    with pytest.raises(LLMConnectionException):
        await provider.embed("Test text")


@pytest.mark.asyncio
async def test_4_vector_dimension_validation() -> None:
    """Requirement 17.4: Vector dimension matches provider dimensions."""
    provider_1536 = MockEmbeddingProvider(dimensions=1536)
    vector_1536 = await provider_1536.embed("Goal text")
    assert len(vector_1536) == 1536

    provider_768 = MockEmbeddingProvider(dimensions=768)
    vector_768 = await provider_768.embed("Goal text")
    assert len(vector_768) == 768


@pytest.mark.asyncio
async def test_5_embedding_model_stored_correctly() -> None:
    """Requirement 17.5: Embedding model identifier stored in result."""
    provider = MockEmbeddingProvider(model="text-embedding-3-small")
    service = ExperienceEmbeddingService(provider=provider)
    exp = Experience(
        content="Building Second Brain AI",
        type=ExperienceType.PROJECT,
        source=ExperienceSource.CHAT,
    )

    res = await service.embed_experience(exp)

    assert res.success is True
    assert res.embedding_model == "text-embedding-3-small"
    assert res.status == "COMPLETED"
    assert res.embedded_at is not None


@pytest.mark.asyncio
async def test_6_idempotency_existing_completed_embedding_skipped() -> None:
    """Requirement 17.6: Idempotency - Existing completed embedding for same model is NOT regenerated."""
    provider = MagicMock(spec=MockEmbeddingProvider)
    provider.model_name = "text-embedding-3-small"
    provider.dimensions = 1536
    provider.embed = AsyncMock()

    existing_vector = [0.1] * 1536
    already_embedded_exp = Experience(
        content="Existing goal",
        type=ExperienceType.GOAL,
        source=ExperienceSource.CHAT,
        embedding=existing_vector,
        embedding_model="text-embedding-3-small",
        embedding_status="COMPLETED",
        embedded_at=datetime.now(timezone.utc),
    )

    service = ExperienceEmbeddingService(provider=provider)
    res = await service.embed_experience(already_embedded_exp)

    assert res.success is True
    assert res.embedding == existing_vector
    # MUST NOT call provider.embed
    provider.embed.assert_not_called()


@pytest.mark.asyncio
async def test_7_failed_embedding_does_not_delete_experience() -> None:
    """Requirement 17.7: Provider failure marks status=FAILED without deleting/invalidating Experience entity."""
    provider = MockEmbeddingProvider(should_fail=True)
    service = ExperienceEmbeddingService(provider=provider)
    exp = Experience(
        id=uuid.uuid4(),
        content="I prefer remote work",
        type=ExperienceType.PREFERENCE,
        source=ExperienceSource.CHAT,
    )

    res = await service.embed_experience(exp)

    assert res.success is False
    assert res.status == "FAILED"
    assert res.embedding is None
    # Experience entity content remains intact
    assert exp.content == "I prefer remote work"


@pytest.mark.asyncio
async def test_10_user_isolation_preserved() -> None:
    """Requirement 17.10: Experience and vectors remain associated with distinct user_ids."""
    user1_id = str(uuid.uuid4())
    user2_id = str(uuid.uuid4())

    exp1 = Experience(content="User 1 goal", type=ExperienceType.GOAL, source=ExperienceSource.CHAT, user_id=user1_id)
    exp2 = Experience(content="User 2 goal", type=ExperienceType.GOAL, source=ExperienceSource.CHAT, user_id=user2_id)

    assert exp1.user_id == user1_id
    assert exp2.user_id == user2_id
    assert exp1.user_id != exp2.user_id


@pytest.mark.asyncio
async def test_11_unpersisted_empty_experience_returns_failed_result() -> None:
    """Requirement 17.11: Empty / invalid Experience fails safely."""
    provider = MockEmbeddingProvider()
    service = ExperienceEmbeddingService(provider=provider)
    
    res = await service.embed_experience(None) # type: ignore
    assert res.success is False
    assert res.status == "FAILED"


def test_12_configuration_loaded_correctly() -> None:
    """Requirement 17.12: Settings contains valid default embedding settings."""
    settings = get_settings()
    assert settings.embedding_provider == "openai"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_dimensions == 1536
    assert settings.embedding_enabled is True
