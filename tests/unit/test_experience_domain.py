from datetime import datetime, timezone
import uuid
import pytest

from personal_ai.domain.experience import (
    Experience,
    ExperienceSource,
    ExperienceStatus,
)


def test_experience_entity_creation_valid() -> None:
    """Verify creation of a valid Experience domain entity with default values."""
    content = "I started learning FastAPI today."
    exp = Experience(content=content, source=ExperienceSource.CHAT)

    assert isinstance(exp.id, uuid.UUID)
    assert exp.content == content
    assert exp.source == ExperienceSource.CHAT
    assert exp.status == ExperienceStatus.RECEIVED
    assert isinstance(exp.created_at, datetime)
    assert exp.created_at.tzinfo == timezone.utc


def test_experience_entity_rejects_empty_content() -> None:
    """Verify domain entity raises ValueError when content is empty."""
    with pytest.raises(ValueError, match="cannot be empty"):
        Experience(content="", source=ExperienceSource.CHAT)


def test_experience_entity_rejects_whitespace_only_content() -> None:
    """Verify domain entity raises ValueError when content is whitespace-only."""
    with pytest.raises(ValueError, match="cannot be empty"):
        Experience(content="   \n\t  ", source=ExperienceSource.CHAT)


def test_experience_entity_rejects_invalid_source() -> None:
    """Verify domain entity raises ValueError when source is invalid."""
    with pytest.raises(ValueError, match="Invalid experience source"):
        Experience(content="Valid content", source="INVALID_SOURCE_NAME")  # type: ignore[arg-type]


def test_experience_domain_isolation() -> None:
    """Verify Experience domain module does not import FastAPI or SQLAlchemy."""
    import personal_ai.domain.experience.entity as entity_module

    imported_modules = set(dir(entity_module))
    assert "FastAPI" not in imported_modules
    assert "SQLAlchemy" not in imported_modules
    assert "Base" not in imported_modules
