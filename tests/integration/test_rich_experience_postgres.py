import uuid
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from personal_ai.application.memory.context_builder import MemoryContextBuilder
from personal_ai.application.memory.retrieval_service import MemoryRetrievalService
from personal_ai.config.settings import get_settings
from personal_ai.db.models import User
from personal_ai.db.repositories.sqlalchemy_experience_repository import SQLAlchemyExperienceRepository
from personal_ai.domain.experience import (
    EmotionalContext,
    Experience,
    ExperienceEvidenceLevel,
    ExperienceImportance,
    ExperienceLifecycle,
    ExperienceSource,
    ExperienceType,
    PersonInvolved,
)
from personal_ai.infrastructure.embedding import MockEmbeddingProvider


@pytest.mark.asyncio
async def test_postgres_schema_has_rich_experience_columns() -> None:
    """Verify PostgreSQL experiences table contains rich emotional and contextual columns."""
    settings = get_settings()
    db_url = settings.database_url.replace("postgresql://", "postgresql+asyncpg://")

    engine = create_async_engine(db_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        result = await session.execute(
            text(
                """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = 'experiences' 
                  AND column_name IN ('emotional_context', 'people_involved', 'temporal_context', 'evidence_level')
                ORDER BY column_name;
                """
            )
        )
        columns = {row[0]: (row[1], row[2], row[3]) for row in result.all()}

        assert "emotional_context" in columns, "experiences.emotional_context column must exist in PostgreSQL"
        assert "people_involved" in columns, "experiences.people_involved column must exist in PostgreSQL"
        assert "temporal_context" in columns, "experiences.temporal_context column must exist in PostgreSQL"
        assert "evidence_level" in columns, "experiences.evidence_level column must exist in PostgreSQL"

        assert columns["emotional_context"][0] in ("json", "jsonb")
        assert columns["people_involved"][0] in ("json", "jsonb")
        assert columns["temporal_context"][0] in ("character varying", "varchar", "text")
        assert columns["evidence_level"][0] in ("character varying", "varchar", "text")
        assert columns["evidence_level"][1] == "NO"  # NOT NULL

    await engine.dispose()


@pytest.mark.asyncio
async def test_postgres_rich_experience_persistence_and_retrieval() -> None:
    """Verify storing and retrieving rich experiences with emotional context in PostgreSQL."""
    settings = get_settings()
    db_url = settings.database_url.replace("postgresql://", "postgresql+asyncpg://")

    engine = create_async_engine(db_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    provider = MockEmbeddingProvider(dimensions=1536)
    test_user_id = uuid.uuid4()

    try:
        async with async_session() as session:
            # 1. Create User
            user = User(
                id=test_user_id,
                email=f"rich_user_{test_user_id.hex[:8]}@example.com",
                password_hash="test_hash",
            )
            session.add(user)
            await session.commit()

            repo = SQLAlchemyExperienceRepository(session=session)

            # 2. Add Rich Experience
            vec = await provider.embed("Failed my AI interview today and feeling anxious")
            exp = Experience(
                content="Failed an AI technical interview",
                source=ExperienceSource.CHAT,
                user_id=str(test_user_id),
                type=ExperienceType.EVENT,
                domain="career",
                importance=ExperienceImportance.HIGH,
                lifecycle=ExperienceLifecycle.TIME_BOUND,
                emotional_context=EmotionalContext(
                    emotion="anxiety",
                    intensity=0.85,
                    trigger="technical interview failure",
                    need="preparation guidance",
                    impact="questioning readiness",
                ),
                people_involved=[PersonInvolved(name="Alex", role="interviewer")],
                temporal_context="today",
                evidence_level=ExperienceEvidenceLevel.EXPLICIT_USER,
                embedding=vec,
                embedding_model=provider.model_name,
                embedding_status="COMPLETED",
            )

            saved = await repo.create(exp)
            assert saved.id is not None
            assert saved.emotional_context is not None
            assert saved.emotional_context.emotion == "anxiety"
            assert saved.emotional_context.intensity == 0.85
            assert saved.people_involved is not None
            assert saved.people_involved[0].name == "Alex"
            assert saved.evidence_level == ExperienceEvidenceLevel.EXPLICIT_USER

            # 3. Retrieve through MemoryRetrievalService
            retrieval_svc = MemoryRetrievalService(embedding_provider=provider, experience_repo=repo)
            results = await retrieval_svc.search(user_id=test_user_id, query="interview performance")

            assert len(results) == 1
            retrieved = results[0]
            assert retrieved.content == "Failed an AI technical interview"
            assert retrieved.emotional_context is not None
            assert retrieved.emotional_context["emotion"] == "anxiety"
            assert retrieved.emotional_context["intensity"] == 0.85
            assert retrieved.people_involved == [{"name": "Alex", "role": "interviewer"}]
            assert retrieved.temporal_context == "today"
            assert retrieved.evidence_level == "EXPLICIT_USER"

            # 4. Format through MemoryContextBuilder
            builder = MemoryContextBuilder()
            context_str = builder.build_context(results)
            assert context_str is not None
            assert "Emotion: anxiety, Intensity: 0.85" in context_str
            assert "People Involved: Alex (interviewer)" in context_str
            assert "Temporal Context: today" in context_str

    finally:
        async with async_session() as session:
            await session.execute(
                text("DELETE FROM users WHERE id = :user_id"),
                {"user_id": test_user_id},
            )
            await session.commit()
        await engine.dispose()
