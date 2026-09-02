import os
import uuid
import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from personal_ai.application.memory import MemoryContextBuilder, MemoryRetrievalService
from personal_ai.config.settings import get_settings
from personal_ai.db.models import ExperienceModel, User
from personal_ai.db.repositories.sqlalchemy_experience_repository import SQLAlchemyExperienceRepository
from personal_ai.domain.experience import (
    Experience,
    ExperienceImportance,
    ExperienceLifecycle,
    ExperienceSource,
    ExperienceType,
)
from personal_ai.infrastructure.embedding import MockEmbeddingProvider


@pytest.mark.asyncio
async def test_postgres_schema_has_importance_and_lifecycle() -> None:
    """Verify PostgreSQL experiences table contains importance and lifecycle columns."""
    settings = get_settings()
    db_url = settings.database_url.replace("postgresql://", "postgresql+asyncpg://")

    engine = create_async_engine(db_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        # Query column information for experiences
        result = await session.execute(
            text(
                """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = 'experiences' AND column_name IN ('importance', 'lifecycle')
                ORDER BY column_name;
                """
            )
        )
        columns = {row[0]: (row[1], row[2], row[3]) for row in result.all()}

        assert "importance" in columns, "experiences.importance column must exist in PostgreSQL"
        assert "lifecycle" in columns, "experiences.lifecycle column must exist in PostgreSQL"

        assert columns["importance"][0] in ("character varying", "varchar", "text")
        assert columns["importance"][1] == "NO"  # NOT NULL
        assert "MEDIUM" in str(columns["importance"][2])

        assert columns["lifecycle"][0] in ("character varying", "varchar", "text")
        assert columns["lifecycle"][1] == "NO"  # NOT NULL
        assert "STABLE" in str(columns["lifecycle"][2])

    await engine.dispose()


@pytest.mark.asyncio
async def test_experience_persistence_and_retrieval_with_quality_and_lifecycle() -> None:
    """Verify storing experiences with different quality, lifecycle, and preserved qualifiers."""
    settings = get_settings()
    db_url = settings.database_url.replace("postgresql://", "postgresql+asyncpg://")

    engine = create_async_engine(db_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    provider = MockEmbeddingProvider(dimensions=1536)

    test_user_id = uuid.uuid4()
    created_exp_ids = []

    try:
        async with async_session() as session:
            # 1. Create a real User fixture
            user = User(
                id=test_user_id,
                email=f"user_pr14_{test_user_id.hex[:8]}@example.com",
                password_hash="test_hash",
            )
            session.add(user)
            await session.commit()

            repo = SQLAlchemyExperienceRepository(session=session)

            # 2. Add Habit experience with preserved qualifier
            gym_vec = await provider.embed("Usually goes to the gym around 6 PM")
            exp_habit = Experience(
                content="Usually goes to the gym around 6 PM",
                source=ExperienceSource.CHAT,
                user_id=str(test_user_id),
                type=ExperienceType.HABIT,
                domain="fitness",
                importance=ExperienceImportance.MEDIUM,
                lifecycle=ExperienceLifecycle.RECURRING,
                embedding=gym_vec,
                embedding_model=provider.model_name,
                embedding_status="COMPLETED",
            )
            saved_habit = await repo.create(exp_habit)
            created_exp_ids.append(saved_habit.id)

            # 3. Add Goal experience (HIGH importance, STABLE)
            goal_vec = await provider.embed("Wants to reach 30 LPA salary")
            exp_goal = Experience(
                content="Wants to reach 30 LPA salary",
                source=ExperienceSource.CHAT,
                user_id=str(test_user_id),
                type=ExperienceType.GOAL,
                domain="career",
                importance=ExperienceImportance.HIGH,
                lifecycle=ExperienceLifecycle.STABLE,
                embedding=goal_vec,
                embedding_model=provider.model_name,
                embedding_status="COMPLETED",
            )
            saved_goal = await repo.create(exp_goal)
            created_exp_ids.append(saved_goal.id)

            # 4. Search experiences using MemoryRetrievalService
            retrieval_service = MemoryRetrievalService(embedding_provider=provider, experience_repo=repo)
            results = await retrieval_service.search(
                user_id=test_user_id, query="Usually goes to the gym around 6 PM", limit=5
            )

            assert len(results) >= 1
            top_result = results[0]
            assert top_result.experience_id == saved_habit.id
            assert top_result.type == "HABIT"
            assert top_result.importance == "MEDIUM"
            assert top_result.lifecycle == "RECURRING"
            assert "usually" in top_result.content.lower()

            # 5. Format with MemoryContextBuilder
            context_builder = MemoryContextBuilder()
            prompt_context = context_builder.build_context(results)

            assert prompt_context is not None
            assert "<user_memory>" in prompt_context
            assert "Type: HABIT" in prompt_context
            assert "Importance: MEDIUM" in prompt_context
            assert "Lifecycle: RECURRING" in prompt_context
            assert "Content: Usually goes to the gym around 6 PM" in prompt_context

    finally:
        # Cleanup
        async with async_session() as session:
            if created_exp_ids:
                await session.execute(
                    text("DELETE FROM experiences WHERE id = ANY(:ids)"),
                    {"ids": created_exp_ids},
                )
            await session.execute(
                text("DELETE FROM users WHERE id = :uid"),
                {"uid": test_user_id},
            )
            await session.commit()

        await engine.dispose()
