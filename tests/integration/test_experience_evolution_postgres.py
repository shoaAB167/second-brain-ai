import uuid
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from personal_ai.config.settings import get_settings
from personal_ai.db.models import User
from personal_ai.db.repositories.sqlalchemy_experience_relationship_repository import (
    SQLAlchemyExperienceRelationshipRepository,
)
from personal_ai.db.repositories.sqlalchemy_experience_repository import (
    SQLAlchemyExperienceRepository,
)
from personal_ai.domain.experience import (
    Experience,
    ExperienceImportance,
    ExperienceLifecycle,
    ExperienceLifecycleStatus,
    ExperienceRelationship,
    ExperienceRelationshipType,
    ExperienceSource,
    ExperienceType,
)
from personal_ai.infrastructure.embedding import MockEmbeddingProvider


@pytest.mark.asyncio
async def test_postgres_schema_has_lifecycle_status_and_relationships_table() -> None:
    """Verify PostgreSQL has lifecycle_status column on experiences and experience_relationships table."""
    settings = get_settings()
    db_url = settings.database_url.replace("postgresql://", "postgresql+asyncpg://")

    engine = create_async_engine(db_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        # Check experiences.lifecycle_status column
        res_col = await session.execute(
            text(
                """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = 'experiences' AND column_name = 'lifecycle_status';
                """
            )
        )
        row = res_col.first()
        assert row is not None, "experiences.lifecycle_status column must exist in PostgreSQL"
        assert row[1] in ("character varying", "varchar", "text")
        assert row[2] == "NO"  # NOT NULL
        assert "ACTIVE" in str(row[3])

        # Check experience_relationships table
        res_tbl = await session.execute(
            text(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'experience_relationships';
                """
            )
        )
        assert res_tbl.first() is not None, "experience_relationships table must exist in PostgreSQL"

    await engine.dispose()


@pytest.mark.asyncio
async def test_postgres_experience_relationship_crud_and_status_transition() -> None:
    """Verify storing experiences, relationships, and updating lifecycle_status in real PostgreSQL."""
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
                email=f"evo_user_{test_user_id.hex[:8]}@example.com",
                password_hash="test_hash",
            )
            session.add(user)
            await session.commit()

            exp_repo = SQLAlchemyExperienceRepository(session=session)
            rel_repo = SQLAlchemyExperienceRelationshipRepository(session=session)

            # 2. Add Old Experience (ACTIVE)
            vec_old = await provider.embed("I go to the gym at 6 PM")
            exp_old = await exp_repo.create(
                Experience(
                    content="I go to the gym at 6 PM",
                    source=ExperienceSource.CHAT,
                    user_id=str(test_user_id),
                    type=ExperienceType.HABIT,
                    importance=ExperienceImportance.MEDIUM,
                    lifecycle=ExperienceLifecycle.RECURRING,
                    lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
                    embedding=vec_old,
                    embedding_model=provider.model_name,
                    embedding_status="COMPLETED",
                )
            )

            # 3. Add New Experience (ACTIVE)
            vec_new = await provider.embed("I now go to the gym at 7 PM")
            exp_new = await exp_repo.create(
                Experience(
                    content="I now go to the gym at 7 PM",
                    source=ExperienceSource.CHAT,
                    user_id=str(test_user_id),
                    type=ExperienceType.HABIT,
                    importance=ExperienceImportance.MEDIUM,
                    lifecycle=ExperienceLifecycle.RECURRING,
                    lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
                    embedding=vec_new,
                    embedding_model=provider.model_name,
                    embedding_status="COMPLETED",
                )
            )

            # 4. Create and persist ExperienceRelationship (UPDATES)
            rel = ExperienceRelationship(
                source_experience_id=exp_new.id,
                target_experience_id=exp_old.id,
                relationship_type=ExperienceRelationshipType.UPDATES,
                confidence=0.93,
                reason="Updated evening gym schedule.",
            )
            saved_rel = await rel_repo.create(rel)
            assert saved_rel.id is not None
            assert saved_rel.relationship_type == ExperienceRelationshipType.UPDATES

            # 5. Transition old experience to SUPERSEDED
            exp_old.lifecycle_status = ExperienceLifecycleStatus.SUPERSEDED
            await exp_repo.update(exp_old)

            # 6. Verify retrieval filtering in PostgreSQL
            # Default ACTIVE search should ONLY return exp_new
            active_results = await exp_repo.search_by_vector(
                user_id=test_user_id,
                query_vector=vec_new,
                limit=5,
                lifecycle_status="ACTIVE",
            )
            assert len(active_results) == 1
            assert active_results[0][0].id == exp_new.id
            assert active_results[0][0].lifecycle_status == ExperienceLifecycleStatus.ACTIVE

            # Unfiltered search (lifecycle_status=None) should return BOTH
            all_results = await exp_repo.search_by_vector(
                user_id=test_user_id,
                query_vector=vec_new,
                limit=5,
                lifecycle_status=None,
            )
            assert len(all_results) == 2
            ids = [item[0].id for item in all_results]
            assert exp_new.id in ids
            assert exp_old.id in ids

            # 7. Query relationships by experience ID
            retrieved_rels = await rel_repo.get_by_source_id(exp_new.id)
            assert len(retrieved_rels) == 1
            assert retrieved_rels[0].target_experience_id == exp_old.id
            assert retrieved_rels[0].relationship_type == ExperienceRelationshipType.UPDATES

    finally:
        # Cleanup
        async with async_session() as session:
            await session.execute(
                text("DELETE FROM users WHERE id = :user_id"),
                {"user_id": test_user_id},
            )
            await session.commit()
        await engine.dispose()
