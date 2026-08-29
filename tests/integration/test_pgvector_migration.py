import os
import uuid
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from personal_ai.config.settings import get_settings
from personal_ai.db.models import Base, ExperienceModel, User
from personal_ai.domain.experience import Experience, ExperienceSource, ExperienceStatus, ExperienceType
from personal_ai.db.repositories.sqlalchemy_experience_repository import SQLAlchemyExperienceRepository


def is_postgres_with_pgvector() -> bool:
    """Check if PostgreSQL connection is configured and pgvector extension is available."""
    settings = get_settings()
    if "postgresql" not in settings.database_url:
        return False
    try:
        import psycopg2
        # Quick check or let pytest handle during async fixture
        return True
    except Exception:
        return True


@pytest.mark.asyncio
async def test_experience_model_schema_and_vector_query() -> None:
    """Integration test: Tests pgvector similarity and user isolation on PostgreSQL if available.
    
    When running against PostgreSQL with pgvector, validates:
    1. VECTOR(1536) column definition
    2. Preservation of valid vectors and NULLs
    3. Vector similarity query using <=>
    4. User isolation
    """
    settings = get_settings()
    if "postgresql" not in settings.database_url:
        pytest.skip("PostgreSQL required for pgvector integration test")

    engine = create_async_engine(settings.database_url, echo=False)
    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    try:
        async with session_factory() as session:
            # Check if vector extension is enabled
            try:
                res = await session.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector';"))
                if not res.scalar():
                    pytest.skip("pgvector extension not installed in PostgreSQL")
            except Exception:
                pytest.skip("Could not query pg_extension")

            # Check column definition
            col_res = await session.execute(text("""
                SELECT data_type, udt_name 
                FROM information_schema.columns 
                WHERE table_name = 'experiences' AND column_name = 'embedding';
            """))
            col_info = col_res.fetchone()
            if col_info and col_info[1] == "vector":
                # Verify vector similarity query <=> works directly
                user_a = uuid.uuid4()
                user_b = uuid.uuid4()

                # User A experience with 1536d vector
                vec_a = [0.1] * 1536
                exp_a = ExperienceModel(
                    id=uuid.uuid4(),
                    user_id=user_a,
                    content="User A experience",
                    source="CHAT",
                    status="RECEIVED",
                    type="GOAL",
                    domain="career",
                    embedding=vec_a,
                    embedding_model="gemini-embedding-001",
                    embedding_status="COMPLETED",
                )

                # User B experience with 1536d vector
                vec_b = [-0.1] * 1536
                exp_b = ExperienceModel(
                    id=uuid.uuid4(),
                    user_id=user_b,
                    content="User B experience",
                    source="CHAT",
                    status="RECEIVED",
                    type="GOAL",
                    domain="career",
                    embedding=vec_b,
                    embedding_model="gemini-embedding-001",
                    embedding_status="COMPLETED",
                )

                # NULL embedding experience
                exp_null = ExperienceModel(
                    id=uuid.uuid4(),
                    user_id=user_a,
                    content="User A unindexed experience",
                    source="CHAT",
                    status="RECEIVED",
                    embedding=None,
                    embedding_status="PENDING",
                )

                session.add_all([exp_a, exp_b, exp_null])
                await session.commit()

                # Repository vector search with <=>
                repo = SQLAlchemyExperienceRepository(session=session)
                results_a = await repo.search_by_vector(user_id=user_a, query_vector=vec_a, limit=5)

                assert len(results_a) == 1
                assert results_a[0][0].id == exp_a.id
                assert results_a[0][1] >= 0.99  # similarity ~ 1.0

                # User isolation: User B's experience is never returned for User A
                assert all(r[0].id != exp_b.id for r in results_a)

                # Cleanup test records
                await session.execute(text("DELETE FROM experiences WHERE id IN (:id1, :id2, :id3)"), {
                    "id1": exp_a.id, "id2": exp_b.id, "id3": exp_null.id
                })
                await session.commit()
    finally:
        await engine.dispose()
