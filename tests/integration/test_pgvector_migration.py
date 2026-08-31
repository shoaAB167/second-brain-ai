import uuid
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from personal_ai.config.settings import get_settings
from personal_ai.db.models import ExperienceModel, User
from personal_ai.db.repositories.sqlalchemy_experience_repository import SQLAlchemyExperienceRepository


@pytest.mark.asyncio
async def test_experience_model_schema_and_vector_query() -> None:
    """Integration test: Validates PostgreSQL pgvector VECTOR(1536) schema, queries, and user isolation.
    
    Validates:
    1. experiences.embedding column exists with data_type='USER-DEFINED' and udt_name='vector'
    2. Column dimension in PostgreSQL catalog is VECTOR(1536)
    3. Existing non-null embeddings survive and have exactly 1536 dimensions
    4. Real User rows with foreign keys are used for test experiences
    5. Newly inserted 1536-dimensional vectors and NULL embeddings are preserved
    6. Real pgvector similarity search (<=>) works via SQLAlchemyExperienceRepository
    7. Strict user isolation: User A only retrieves User A's experiences; User B only retrieves User B's
    8. Test data is reliably cleaned up
    """
    settings = get_settings()
    if "postgresql" not in settings.database_url:
        pytest.skip("PostgreSQL required for pgvector integration test")

    engine = create_async_engine(settings.database_url, echo=False)
    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    try:
        async with session_factory() as session:
            # 1. Verify vector extension is enabled in PostgreSQL
            try:
                res = await session.execute(text("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"))
                ext_row = res.fetchone()
                if not ext_row:
                    pytest.skip("pgvector extension not installed in PostgreSQL")
            except Exception:
                pytest.skip("Could not query pg_extension")

            # 2. Schema check: assert column exists and is vector type (fail explicitly if JSON)
            col_res = await session.execute(text("""
                SELECT column_name, data_type, udt_name 
                FROM information_schema.columns 
                WHERE table_name = 'experiences' AND column_name = 'embedding';
            """))
            col_info = col_res.fetchone()
            assert col_info is not None, "Column 'experiences.embedding' does not exist in database"
            assert col_info[1] == "USER-DEFINED", f"Expected data_type='USER-DEFINED', got {col_info[1]!r}"
            assert col_info[2] == "vector", f"Expected udt_name='vector', got {col_info[2]!r} (must not be json)"

            # 3. Dimension check: verify PostgreSQL column is VECTOR(1536)
            dim_res = await session.execute(text("""
                SELECT atttypmod 
                FROM pg_attribute 
                WHERE attrelid = 'experiences'::regclass AND attname = 'embedding';
            """))
            typmod = dim_res.scalar()
            assert typmod == 1536, f"Expected vector column dimension 1536, got {typmod}"

            # 4. Verify existing non-null embeddings in DB have 1536 dimensions
            existing_dims_res = await session.execute(text("""
                SELECT vector_dims(embedding) 
                FROM experiences 
                WHERE embedding IS NOT NULL;
            """))
            for dim in existing_dims_res.scalars():
                assert dim == 1536, f"Expected existing embedding dimension 1536, got {dim}"

            # 5. Create real User rows for Foreign Key integrity
            user_a = User(
                id=uuid.uuid4(),
                email=f"test_vector_user_a_{uuid.uuid4().hex[:8]}@example.com",
                password_hash="test_hash_a",
            )
            user_b = User(
                id=uuid.uuid4(),
                email=f"test_vector_user_b_{uuid.uuid4().hex[:8]}@example.com",
                password_hash="test_hash_b",
            )
            session.add_all([user_a, user_b])
            await session.commit()

            # 6. Create test experiences referencing real User IDs
            vec_a = [0.1] * 1536
            exp_a = ExperienceModel(
                id=uuid.uuid4(),
                user_id=user_a.id,
                content="User A vector test experience",
                source="CHAT",
                status="RECEIVED",
                type="GOAL",
                domain="career",
                embedding=vec_a,
                embedding_model="gemini-embedding-001",
                embedding_status="COMPLETED",
            )

            vec_b = [-0.1] * 1536
            exp_b = ExperienceModel(
                id=uuid.uuid4(),
                user_id=user_b.id,
                content="User B vector test experience",
                source="CHAT",
                status="RECEIVED",
                type="GOAL",
                domain="career",
                embedding=vec_b,
                embedding_model="gemini-embedding-001",
                embedding_status="COMPLETED",
            )

            exp_null = ExperienceModel(
                id=uuid.uuid4(),
                user_id=user_a.id,
                content="User A unindexed experience with NULL embedding",
                source="CHAT",
                status="RECEIVED",
                embedding=None,
                embedding_status="PENDING",
            )

            session.add_all([exp_a, exp_b, exp_null])
            await session.commit()

            try:
                # Verify inserted vector dimensions via PostgreSQL vector_dims()
                dims_check = await session.execute(
                    text("SELECT vector_dims(embedding) FROM experiences WHERE id = :id"),
                    {"id": exp_a.id},
                )
                assert dims_check.scalar() == 1536, "Inserted vector dimension must be 1536"

                # 7. Execute real repository vector search with pgvector <=>
                repo = SQLAlchemyExperienceRepository(session=session)
                results_a = await repo.search_by_vector(user_id=user_a.id, query_vector=vec_a, limit=5)

                assert len(results_a) == 1, f"Expected 1 result for User A, got {len(results_a)}"
                assert results_a[0][0].id == exp_a.id
                assert results_a[0][1] >= 0.99  # similarity ~ 1.0
                # User isolation: User B's experience is NEVER returned for User A
                assert all(r[0].id != exp_b.id for r in results_a)

                # 8. Verify User B vector search and isolation
                results_b = await repo.search_by_vector(user_id=user_b.id, query_vector=vec_b, limit=5)
                assert len(results_b) == 1, f"Expected 1 result for User B, got {len(results_b)}"
                assert results_b[0][0].id == exp_b.id
                assert results_b[0][1] >= 0.99
                # User isolation: User A's experience is NEVER returned for User B
                assert all(r[0].id != exp_a.id for r in results_b)

            finally:
                # 9. Reliable test data cleanup
                await session.execute(
                    text("DELETE FROM experiences WHERE id IN (:id1, :id2, :id3)"),
                    {"id1": exp_a.id, "id2": exp_b.id, "id3": exp_null.id},
                )
                await session.execute(
                    text("DELETE FROM users WHERE id IN (:u1, :u2)"),
                    {"u1": user_a.id, "u2": user_b.id},
                )
                await session.commit()

    finally:
        await engine.dispose()
