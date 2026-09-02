import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
import uuid

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from personal_ai.application.experience.evolution_classifier import (
    ExperienceEvolutionClassificationResult,
    ExperienceEvolutionClassifier,
)
from personal_ai.application.experience.evolution_service import ExperienceEvolutionService
from personal_ai.db.models import Base, Message, MessageRole, User
from personal_ai.db.repositories.sqlalchemy_experience_relationship_repository import (
    SQLAlchemyExperienceRelationshipRepository,
)
from personal_ai.db.repositories.sqlalchemy_experience_repository import (
    SQLAlchemyExperienceRepository,
)
from personal_ai.domain.experience.entity import Experience
from personal_ai.domain.experience.enums import (
    ExperienceImportance,
    ExperienceLifecycle,
    ExperienceLifecycleStatus,
    ExperienceRelationshipType,
    ExperienceSource,
    ExperienceStatus,
    ExperienceType,
)
from personal_ai.domain.experience.relationship import ExperienceRelationship
from personal_ai.infrastructure.embedding.provider import EmbeddingProvider
from personal_ai.infrastructure.experience.background_processor import (
    SQLAlchemyBackgroundExperienceProcessor,
)
from personal_ai.llm.client import LLMClient
from personal_ai.llm.models import LLMMessage, LLMResponse


@pytest_asyncio.fixture
async def session_maker():
    """Fixture providing isolated in-memory SQLite database sessionmaker."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    yield maker
    await engine.dispose()


# ==============================================================================
# Requirement A: New Experience Defaults to ACTIVE
# ==============================================================================

def test_new_experience_defaults_to_active_lifecycle_status():
    """Requirement A: Verify new experience defaults to ACTIVE lifecycle status."""
    exp = Experience(
        content="I prefer tea over coffee.",
        source=ExperienceSource.CHAT,
    )
    assert exp.lifecycle_status == ExperienceLifecycleStatus.ACTIVE


def test_invalid_lifecycle_status_raises_error():
    """Verify invalid lifecycle status raises ValueError."""
    with pytest.raises(ValueError, match="Invalid experience lifecycle status"):
        Experience(
            content="I prefer tea over coffee.",
            source=ExperienceSource.CHAT,
            lifecycle_status="INVALID_STATUS",  # type: ignore
        )


# ==============================================================================
# Requirement F: Invalid Classifier Output Fails Closed
# ==============================================================================

@pytest.mark.asyncio
async def test_evolution_classifier_invalid_output_fails_closed():
    """Requirement F: Return invalid relationship JSON and verify failure closed to UNRELATED with 0 confidence."""
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(
            content='{"relationship": "INVALID_RELATIONSHIP", "confidence": 0.99, "reason": "gibberish"}',
            provider="gemini",
            model="gemini-3.6-flash",
            latency_ms=50.0,
        )
    )

    classifier = ExperienceEvolutionClassifier(llm_client=mock_llm)
    exp1 = Experience(content="Gym at 6 PM", source=ExperienceSource.CHAT)
    exp2 = Experience(content="Gym at 7 PM", source=ExperienceSource.CHAT)

    result = await classifier.classify_relationship(new_experience=exp2, existing_experience=exp1)
    assert result.relationship == ExperienceRelationshipType.UNRELATED
    assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_evolution_classifier_malformed_json_fails_closed():
    """Verify non-JSON output fails closed."""
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(
            content="I cannot classify these memories.",
            provider="gemini",
            model="gemini-3.6-flash",
            latency_ms=50.0,
        )
    )

    classifier = ExperienceEvolutionClassifier(llm_client=mock_llm)
    exp1 = Experience(content="Gym at 6 PM", source=ExperienceSource.CHAT)
    exp2 = Experience(content="Gym at 7 PM", source=ExperienceSource.CHAT)

    result = await classifier.classify_relationship(new_experience=exp2, existing_experience=exp1)
    assert result.relationship == ExperienceRelationshipType.UNRELATED
    assert result.confidence == 0.0


# ==============================================================================
# Requirement B: UPDATES Relationship (Old becomes SUPERSEDED, New is ACTIVE)
# ==============================================================================

@pytest.mark.asyncio
async def test_evolution_service_updates_supersedes_old_memory(session_maker):
    """Requirement B: UPDATES relationship marks old memory SUPERSEDED and creates relationship."""
    user_id = uuid.uuid4()
    dummy_vec = [0.1] * 1536

    async with session_maker() as session:
        exp_repo = SQLAlchemyExperienceRepository(session=session)
        rel_repo = SQLAlchemyExperienceRelationshipRepository(session=session)

        # 1. Old Experience (ACTIVE)
        old_exp = Experience(
            content="I go to gym at 6 PM.",
            source=ExperienceSource.CHAT,
            user_id=str(user_id),
            embedding=dummy_vec,
            embedding_status="COMPLETED",
            lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
        )
        saved_old = await exp_repo.create(old_exp)

        # 2. New Experience (ACTIVE)
        new_exp = Experience(
            content="I now go to gym at 7 PM.",
            source=ExperienceSource.CHAT,
            user_id=str(user_id),
            embedding=dummy_vec,
            embedding_status="COMPLETED",
            lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
        )
        saved_new = await exp_repo.create(new_exp)

        # Mock classifier returning UPDATES
        mock_classifier = MagicMock(spec=ExperienceEvolutionClassifier)
        mock_classifier.classify_relationship = AsyncMock(
            return_value=ExperienceEvolutionClassificationResult(
                relationship=ExperienceRelationshipType.UPDATES,
                confidence=0.92,
                reason="The new gym time replaces the previously stated schedule.",
            )
        )

        service = ExperienceEvolutionService(
            experience_repo=exp_repo,
            relationship_repo=rel_repo,
            classifier=mock_classifier,
        )

        rels = await service.evolve_experience(experience=saved_new, user_id=user_id)

        assert len(rels) == 1
        assert rels[0].source_experience_id == saved_new.id
        assert rels[0].target_experience_id == saved_old.id
        assert rels[0].relationship_type == ExperienceRelationshipType.UPDATES
        assert rels[0].confidence == 0.92

        # Verify old memory is now SUPERSEDED
        refreshed_old = await exp_repo.get_by_id(saved_old.id)
        assert refreshed_old is not None
        assert refreshed_old.lifecycle_status == ExperienceLifecycleStatus.SUPERSEDED

        # Verify new memory is still ACTIVE
        refreshed_new = await exp_repo.get_by_id(saved_new.id)
        assert refreshed_new is not None
        assert refreshed_new.lifecycle_status == ExperienceLifecycleStatus.ACTIVE


# ==============================================================================
# Requirement C: CONTRADICTS Relationship (Both preserved, neither superseded)
# ==============================================================================

@pytest.mark.asyncio
async def test_evolution_service_contradicts_preserves_both_active(session_maker):
    """Requirement C: CONTRADICTS relationship creates link and preserves BOTH memories as ACTIVE."""
    user_id = uuid.uuid4()
    dummy_vec = [0.2] * 1536

    async with session_maker() as session:
        exp_repo = SQLAlchemyExperienceRepository(session=session)
        rel_repo = SQLAlchemyExperienceRelationshipRepository(session=session)

        old_exp = await exp_repo.create(
            Experience(
                content="I don't want to work in AI.",
                source=ExperienceSource.CHAT,
                user_id=str(user_id),
                embedding=dummy_vec,
                embedding_status="COMPLETED",
                lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
            )
        )
        new_exp = await exp_repo.create(
            Experience(
                content="I want to become an AI engineer.",
                source=ExperienceSource.CHAT,
                user_id=str(user_id),
                embedding=dummy_vec,
                embedding_status="COMPLETED",
                lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
            )
        )

        mock_classifier = MagicMock(spec=ExperienceEvolutionClassifier)
        mock_classifier.classify_relationship = AsyncMock(
            return_value=ExperienceEvolutionClassificationResult(
                relationship=ExperienceRelationshipType.CONTRADICTS,
                confidence=0.88,
                reason="Conflicting personal career preferences.",
            )
        )

        service = ExperienceEvolutionService(
            experience_repo=exp_repo,
            relationship_repo=rel_repo,
            classifier=mock_classifier,
        )

        rels = await service.evolve_experience(experience=new_exp, user_id=user_id)

        assert len(rels) == 1
        assert rels[0].relationship_type == ExperienceRelationshipType.CONTRADICTS

        # Verify BOTH experiences remain ACTIVE and neither is deleted
        refreshed_old = await exp_repo.get_by_id(old_exp.id)
        refreshed_new = await exp_repo.get_by_id(new_exp.id)
        assert refreshed_old.lifecycle_status == ExperienceLifecycleStatus.ACTIVE
        assert refreshed_new.lifecycle_status == ExperienceLifecycleStatus.ACTIVE


# ==============================================================================
# Requirement D: REINFORCES Relationship (Existing remains ACTIVE)
# ==============================================================================

@pytest.mark.asyncio
async def test_evolution_service_reinforces_keeps_existing_active(session_maker):
    """Requirement D: REINFORCES relationship creates link and keeps existing memory ACTIVE."""
    user_id = uuid.uuid4()
    dummy_vec = [0.3] * 1536

    async with session_maker() as session:
        exp_repo = SQLAlchemyExperienceRepository(session=session)
        rel_repo = SQLAlchemyExperienceRelationshipRepository(session=session)

        old_exp = await exp_repo.create(
            Experience(
                content="I like volleyball.",
                source=ExperienceSource.CHAT,
                user_id=str(user_id),
                embedding=dummy_vec,
                embedding_status="COMPLETED",
                lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
            )
        )
        new_exp = await exp_repo.create(
            Experience(
                content="I really enjoy playing volleyball.",
                source=ExperienceSource.CHAT,
                user_id=str(user_id),
                embedding=dummy_vec,
                embedding_status="COMPLETED",
                lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
            )
        )

        mock_classifier = MagicMock(spec=ExperienceEvolutionClassifier)
        mock_classifier.classify_relationship = AsyncMock(
            return_value=ExperienceEvolutionClassificationResult(
                relationship=ExperienceRelationshipType.REINFORCES,
                confidence=0.95,
                reason="Reiterates strong volleyball preference.",
            )
        )

        service = ExperienceEvolutionService(
            experience_repo=exp_repo,
            relationship_repo=rel_repo,
            classifier=mock_classifier,
        )

        rels = await service.evolve_experience(experience=new_exp, user_id=user_id)

        assert len(rels) == 1
        assert rels[0].relationship_type == ExperienceRelationshipType.REINFORCES

        refreshed_old = await exp_repo.get_by_id(old_exp.id)
        assert refreshed_old.lifecycle_status == ExperienceLifecycleStatus.ACTIVE


# ==============================================================================
# Requirement E: RELATED Relationship (No lifecycle mutation)
# ==============================================================================

@pytest.mark.asyncio
async def test_evolution_service_related_no_lifecycle_mutation(session_maker):
    """Requirement E: RELATED relationship creates link with no lifecycle mutation."""
    user_id = uuid.uuid4()
    dummy_vec = [0.4] * 1536

    async with session_maker() as session:
        exp_repo = SQLAlchemyExperienceRepository(session=session)
        rel_repo = SQLAlchemyExperienceRelationshipRepository(session=session)

        old_exp = await exp_repo.create(
            Experience(
                content="I play volleyball.",
                source=ExperienceSource.CHAT,
                user_id=str(user_id),
                embedding=dummy_vec,
                embedding_status="COMPLETED",
                lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
            )
        )
        new_exp = await exp_repo.create(
            Experience(
                content="I played volleyball yesterday.",
                source=ExperienceSource.CHAT,
                user_id=str(user_id),
                embedding=dummy_vec,
                embedding_status="COMPLETED",
                lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
            )
        )

        mock_classifier = MagicMock(spec=ExperienceEvolutionClassifier)
        mock_classifier.classify_relationship = AsyncMock(
            return_value=ExperienceEvolutionClassificationResult(
                relationship=ExperienceRelationshipType.RELATED,
                confidence=0.80,
                reason="Event related to volleyball habit.",
            )
        )

        service = ExperienceEvolutionService(
            experience_repo=exp_repo,
            relationship_repo=rel_repo,
            classifier=mock_classifier,
        )

        rels = await service.evolve_experience(experience=new_exp, user_id=user_id)

        assert len(rels) == 1
        assert rels[0].relationship_type == ExperienceRelationshipType.RELATED

        refreshed_old = await exp_repo.get_by_id(old_exp.id)
        refreshed_new = await exp_repo.get_by_id(new_exp.id)
        assert refreshed_old.lifecycle_status == ExperienceLifecycleStatus.ACTIVE
        assert refreshed_new.lifecycle_status == ExperienceLifecycleStatus.ACTIVE


# ==============================================================================
# Requirement G: Duplicate Relationship Prevention
# ==============================================================================

@pytest.mark.asyncio
async def test_duplicate_relationship_prevention(session_maker):
    """Requirement G: Running evolution twice does not create duplicate relationships."""
    user_id = uuid.uuid4()
    dummy_vec = [0.5] * 1536

    async with session_maker() as session:
        exp_repo = SQLAlchemyExperienceRepository(session=session)
        rel_repo = SQLAlchemyExperienceRelationshipRepository(session=session)

        old_exp = await exp_repo.create(
            Experience(
                content="My salary goal is 20 LPA.",
                source=ExperienceSource.CHAT,
                user_id=str(user_id),
                embedding=dummy_vec,
                embedding_status="COMPLETED",
            )
        )
        new_exp = await exp_repo.create(
            Experience(
                content="My salary goal is now 30 LPA.",
                source=ExperienceSource.CHAT,
                user_id=str(user_id),
                embedding=dummy_vec,
                embedding_status="COMPLETED",
            )
        )

        mock_classifier = MagicMock(spec=ExperienceEvolutionClassifier)
        mock_classifier.classify_relationship = AsyncMock(
            return_value=ExperienceEvolutionClassificationResult(
                relationship=ExperienceRelationshipType.UPDATES,
                confidence=0.90,
                reason="Updated salary target.",
            )
        )

        service = ExperienceEvolutionService(
            experience_repo=exp_repo,
            relationship_repo=rel_repo,
            classifier=mock_classifier,
        )

        # Run 1
        rels_1 = await service.evolve_experience(experience=new_exp, user_id=user_id)
        assert len(rels_1) == 1

        # Run 2
        rels_2 = await service.evolve_experience(experience=new_exp, user_id=user_id)
        assert len(rels_2) == 0  # Deduplicated

        all_rels = await rel_repo.get_by_source_id(new_exp.id)
        assert len(all_rels) == 1


# ==============================================================================
# Requirement H: Historical Preservation (Superseded experience remains in DB)
# ==============================================================================

@pytest.mark.asyncio
async def test_historical_preservation_superseded_memory_remains_in_db(session_maker):
    """Requirement H: Superseded experience is NEVER deleted and remains queryable."""
    user_id = uuid.uuid4()
    dummy_vec = [0.6] * 1536

    async with session_maker() as session:
        exp_repo = SQLAlchemyExperienceRepository(session=session)

        exp = await exp_repo.create(
            Experience(
                content="I used to work in banking.",
                source=ExperienceSource.CHAT,
                user_id=str(user_id),
                embedding=dummy_vec,
                embedding_status="COMPLETED",
                lifecycle_status=ExperienceLifecycleStatus.SUPERSEDED,
            )
        )

        # Query by ID
        fetched = await exp_repo.get_by_id(exp.id)
        assert fetched is not None
        assert fetched.content == "I used to work in banking."
        assert fetched.lifecycle_status == ExperienceLifecycleStatus.SUPERSEDED

        # Vector search with default ACTIVE ignores it
        active_results = await exp_repo.search_by_vector(
            user_id=user_id,
            query_vector=dummy_vec,
            lifecycle_status="ACTIVE",
        )
        assert len(active_results) == 0

        # Vector search with lifecycle_status=None retrieves it
        all_results = await exp_repo.search_by_vector(
            user_id=user_id,
            query_vector=dummy_vec,
            lifecycle_status=None,
        )
        assert len(all_results) == 1
        assert all_results[0][0].id == exp.id


# ==============================================================================
# Requirement I: Candidate Isolation Across Users
# ==============================================================================

@pytest.mark.asyncio
async def test_evolution_candidate_isolation_across_users(session_maker):
    """Requirement I: User A's experience is NEVER compared or related to User B's experiences."""
    user_a = uuid.uuid4()
    user_b = uuid.uuid4()
    dummy_vec = [0.7] * 1536

    async with session_maker() as session:
        exp_repo = SQLAlchemyExperienceRepository(session=session)
        rel_repo = SQLAlchemyExperienceRelationshipRepository(session=session)

        # User B has an existing experience
        user_b_exp = await exp_repo.create(
            Experience(
                content="I go to gym at 6 PM.",
                source=ExperienceSource.CHAT,
                user_id=str(user_b),
                embedding=dummy_vec,
                embedding_status="COMPLETED",
            )
        )

        # User A records a new experience
        user_a_exp = await exp_repo.create(
            Experience(
                content="I now go to gym at 7 PM.",
                source=ExperienceSource.CHAT,
                user_id=str(user_a),
                embedding=dummy_vec,
                embedding_status="COMPLETED",
            )
        )

        mock_classifier = MagicMock(spec=ExperienceEvolutionClassifier)
        mock_classifier.classify_relationship = AsyncMock()

        service = ExperienceEvolutionService(
            experience_repo=exp_repo,
            relationship_repo=rel_repo,
            classifier=mock_classifier,
        )

        # Evolve for User A
        rels = await service.evolve_experience(experience=user_a_exp, user_id=user_a)

        # User B's memory was NEVER sent to classifier
        assert len(rels) == 0
        mock_classifier.classify_relationship.assert_not_called()

        # User B's experience is untouched and remains ACTIVE
        refreshed_b = await exp_repo.get_by_id(user_b_exp.id)
        assert refreshed_b.lifecycle_status == ExperienceLifecycleStatus.ACTIVE


# ==============================================================================
# Requirement J: Evolution Failure Handled Gracefully in Background Pipeline
# ==============================================================================

@pytest.mark.asyncio
async def test_background_processor_graceful_evolution_failure(session_maker):
    """Requirement J: If evolution fails, background processor logs safely and keeps experience intact."""
    user_id = uuid.uuid4()

    # Create User, Conversation, and Message in DB
    conv_id = uuid.uuid4()
    msg_id = uuid.uuid4()
    async with session_maker() as session:
        user = User(id=user_id, email="evouser@example.com", password_hash="hash")
        session.add(user)
        await session.commit()

        from personal_ai.db.models import Conversation
        conv = Conversation(id=conv_id, user_id=user_id)
        session.add(conv)
        await session.commit()

        message = Message(
            id=msg_id,
            conversation_id=conv_id,
            role=MessageRole.USER,
            content="I go to gym at 6 PM every day.",
            created_at=datetime.now(timezone.utc),
        )
        session.add(message)
        await session.commit()

    mock_llm = MagicMock(spec=LLMClient)
    # 1. Classifier call -> 2. Extractor call
    mock_llm.generate_response = AsyncMock(
        side_effect=[
            LLMResponse(
                content='{"is_experience": true, "type": "HABIT", "importance": 0.8, "confidence": 0.9}',
                provider="gemini",
                model="gemini-3.6-flash",
                latency_ms=10.0,
            ),
            LLMResponse(
                content='{"content": "Goes to gym at 6 PM", "type": "HABIT", "domain": "Fitness", "importance": "HIGH", "lifecycle": "RECURRING", "confidence": 0.95}',
                provider="gemini",
                model="gemini-3.6-flash",
                latency_ms=10.0,
            ),
        ]
    )

    mock_embedder = MagicMock(spec=EmbeddingProvider)
    mock_embedder.model_name = "text-embedding-004"
    mock_embedder.dimensions = 1536
    mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

    # Mock evolution classifier that raises an unexpected exception
    failing_classifier = MagicMock(spec=ExperienceEvolutionClassifier)
    failing_classifier.classify_relationship = AsyncMock(side_effect=RuntimeError("Evolution API crashed"))

    processor = SQLAlchemyBackgroundExperienceProcessor(
        session_factory=session_maker,
        llm_client=mock_llm,
        embedding_provider=mock_embedder,
        evolution_classifier=failing_classifier,
    )

    # Process background promotion
    await processor.process_background_promotion(message=message, user_id=user_id)

    # Verify experience and embedding were still successfully persisted
    async with session_maker() as session:
        exp_repo = SQLAlchemyExperienceRepository(session=session)
        exp = await exp_repo.get_by_source_message_id(message.id)
        assert exp is not None
        assert exp.content == "Goes to gym at 6 PM"
        assert exp.embedding_status == "COMPLETED"
        assert exp.lifecycle_status == ExperienceLifecycleStatus.ACTIVE
