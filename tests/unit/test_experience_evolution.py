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
# Requirement: New Experience Defaults to ACTIVE
# ==============================================================================

def test_new_experience_defaults_to_active_lifecycle_status():
    """Verify new experience defaults to ACTIVE lifecycle status."""
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
# Requirement: Batch LLM Call Counts (0 candidates -> 0 calls, 1-3 -> 1 call)
# ==============================================================================

@pytest.mark.asyncio
async def test_zero_candidates_results_in_zero_llm_calls(session_maker):
    """Verify 0 candidates results in ZERO LLM classifier calls."""
    user_id = uuid.uuid4()
    dummy_vec = [0.1] * 1536

    async with session_maker() as session:
        exp_repo = SQLAlchemyExperienceRepository(session=session)
        rel_repo = SQLAlchemyExperienceRelationshipRepository(session=session)

        new_exp = await exp_repo.create(
            Experience(
                content="I go to gym at 7 PM.",
                source=ExperienceSource.CHAT,
                user_id=str(user_id),
                embedding=dummy_vec,
                embedding_status="COMPLETED",
            )
        )

        mock_classifier = MagicMock(spec=ExperienceEvolutionClassifier)
        mock_classifier.classify_relationships = AsyncMock()

        service = ExperienceEvolutionService(
            experience_repo=exp_repo,
            relationship_repo=rel_repo,
            classifier=mock_classifier,
        )

        rels = await service.evolve_experience(experience=new_exp, user_id=user_id)
        assert len(rels) == 0
        mock_classifier.classify_relationships.assert_not_called()


@pytest.mark.asyncio
async def test_one_candidate_results_in_exactly_one_llm_call(session_maker):
    """Verify 1 candidate results in EXACTLY ONE LLM classifier call."""
    user_id = uuid.uuid4()
    dummy_vec = [0.1] * 1536

    async with session_maker() as session:
        exp_repo = SQLAlchemyExperienceRepository(session=session)
        rel_repo = SQLAlchemyExperienceRelationshipRepository(session=session)

        old_exp = await exp_repo.create(
            Experience(
                content="I go to gym at 6 PM.",
                source=ExperienceSource.CHAT,
                user_id=str(user_id),
                embedding=dummy_vec,
                embedding_status="COMPLETED",
            )
        )
        new_exp = await exp_repo.create(
            Experience(
                content="I now go to gym at 7 PM.",
                source=ExperienceSource.CHAT,
                user_id=str(user_id),
                embedding=dummy_vec,
                embedding_status="COMPLETED",
            )
        )

        mock_classifier = MagicMock(spec=ExperienceEvolutionClassifier)
        mock_classifier.classify_relationships = AsyncMock(
            return_value={
                old_exp.id: ExperienceEvolutionClassificationResult(
                    candidate_id=old_exp.id,
                    relationship=ExperienceRelationshipType.UPDATES,
                    confidence=0.92,
                    reason="Updated gym time.",
                )
            }
        )

        service = ExperienceEvolutionService(
            experience_repo=exp_repo,
            relationship_repo=rel_repo,
            classifier=mock_classifier,
        )

        rels = await service.evolve_experience(experience=new_exp, user_id=user_id)
        assert len(rels) == 1
        assert mock_classifier.classify_relationships.call_count == 1


@pytest.mark.asyncio
async def test_three_candidates_results_in_exactly_one_llm_call(session_maker):
    """Verify 3 candidates results in EXACTLY ONE LLM classifier call (batching all candidates)."""
    user_id = uuid.uuid4()
    dummy_vec = [0.1] * 1536

    async with session_maker() as session:
        exp_repo = SQLAlchemyExperienceRepository(session=session)
        rel_repo = SQLAlchemyExperienceRelationshipRepository(session=session)

        old_1 = await exp_repo.create(
            Experience(
                content="I go to gym at 6 PM.",
                source=ExperienceSource.CHAT,
                user_id=str(user_id),
                embedding=dummy_vec,
                embedding_status="COMPLETED",
            )
        )
        old_2 = await exp_repo.create(
            Experience(
                content="I drink protein shake at 6 PM.",
                source=ExperienceSource.CHAT,
                user_id=str(user_id),
                embedding=dummy_vec,
                embedding_status="COMPLETED",
            )
        )
        old_3 = await exp_repo.create(
            Experience(
                content="I workout with Alex at 6 PM.",
                source=ExperienceSource.CHAT,
                user_id=str(user_id),
                embedding=dummy_vec,
                embedding_status="COMPLETED",
            )
        )
        new_exp = await exp_repo.create(
            Experience(
                content="I now go to gym at 7 PM.",
                source=ExperienceSource.CHAT,
                user_id=str(user_id),
                embedding=dummy_vec,
                embedding_status="COMPLETED",
            )
        )

        mock_classifier = MagicMock(spec=ExperienceEvolutionClassifier)
        mock_classifier.classify_relationships = AsyncMock(
            return_value={
                old_1.id: ExperienceEvolutionClassificationResult(
                    candidate_id=old_1.id,
                    relationship=ExperienceRelationshipType.UPDATES,
                    confidence=0.92,
                    reason="Updated gym time.",
                ),
                old_2.id: ExperienceEvolutionClassificationResult(
                    candidate_id=old_2.id,
                    relationship=ExperienceRelationshipType.RELATED,
                    confidence=0.80,
                    reason="Related workout habit.",
                ),
                old_3.id: ExperienceEvolutionClassificationResult(
                    candidate_id=old_3.id,
                    relationship=ExperienceRelationshipType.RELATED,
                    confidence=0.85,
                    reason="Related workout partner.",
                ),
            }
        )

        service = ExperienceEvolutionService(
            experience_repo=exp_repo,
            relationship_repo=rel_repo,
            classifier=mock_classifier,
            candidate_limit=3,
        )

        rels = await service.evolve_experience(experience=new_exp, user_id=user_id)
        assert len(rels) == 3
        # Exactly 1 LLM call for all 3 candidates
        assert mock_classifier.classify_relationships.call_count == 1
        call_args = mock_classifier.classify_relationships.call_args[1]
        assert len(call_args["candidate_experiences"]) == 3


# ==============================================================================
# Requirement: Candidate Limit and Similarity Threshold Configuration
# ==============================================================================

@pytest.mark.asyncio
async def test_candidate_limit_is_respected(session_maker):
    """Verify candidate_limit (e.g. 2) strictly caps candidates sent to the classifier."""
    user_id = uuid.uuid4()
    dummy_vec = [0.1] * 1536

    async with session_maker() as session:
        exp_repo = SQLAlchemyExperienceRepository(session=session)
        rel_repo = SQLAlchemyExperienceRelationshipRepository(session=session)

        for i in range(5):
            await exp_repo.create(
                Experience(
                    content=f"Old memory {i}",
                    source=ExperienceSource.CHAT,
                    user_id=str(user_id),
                    embedding=dummy_vec,
                    embedding_status="COMPLETED",
                )
            )

        new_exp = await exp_repo.create(
            Experience(
                content="New memory",
                source=ExperienceSource.CHAT,
                user_id=str(user_id),
                embedding=dummy_vec,
                embedding_status="COMPLETED",
            )
        )

        mock_classifier = MagicMock(spec=ExperienceEvolutionClassifier)
        mock_classifier.classify_relationships = AsyncMock(return_value={})

        service = ExperienceEvolutionService(
            experience_repo=exp_repo,
            relationship_repo=rel_repo,
            classifier=mock_classifier,
            candidate_limit=2,
        )

        await service.evolve_experience(experience=new_exp, user_id=user_id)
        assert mock_classifier.classify_relationships.call_count == 1
        call_args = mock_classifier.classify_relationships.call_args[1]
        assert len(call_args["candidate_experiences"]) == 2


@pytest.mark.asyncio
async def test_configurable_similarity_threshold(session_maker):
    """Verify candidate_similarity_threshold filters out low similarity memories before classification."""
    user_id = uuid.uuid4()
    vec_a = [1.0] + [0.0] * 1535  # orthogonal to vec_b
    vec_b = [0.0, 1.0] + [0.0] * 1534

    async with session_maker() as session:
        exp_repo = SQLAlchemyExperienceRepository(session=session)
        rel_repo = SQLAlchemyExperienceRelationshipRepository(session=session)

        # Candidate with 0.0 cosine similarity
        await exp_repo.create(
            Experience(
                content="Orthogonal memory",
                source=ExperienceSource.CHAT,
                user_id=str(user_id),
                embedding=vec_b,
                embedding_status="COMPLETED",
            )
        )

        new_exp = await exp_repo.create(
            Experience(
                content="New memory",
                source=ExperienceSource.CHAT,
                user_id=str(user_id),
                embedding=vec_a,
                embedding_status="COMPLETED",
            )
        )

        mock_classifier = MagicMock(spec=ExperienceEvolutionClassifier)
        mock_classifier.classify_relationships = AsyncMock(return_value={})

        service = ExperienceEvolutionService(
            experience_repo=exp_repo,
            relationship_repo=rel_repo,
            classifier=mock_classifier,
            candidate_similarity_threshold=0.5,  # 0.0 < 0.5 -> filtered out
        )

        await service.evolve_experience(experience=new_exp, user_id=user_id)
        # 0 candidates passed threshold -> 0 LLM calls
        mock_classifier.classify_relationships.assert_not_called()


# ==============================================================================
# Requirement: Classifier Fail-Closed Handling
# ==============================================================================

@pytest.mark.asyncio
async def test_classifier_invalid_json_fails_closed():
    """Verify invalid JSON returns empty mapping safely."""
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(
            content="This is not valid JSON.",
            provider="gemini",
            model="gemini-3.6-flash",
            latency_ms=10.0,
        )
    )

    classifier = ExperienceEvolutionClassifier(llm_client=mock_llm)
    cand_id = uuid.uuid4()
    cand = Experience(id=cand_id, content="Old", source=ExperienceSource.CHAT)
    new_exp = Experience(content="New", source=ExperienceSource.CHAT)

    results = await classifier.classify_relationships(new_experience=new_exp, candidate_experiences=[cand])
    assert results == {}


@pytest.mark.asyncio
async def test_classifier_unrecognized_candidate_id_ignored():
    """Verify candidate_ids not in the requested candidate list are ignored."""
    cand_id_real = uuid.uuid4()
    cand_id_fake = uuid.uuid4()

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(
            content=f'{{"relationships": [{{"candidate_id": "{cand_id_fake}", "relationship": "UPDATES", "confidence": 0.95}}]}}',
            provider="gemini",
            model="gemini-3.6-flash",
            latency_ms=10.0,
        )
    )

    classifier = ExperienceEvolutionClassifier(llm_client=mock_llm)
    cand = Experience(id=cand_id_real, content="Old", source=ExperienceSource.CHAT)
    new_exp = Experience(content="New", source=ExperienceSource.CHAT)

    results = await classifier.classify_relationships(new_experience=new_exp, candidate_experiences=[cand])
    assert results == {}


# ==============================================================================
# Requirement: UPDATES Lifecycle Transition (Old SUPERSEDED, New ACTIVE)
# ==============================================================================

@pytest.mark.asyncio
async def test_evolution_updates_supersedes_old_memory(session_maker):
    """Verify UPDATES transitions old memory to SUPERSEDED while new stays ACTIVE."""
    user_id = uuid.uuid4()
    dummy_vec = [0.1] * 1536

    async with session_maker() as session:
        exp_repo = SQLAlchemyExperienceRepository(session=session)
        rel_repo = SQLAlchemyExperienceRelationshipRepository(session=session)

        old_exp = await exp_repo.create(
            Experience(
                content="I go to gym at 6 PM.",
                source=ExperienceSource.CHAT,
                user_id=str(user_id),
                embedding=dummy_vec,
                embedding_status="COMPLETED",
                lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
            )
        )
        new_exp = await exp_repo.create(
            Experience(
                content="I now go to gym at 7 PM.",
                source=ExperienceSource.CHAT,
                user_id=str(user_id),
                embedding=dummy_vec,
                embedding_status="COMPLETED",
                lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
            )
        )

        mock_classifier = MagicMock(spec=ExperienceEvolutionClassifier)
        mock_classifier.classify_relationships = AsyncMock(
            return_value={
                old_exp.id: ExperienceEvolutionClassificationResult(
                    candidate_id=old_exp.id,
                    relationship=ExperienceRelationshipType.UPDATES,
                    confidence=0.92,
                    reason="Updated gym time.",
                )
            }
        )

        service = ExperienceEvolutionService(
            experience_repo=exp_repo,
            relationship_repo=rel_repo,
            classifier=mock_classifier,
        )

        rels = await service.evolve_experience(experience=new_exp, user_id=user_id)
        assert len(rels) == 1
        assert rels[0].relationship_type == ExperienceRelationshipType.UPDATES

        refreshed_old = await exp_repo.get_by_id(old_exp.id)
        refreshed_new = await exp_repo.get_by_id(new_exp.id)
        assert refreshed_old.lifecycle_status == ExperienceLifecycleStatus.SUPERSEDED
        assert refreshed_new.lifecycle_status == ExperienceLifecycleStatus.ACTIVE


# ==============================================================================
# Requirement: CONTRADICTS Lifecycle Transition (Both ACTIVE)
# ==============================================================================

@pytest.mark.asyncio
async def test_evolution_contradicts_preserves_both_active(session_maker):
    """Verify CONTRADICTS creates relationship and preserves both memories as ACTIVE."""
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
        mock_classifier.classify_relationships = AsyncMock(
            return_value={
                old_exp.id: ExperienceEvolutionClassificationResult(
                    candidate_id=old_exp.id,
                    relationship=ExperienceRelationshipType.CONTRADICTS,
                    confidence=0.88,
                    reason="Conflicting career desire.",
                )
            }
        )

        service = ExperienceEvolutionService(
            experience_repo=exp_repo,
            relationship_repo=rel_repo,
            classifier=mock_classifier,
        )

        rels = await service.evolve_experience(experience=new_exp, user_id=user_id)
        assert len(rels) == 1
        assert rels[0].relationship_type == ExperienceRelationshipType.CONTRADICTS

        refreshed_old = await exp_repo.get_by_id(old_exp.id)
        refreshed_new = await exp_repo.get_by_id(new_exp.id)
        assert refreshed_old.lifecycle_status == ExperienceLifecycleStatus.ACTIVE
        assert refreshed_new.lifecycle_status == ExperienceLifecycleStatus.ACTIVE


# ==============================================================================
# Requirement: REINFORCES / RELATED Lifecycle Behavior (Both ACTIVE)
# ==============================================================================

@pytest.mark.asyncio
async def test_evolution_reinforces_and_related_keeps_both_active(session_maker):
    """Verify REINFORCES and RELATED maintain ACTIVE status for all involved memories."""
    user_id = uuid.uuid4()
    dummy_vec = [0.3] * 1536

    async with session_maker() as session:
        exp_repo = SQLAlchemyExperienceRepository(session=session)
        rel_repo = SQLAlchemyExperienceRelationshipRepository(session=session)

        exp_1 = await exp_repo.create(
            Experience(
                content="I like volleyball.",
                source=ExperienceSource.CHAT,
                user_id=str(user_id),
                embedding=dummy_vec,
                embedding_status="COMPLETED",
            )
        )
        exp_2 = await exp_repo.create(
            Experience(
                content="I really enjoy playing volleyball.",
                source=ExperienceSource.CHAT,
                user_id=str(user_id),
                embedding=dummy_vec,
                embedding_status="COMPLETED",
            )
        )

        mock_classifier = MagicMock(spec=ExperienceEvolutionClassifier)
        mock_classifier.classify_relationships = AsyncMock(
            return_value={
                exp_1.id: ExperienceEvolutionClassificationResult(
                    candidate_id=exp_1.id,
                    relationship=ExperienceRelationshipType.REINFORCES,
                    confidence=0.95,
                    reason="Reiterates passion for volleyball.",
                )
            }
        )

        service = ExperienceEvolutionService(
            experience_repo=exp_repo,
            relationship_repo=rel_repo,
            classifier=mock_classifier,
        )

        rels = await service.evolve_experience(experience=exp_2, user_id=user_id)
        assert len(rels) == 1
        assert rels[0].relationship_type == ExperienceRelationshipType.REINFORCES

        refreshed_1 = await exp_repo.get_by_id(exp_1.id)
        refreshed_2 = await exp_repo.get_by_id(exp_2.id)
        assert refreshed_1.lifecycle_status == ExperienceLifecycleStatus.ACTIVE
        assert refreshed_2.lifecycle_status == ExperienceLifecycleStatus.ACTIVE


# ==============================================================================
# Requirement: UNRELATED Behavior (No relationship persisted, no lifecycle change)
# ==============================================================================

@pytest.mark.asyncio
async def test_evolution_unrelated_no_persistence(session_maker):
    """Verify UNRELATED creates no relationship and mutates no lifecycle status."""
    user_id = uuid.uuid4()
    dummy_vec = [0.4] * 1536

    async with session_maker() as session:
        exp_repo = SQLAlchemyExperienceRepository(session=session)
        rel_repo = SQLAlchemyExperienceRelationshipRepository(session=session)

        exp_1 = await exp_repo.create(
            Experience(
                content="I like volleyball.",
                source=ExperienceSource.CHAT,
                user_id=str(user_id),
                embedding=dummy_vec,
                embedding_status="COMPLETED",
            )
        )
        exp_2 = await exp_repo.create(
            Experience(
                content="I visited London in 2022.",
                source=ExperienceSource.CHAT,
                user_id=str(user_id),
                embedding=dummy_vec,
                embedding_status="COMPLETED",
            )
        )

        mock_classifier = MagicMock(spec=ExperienceEvolutionClassifier)
        mock_classifier.classify_relationships = AsyncMock(
            return_value={
                exp_1.id: ExperienceEvolutionClassificationResult(
                    candidate_id=exp_1.id,
                    relationship=ExperienceRelationshipType.UNRELATED,
                    confidence=0.99,
                    reason="Completely different topics.",
                )
            }
        )

        service = ExperienceEvolutionService(
            experience_repo=exp_repo,
            relationship_repo=rel_repo,
            classifier=mock_classifier,
        )

        rels = await service.evolve_experience(experience=exp_2, user_id=user_id)
        assert len(rels) == 0

        all_rels = await rel_repo.get_by_source_id(exp_2.id)
        assert len(all_rels) == 0


# ==============================================================================
# Requirement: Candidate Isolation Across Users
# ==============================================================================

@pytest.mark.asyncio
async def test_candidate_isolation_across_users(session_maker):
    """Verify user A's memories are NEVER considered as candidates for user B."""
    user_a = uuid.uuid4()
    user_b = uuid.uuid4()
    dummy_vec = [0.5] * 1536

    async with session_maker() as session:
        exp_repo = SQLAlchemyExperienceRepository(session=session)
        rel_repo = SQLAlchemyExperienceRelationshipRepository(session=session)

        user_b_exp = await exp_repo.create(
            Experience(
                content="I go to gym at 6 PM.",
                source=ExperienceSource.CHAT,
                user_id=str(user_b),
                embedding=dummy_vec,
                embedding_status="COMPLETED",
            )
        )

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
        mock_classifier.classify_relationships = AsyncMock()

        service = ExperienceEvolutionService(
            experience_repo=exp_repo,
            relationship_repo=rel_repo,
            classifier=mock_classifier,
        )

        rels = await service.evolve_experience(experience=user_a_exp, user_id=user_a)
        assert len(rels) == 0
        mock_classifier.classify_relationships.assert_not_called()

        refreshed_b = await exp_repo.get_by_id(user_b_exp.id)
        assert refreshed_b.lifecycle_status == ExperienceLifecycleStatus.ACTIVE


# ==============================================================================
# Requirement: Graceful Failure in Background Processor
# ==============================================================================

@pytest.mark.asyncio
async def test_background_processor_graceful_evolution_failure(session_maker):
    """Verify LLM failure during evolution does not fail experience creation or vector storage."""
    user_id = uuid.uuid4()
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

    # Failing evolution classifier
    failing_classifier = MagicMock(spec=ExperienceEvolutionClassifier)
    failing_classifier.classify_relationships = AsyncMock(side_effect=RuntimeError("Evolution API crashed"))

    processor = SQLAlchemyBackgroundExperienceProcessor(
        session_factory=session_maker,
        llm_client=mock_llm,
        embedding_provider=mock_embedder,
        evolution_classifier=failing_classifier,
    )

    await processor.process_background_promotion(message=message, user_id=user_id)

    async with session_maker() as session:
        exp_repo = SQLAlchemyExperienceRepository(session=session)
        exp = await exp_repo.get_by_source_message_id(message.id)
        assert exp is not None
        assert exp.content == "Goes to gym at 6 PM"
        assert exp.embedding_status == "COMPLETED"
        assert exp.lifecycle_status == ExperienceLifecycleStatus.ACTIVE
