from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from personal_ai.application.memory.dimension_analyzer import QueryDimensionAnalyzer
from personal_ai.application.memory.personal_context_builder import PersonalContextBuilder
from personal_ai.application.memory.personal_context_service import PersonalContextRetrievalService
from personal_ai.config.settings import Settings, get_settings
from personal_ai.core.exceptions import AppException
from personal_ai.db.models import Base
from personal_ai.db.repositories.base import ConversationRepository
from personal_ai.db.repositories.sqlalchemy_experience_repository import SQLAlchemyExperienceRepository
from personal_ai.domain.experience import (
    EmotionalContext,
    Experience,
    ExperienceEvidenceLevel,
    ExperienceImportance,
    ExperienceLifecycle,
    ExperienceLifecycleStatus,
    ExperienceSource,
    ExperienceStatus,
    ExperienceType,
    PersonInvolved,
    PersonalContext,
    PersonalContextItem,
    RetrievalDimension,
)
from personal_ai.infrastructure.embedding import MockEmbeddingProvider
from personal_ai.llm.client import LLMClient
from personal_ai.llm.models import LLMMessage, LLMResponse
from personal_ai.models.chat import ChatRequest
from personal_ai.services.chat_service import ChatService


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
# 1. Query & Context Dimension Analysis (High-Confidence vs Empty)
# ==============================================================================

def test_query_dimension_analyzer_single_dimensions():
    """Verify single dimension detection across all PR #18 taxonomy dimensions."""
    analyzer = QueryDimensionAnalyzer()

    assert RetrievalDimension.GOALS in analyzer.analyze_query("I want to reach 30 LPA as a software architect.")
    assert RetrievalDimension.PROJECTS in analyzer.analyze_query("How is my Second Brain AI project coming along?")
    assert RetrievalDimension.PREFERENCES in analyzer.analyze_query("I prefer dark mode in VS Code.")
    assert RetrievalDimension.HABITS in analyzer.analyze_query("At what time do I usually go to gym?")
    assert RetrievalDimension.RELATIONSHIPS in analyzer.analyze_query("Where does my sister live?")
    assert RetrievalDimension.EMOTIONS in analyzer.analyze_query("I feel anxious about my job interview.")
    assert RetrievalDimension.CURRENT_STATE in analyzer.analyze_query("How am I feeling today?")
    assert RetrievalDimension.CONSTRAINTS in analyzer.analyze_query("What are my constraints?")
    assert RetrievalDimension.PAST_EXPERIENCES in analyzer.analyze_query("What did I do in the past?")
    assert RetrievalDimension.PERSONALITY in analyzer.analyze_query("Who am I and what is my name?")


def test_ordinary_conversational_query_returns_no_dimensions():
    """Requirement: Ordinary conversational queries must NOT trigger broad accidental dimensions."""
    analyzer = QueryDimensionAnalyzer()

    assert analyzer.analyze_query("Can you help me solve this bug?") == []
    assert analyzer.analyze_query("Tell me about Python decorators.") == []
    assert analyzer.analyze_query("How does async/await work in FastAPI?") == []
    assert analyzer.analyze_query("Write a quick sorting algorithm.") == []


def test_conservative_historical_query_detection():
    """Requirement: Historical detection only triggers on high-confidence historical inquiries."""
    analyzer = QueryDimensionAnalyzer()

    # High-confidence historical patterns -> True
    assert analyzer.is_historical_query("what happened before...") is True
    assert analyzer.is_historical_query("what did I say previously about my goals?") is True
    assert analyzer.is_historical_query("remember when we talked about London?") is True
    assert analyzer.is_historical_query("what was I doing last year?") is True
    assert analyzer.is_historical_query("Where did I live in the past?") is True

    # Ordinary queries containing words like 'before', 'like', 'work' -> False
    assert analyzer.is_historical_query("I need to finish this before 5 PM.") is False
    assert analyzer.is_historical_query("I like to work on AI.") is False
    assert analyzer.is_historical_query("What is the state of this function?") is False


def test_query_dimension_analyzer_decision_support_query():
    """Verify decision-support query detects multi-dimensional intent."""
    analyzer = QueryDimensionAnalyzer()
    query = "Should I continue working on this project?"
    dimensions = analyzer.analyze_query(query)

    assert RetrievalDimension.DECISIONS in dimensions
    assert RetrievalDimension.PROJECTS in dimensions
    assert RetrievalDimension.GOALS in dimensions


def test_query_dimension_analyzer_conversation_context():
    """Verify short-term conversation context enhances dimension detection."""
    analyzer = QueryDimensionAnalyzer()
    context = [
        LLMMessage(role="user", content="I am feeling anxious about my upcoming presentation."),
        LLMMessage(role="assistant", content="Take a deep breath."),
    ]
    dimensions = analyzer.analyze_query("How can I prepare better?", conversation_context=context)
    assert RetrievalDimension.EMOTIONS in dimensions


def test_dimension_matching_from_experience():
    """Verify mapping Experience attributes to domain RetrievalDimensions."""
    analyzer = QueryDimensionAnalyzer()

    exp_goal = Experience(
        content="Reach 30 LPA",
        source=ExperienceSource.CHAT,
        type=ExperienceType.GOAL,
        importance=ExperienceImportance.HIGH,
    )
    assert RetrievalDimension.GOALS in analyzer.match_experience_dimensions(exp_goal)

    exp_rich = Experience(
        content="Failed interview with Alex",
        source=ExperienceSource.CHAT,
        type=ExperienceType.EVENT,
        emotional_context=EmotionalContext(emotion="fear", intensity=0.8),
        people_involved=[PersonInvolved(name="Alex", role="interviewer")],
    )
    matched = analyzer.match_experience_dimensions(exp_rich)
    assert RetrievalDimension.PAST_EXPERIENCES in matched
    assert RetrievalDimension.EMOTIONS in matched
    assert RetrievalDimension.RELATIONSHIPS in matched


# ==============================================================================
# 2. Goal & Project Retrieval
# ==============================================================================

@pytest.mark.asyncio
async def test_relevant_goal_and_project_retrieval(session_maker):
    """Requirement: Retrieve relevant goals and projects with dimension boosting."""
    user_id = uuid.uuid4()
    provider = MockEmbeddingProvider(dimensions=1536)

    async with session_maker() as session:
        repo = SQLAlchemyExperienceRepository(session=session)

        # Create goal experience
        vec_goal = await provider.embed("Career goal to reach 30 LPA in AI engineering")
        await repo.create(
            Experience(
                content="Goal is to reach 30 LPA in AI engineering",
                source=ExperienceSource.CHAT,
                user_id=str(user_id),
                type=ExperienceType.GOAL,
                domain="career",
                importance=ExperienceImportance.HIGH,
                embedding=vec_goal,
                embedding_status="COMPLETED",
            )
        )

        # Create project experience
        vec_proj = await provider.embed("Building Second Brain AI with FastAPI and pgvector")
        await repo.create(
            Experience(
                content="Building Second Brain AI assistant",
                source=ExperienceSource.CHAT,
                user_id=str(user_id),
                type=ExperienceType.PROJECT,
                domain="projects",
                importance=ExperienceImportance.HIGH,
                embedding=vec_proj,
                embedding_status="COMPLETED",
            )
        )

        service = PersonalContextRetrievalService(embedding_provider=provider, experience_repo=repo)
        context = await service.retrieve_context(user_id=user_id, query="What is my career goal?")

        assert not context.is_empty
        assert RetrievalDimension.GOALS in context.detected_dimensions
        top_item = context.items[0]
        assert "30 LPA" in top_item.content
        assert top_item.type == "GOAL"
        assert RetrievalDimension.GOALS in top_item.matched_dimensions


# ==============================================================================
# 3. Semantic Retrieval Works When Dimensions are Empty
# ==============================================================================

@pytest.mark.asyncio
async def test_semantic_retrieval_works_when_dimensions_empty(session_maker):
    """Requirement: When no dimension is detected, semantic vector search still retrieves relevant memories."""
    user_id = uuid.uuid4()
    provider = MockEmbeddingProvider(dimensions=1536)

    async with session_maker() as session:
        repo = SQLAlchemyExperienceRepository(session=session)

        vec = await provider.embed("Python asyncio programming with task groups")
        await repo.create(
            Experience(
                content="Learned Python asyncio task groups pattern",
                source=ExperienceSource.CHAT,
                user_id=str(user_id),
                type=ExperienceType.FACT,
                domain="skills",
                embedding=vec,
                embedding_status="COMPLETED",
            )
        )

        service = PersonalContextRetrievalService(embedding_provider=provider, experience_repo=repo)
        # Query has no dimension keywords
        context = await service.retrieve_context(user_id=user_id, query="Python asyncio programming with task groups")

        assert not context.is_empty
        assert context.detected_dimensions == []
        assert "asyncio task groups" in context.items[0].content


# ==============================================================================
# 4. Emotional Context Retrieval & Surfacing
# ==============================================================================

@pytest.mark.asyncio
async def test_emotional_context_retrieval_surfaces_all_attributes(session_maker):
    """Requirement: For emotional queries, surface emotion, intensity, trigger, need, impact, and evidence level."""
    user_id = uuid.uuid4()
    provider = MockEmbeddingProvider(dimensions=1536)

    async with session_maker() as session:
        repo = SQLAlchemyExperienceRepository(session=session)

        vec = await provider.embed("Failed technical interview and feeling anxious about AI readiness")
        await repo.create(
            Experience(
                content="Failed an AI technical interview",
                source=ExperienceSource.CHAT,
                user_id=str(user_id),
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
                temporal_context="yesterday",
                evidence_level=ExperienceEvidenceLevel.EXPLICIT_USER,
                embedding=vec,
                embedding_status="COMPLETED",
            )
        )

        service = PersonalContextRetrievalService(embedding_provider=provider, experience_repo=repo)
        context = await service.retrieve_context(user_id=user_id, query="How did I feel after my interview yesterday?")

        assert not context.is_empty
        assert RetrievalDimension.EMOTIONS in context.detected_dimensions
        item = context.items[0]
        assert item.emotional_context is not None
        assert item.emotional_context["emotion"] == "anxiety"
        assert item.emotional_context["intensity"] == 0.85
        assert item.emotional_context["trigger"] == "technical interview failure"
        assert item.emotional_context["need"] == "preparation guidance"
        assert item.emotional_context["impact"] == "questioning readiness"
        assert item.temporal_context == "yesterday"
        assert item.evidence_level == "EXPLICIT_USER"
        assert item.people_involved == [{"name": "Alex", "role": "interviewer"}]


# ==============================================================================
# 5. Preference & Habit Retrieval
# ==============================================================================

@pytest.mark.asyncio
async def test_preference_and_habit_retrieval(session_maker):
    """Requirement: Retrieve habits and preferences accurately."""
    user_id = uuid.uuid4()
    provider = MockEmbeddingProvider(dimensions=1536)

    async with session_maker() as session:
        repo = SQLAlchemyExperienceRepository(session=session)

        vec_habit = await provider.embed("Usually goes to gym at 6 PM")
        await repo.create(
            Experience(
                content="Usually goes to gym around 6 PM",
                source=ExperienceSource.CHAT,
                user_id=str(user_id),
                type=ExperienceType.HABIT,
                domain="fitness",
                lifecycle=ExperienceLifecycle.RECURRING,
                embedding=vec_habit,
                embedding_status="COMPLETED",
            )
        )

        service = PersonalContextRetrievalService(embedding_provider=provider, experience_repo=repo)
        context = await service.retrieve_context(user_id=user_id, query="At what time do I usually go to gym?")

        assert not context.is_empty
        assert RetrievalDimension.HABITS in context.detected_dimensions
        assert "6 PM" in context.items[0].content


# ==============================================================================
# 6. Multi-Signal Bounded Context & Simplified Ranking
# ==============================================================================

@pytest.mark.asyncio
async def test_bounded_final_context_and_importance_weighting(session_maker):
    """Requirement: Candidate limit and final limit are strictly enforced with similarity dominance."""
    user_id = uuid.uuid4()
    provider = MockEmbeddingProvider(dimensions=1536)

    async with session_maker() as session:
        repo = SQLAlchemyExperienceRepository(session=session)

        # Create 10 experiences with identical base vector to strictly verify importance scoring boost
        base_vec = await provider.embed("career development items")
        for i in range(10):
            imp = ExperienceImportance.HIGH if i == 7 else ExperienceImportance.LOW
            await repo.create(
                Experience(
                    content=f"Career observation {i}",
                    source=ExperienceSource.CHAT,
                    user_id=str(user_id),
                    type=ExperienceType.FACT,
                    domain="career",
                    importance=imp,
                    embedding=base_vec,
                    embedding_status="COMPLETED",
                )
            )

        service = PersonalContextRetrievalService(embedding_provider=provider, experience_repo=repo)
        context = await service.retrieve_context(
            user_id=user_id,
            query="career development items",
            candidate_limit=8,
            final_limit=3,
        )

        assert len(context.items) == 3
        assert context.total_candidates == 8
        # HIGH importance item should rank highest among equal-similarity candidates
        top_item = context.items[0]
        assert top_item.importance == "HIGH"
        assert top_item.content == "Career observation 7"


# ==============================================================================
# 7. Conservative Lifecycle Filtering & Historical Memories
# ==============================================================================

@pytest.mark.asyncio
async def test_lifecycle_filtering_active_vs_historical(session_maker):
    """Requirement: Active memories default for standard queries; historical retrieved when past is queried."""
    user_id = uuid.uuid4()
    provider = MockEmbeddingProvider(dimensions=1536)

    async with session_maker() as session:
        repo = SQLAlchemyExperienceRepository(session=session)

        vec_old = await provider.embed("Lived in London in the past")
        await repo.create(
            Experience(
                content="Lived in London",
                source=ExperienceSource.CHAT,
                user_id=str(user_id),
                type=ExperienceType.FACT,
                domain="location",
                lifecycle_status=ExperienceLifecycleStatus.SUPERSEDED,
                embedding=vec_old,
                embedding_status="COMPLETED",
            )
        )

        vec_new = await provider.embed("Lives in Bangalore currently")
        await repo.create(
            Experience(
                content="Lives in Bangalore",
                source=ExperienceSource.CHAT,
                user_id=str(user_id),
                type=ExperienceType.FACT,
                domain="location",
                lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
                embedding=vec_new,
                embedding_status="COMPLETED",
            )
        )

        service = PersonalContextRetrievalService(embedding_provider=provider, experience_repo=repo)

        # Standard current query -> ACTIVE only
        curr_context = await service.retrieve_context(user_id=user_id, query="Where do I currently live?")
        contents = [item.content for item in curr_context.items]
        assert "Lives in Bangalore" in contents
        assert "Lived in London" not in contents

        # Conservative historical query -> Includes SUPERSEDED
        hist_context = await service.retrieve_context(user_id=user_id, query="Where did I live in the past?")
        hist_contents = [item.content for item in hist_context.items]
        assert "Lived in London" in hist_contents

        # Explicit include_historical=True -> Includes SUPERSEDED
        explicit_hist_context = await service.retrieve_context(user_id=user_id, query="Where do I live?", include_historical=True)
        assert any(item.content == "Lived in London" for item in explicit_hist_context.items)


# ==============================================================================
# 8. Strict User Isolation
# ==============================================================================

@pytest.mark.asyncio
async def test_strict_user_isolation(session_maker):
    """Requirement: Retrieval MUST NEVER return another user's personal memories."""
    user_a = uuid.uuid4()
    user_b = uuid.uuid4()
    provider = MockEmbeddingProvider(dimensions=1536)

    async with session_maker() as session:
        repo = SQLAlchemyExperienceRepository(session=session)

        vec = await provider.embed("Confidential secret project info")
        await repo.create(
            Experience(
                content="User A's confidential financial goal is 50 LPA",
                source=ExperienceSource.CHAT,
                user_id=str(user_a),
                type=ExperienceType.GOAL,
                embedding=vec,
                embedding_status="COMPLETED",
            )
        )

        service = PersonalContextRetrievalService(embedding_provider=provider, experience_repo=repo)

        # User B queries about financial goal
        context_b = await service.retrieve_context(user_id=user_b, query="financial goal LPA")
        assert context_b.is_empty
        assert len(context_b.items) == 0


# ==============================================================================
# 9. Legacy Experiences & Missing Fields Compatibility
# ==============================================================================

@pytest.mark.asyncio
async def test_legacy_experiences_compatibility(session_maker):
    """Requirement: Legacy experiences without rich/emotional fields retrieve cleanly without errors."""
    user_id = uuid.uuid4()
    provider = MockEmbeddingProvider(dimensions=1536)

    async with session_maker() as session:
        repo = SQLAlchemyExperienceRepository(session=session)

        vec = await provider.embed("Likes drinking green tea")
        await repo.create(
            Experience(
                content="Likes drinking green tea",
                source=ExperienceSource.CHAT,
                user_id=str(user_id),
                type=ExperienceType.PREFERENCE,
                embedding=vec,
                embedding_status="COMPLETED",
            )
        )

        service = PersonalContextRetrievalService(embedding_provider=provider, experience_repo=repo)
        context = await service.retrieve_context(user_id=user_id, query="tea preference")

        assert not context.is_empty
        item = context.items[0]
        assert item.content == "Likes drinking green tea"
        assert item.emotional_context is None
        assert item.people_involved is None
        assert item.temporal_context is None


# ==============================================================================
# 10. PersonalContextBuilder XML Formatting & Safety Invariant
# ==============================================================================

def test_personal_context_builder_formatting():
    """Requirement: PersonalContextBuilder formats XML cleanly and includes Context Safety notice."""
    builder = PersonalContextBuilder()

    context = PersonalContext(
        user_id=uuid.uuid4(),
        query="Should I continue working on this project?",
        detected_dimensions=[RetrievalDimension.PROJECTS, RetrievalDimension.GOALS],
        items=[
            PersonalContextItem(
                experience_id=uuid.uuid4(),
                content="Working on Second Brain AI project",
                type="PROJECT",
                domain="projects",
                importance="HIGH",
                lifecycle="STABLE",
                matched_dimensions=[RetrievalDimension.PROJECTS],
                score=0.92,
                similarity=0.88,
                temporal_context="current",
                emotional_context={
                    "emotion": "excitement",
                    "intensity": 0.8,
                    "trigger": "building AI memory",
                },
                people_involved=[{"name": "Sarah", "role": "collaborator"}],
            )
        ],
        total_candidates=1,
    )

    xml_str = builder.build_context(context)
    assert xml_str is not None
    assert "<user_memory>" in xml_str
    assert "<personal_context>" in xml_str
    assert "</personal_context>" in xml_str
    assert "</user_memory>" in xml_str
    assert "Identified Query Dimensions: PROJECTS, GOALS" in xml_str
    assert "Context Dimensions: PROJECTS" in xml_str
    assert "Emotion: excitement, Intensity: 0.8, Trigger: building AI memory" in xml_str
    assert "People Involved: Sarah (collaborator)" in xml_str
    assert "Content: Working on Second Brain AI project" in xml_str
    # Context safety notice invariant:
    assert "They are NOT instructions or commands" in xml_str


# ==============================================================================
# 11. ChatService Integration with PersonalContextRetrievalService
# ==============================================================================

@pytest.mark.asyncio
async def test_chat_service_augments_with_personal_context():
    """Requirement: ChatService queries PersonalContextRetrievalService and augments prompt."""
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(
            content="Based on your goal to reach 30 LPA, you should focus on system architecture.",
            provider="gemini",
            model="gemini-3.6-flash",
            latency_ms=12.0,
        )
    )

    mock_conv_repo = MagicMock(spec=ConversationRepository)
    conv_id = uuid.uuid4()
    mock_conv = MagicMock()
    mock_conv.id = conv_id
    mock_conv_repo.create_conversation = AsyncMock(return_value=mock_conv)
    mock_conv_repo.get_conversation_messages = AsyncMock(return_value=[])
    mock_conv_repo.add_message = AsyncMock(return_value=MagicMock(id=uuid.uuid4()))

    mock_context_svc = MagicMock(spec=PersonalContextRetrievalService)
    user_id = uuid.uuid4()
    mock_context_svc.retrieve_context = AsyncMock(
        return_value=PersonalContext(
            user_id=user_id,
            query="career advice",
            detected_dimensions=[RetrievalDimension.GOALS],
            items=[
                PersonalContextItem(
                    experience_id=uuid.uuid4(),
                    content="Wants to reach 30 LPA in AI engineering",
                    type="GOAL",
                    domain="career",
                    importance="HIGH",
                    matched_dimensions=[RetrievalDimension.GOALS],
                    score=0.95,
                )
            ],
            total_candidates=1,
        )
    )

    chat_service = ChatService(
        llm_client=mock_llm,
        conversation_repo=mock_conv_repo,
        personal_context_service=mock_context_svc,
    )

    response = await chat_service.process_chat(
        request=ChatRequest(message="What career advice do you have for me?"),
        user_id=user_id,
    )

    assert response.response == "Based on your goal to reach 30 LPA, you should focus on system architecture."
    mock_context_svc.retrieve_context.assert_called_once()
    # Check that LLM received system message containing <personal_context>
    call_messages = mock_llm.generate_response.call_args.kwargs["messages"]
    system_msg = next((m for m in call_messages if m.role == "system"), None)
    assert system_msg is not None
    assert "<personal_context>" in system_msg.content
    assert "30 LPA" in system_msg.content


@pytest.mark.asyncio
async def test_chat_service_fail_safe_on_retrieval_failure():
    """Requirement: ChatService gracefully continues without breaking if PersonalContextRetrievalService fails."""
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(
            content="General response without memory context.",
            provider="gemini",
            model="gemini-3.6-flash",
            latency_ms=10.0,
        )
    )

    mock_conv_repo = MagicMock(spec=ConversationRepository)
    conv_id = uuid.uuid4()
    mock_conv = MagicMock()
    mock_conv.id = conv_id
    mock_conv_repo.create_conversation = AsyncMock(return_value=mock_conv)
    mock_conv_repo.get_conversation_messages = AsyncMock(return_value=[])
    mock_conv_repo.add_message = AsyncMock(return_value=MagicMock(id=uuid.uuid4()))

    mock_context_svc = MagicMock(spec=PersonalContextRetrievalService)
    user_id = uuid.uuid4()
    mock_context_svc.retrieve_context = AsyncMock(side_effect=Exception("Database connection timeout"))

    chat_service = ChatService(
        llm_client=mock_llm,
        conversation_repo=mock_conv_repo,
        personal_context_service=mock_context_svc,
    )

    # Should not raise exception
    response = await chat_service.process_chat(
        request=ChatRequest(message="Hello there!"),
        user_id=user_id,
    )

    assert response.response == "General response without memory context."


@pytest.mark.asyncio
async def test_ranking_remains_dominated_by_similarity(session_maker):
    """Requirement: Semantic similarity must remain the dominant ranking signal (e.g. 0.70 weight)."""
    user_id = uuid.uuid4()
    mock_repo = MagicMock(spec=SQLAlchemyExperienceRepository)
    provider = MockEmbeddingProvider(dimensions=1536)

    # Item A: High semantic similarity (0.95), no boosts (0.70 * 0.95 = 0.665)
    exp_a = Experience(
        id=uuid.uuid4(),
        content="Direct semantic answer with high relevance",
        source=ExperienceSource.CHAT,
        user_id=str(user_id),
        type=ExperienceType.FACT,
        importance=ExperienceImportance.LOW,
        created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )

    # Item B: Moderate similarity (0.40), all boosts maxed (0.70 * 0.40 + 0.15 + 0.10 + 0.05 = 0.58)
    exp_b = Experience(
        id=uuid.uuid4(),
        content="Wants to reach career goal",
        source=ExperienceSource.CHAT,
        user_id=str(user_id),
        type=ExperienceType.GOAL,
        importance=ExperienceImportance.HIGH,
        created_at=datetime.now(timezone.utc),
    )

    mock_repo.search_by_vector = AsyncMock(
        return_value=[
            (exp_a, 0.95),
            (exp_b, 0.40),
        ]
    )

    service = PersonalContextRetrievalService(embedding_provider=provider, experience_repo=mock_repo)
    context = await service.retrieve_context(user_id=user_id, query="What is my goal?")

    assert len(context.items) == 2
    # Item A should be ranked #1 because similarity dominates all other combined boosts
    assert context.items[0].content == exp_a.content
    assert context.items[1].content == exp_b.content
    assert context.items[0].score > context.items[1].score


@pytest.mark.asyncio
async def test_configurable_ranking_weights(session_maker):
    """Requirement: Scoring weights must be configurable via settings."""
    user_id = uuid.uuid4()
    mock_repo = MagicMock(spec=SQLAlchemyExperienceRepository)
    provider = MockEmbeddingProvider(dimensions=1536)

    exp = Experience(
        id=uuid.uuid4(),
        content="Configurable weight test memory",
        source=ExperienceSource.CHAT,
        user_id=str(user_id),
        type=ExperienceType.GOAL,
        importance=ExperienceImportance.HIGH,
        created_at=datetime.now(timezone.utc),
    )

    mock_repo.search_by_vector = AsyncMock(return_value=[(exp, 1.0)])

    # Test with custom settings: similarity=0.50, dimension=0.20, importance=0.20, recency=0.10
    custom_settings = Settings(
        personal_context_weight_similarity=0.50,
        personal_context_weight_dimension=0.20,
        personal_context_weight_importance=0.20,
        personal_context_weight_recency=0.10,
    )

    with patch("personal_ai.application.memory.personal_context_service.get_settings", return_value=custom_settings):
        service = PersonalContextRetrievalService(embedding_provider=provider, experience_repo=mock_repo)
        context = await service.retrieve_context(user_id=user_id, query="My goals")

        # 0.50 * 1.0 + 0.20 * 1.0 (goal query matches GOAL dimension) + 0.20 * 1.0 (HIGH) + 0.10 * 1.0 (recent) = 1.00
        assert len(context.items) == 1
        assert context.items[0].score == 1.0

