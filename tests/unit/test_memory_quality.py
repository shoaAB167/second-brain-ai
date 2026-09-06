from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
import uuid

import pytest

from personal_ai.application.memory.dimension_analyzer import QueryDimensionAnalyzer
from personal_ai.application.memory.quality_service import (
    MemoryQualityService,
    _normalize_user_id,
)
from personal_ai.domain.experience import (
    Experience,
    ExperienceEvidenceLevel,
    ExperienceImportance,
    ExperienceLifecycle,
    ExperienceLifecycleStatus,
    ExperienceSource,
    PersonalContextItem,
    PersonalPatternContextItem,
    RetrievalDimension,
)
from personal_ai.domain.pattern.entity import PersonalPattern
from personal_ai.domain.pattern.enums import PatternDomain, PatternStatus


@pytest.fixture
def target_user_id() -> uuid.UUID:
    """Fixture providing a primary test user UUID."""
    return uuid.uuid4()


@pytest.fixture
def other_user_id() -> uuid.UUID:
    """Fixture providing a secondary/cross-user test user UUID."""
    return uuid.uuid4()


@pytest.fixture
def quality_service() -> MemoryQualityService:
    """Fixture providing a standard MemoryQualityService instance."""
    return MemoryQualityService()


def create_test_experience(
    user_id: uuid.UUID,
    content: str,
    importance: ExperienceImportance = ExperienceImportance.MEDIUM,
    lifecycle: ExperienceLifecycle = ExperienceLifecycle.STABLE,
    lifecycle_status: ExperienceLifecycleStatus = ExperienceLifecycleStatus.ACTIVE,
    evidence_level: ExperienceEvidenceLevel = ExperienceEvidenceLevel.EXTRACTED,
    temporal_context: str | None = None,
    created_at: datetime | None = None,
    exp_id: uuid.UUID | None = None,
) -> Experience:
    """Helper to create test Experience instances with defaults."""
    return Experience(
        id=exp_id or uuid.uuid4(),
        user_id=str(user_id),
        content=content,
        source=ExperienceSource.CHAT,
        importance=importance,
        lifecycle=lifecycle,
        lifecycle_status=lifecycle_status,
        evidence_level=evidence_level,
        temporal_context=temporal_context,
        created_at=created_at or datetime.now(timezone.utc),
    )


def create_test_pattern(
    user_id: uuid.UUID,
    description: str,
    domain: PatternDomain = PatternDomain.CAREER,
    confidence: float = 0.85,
    status: PatternStatus = PatternStatus.HYPOTHESIS,
    evidence_count: int = 3,
    pat_id: uuid.UUID | None = None,
) -> PersonalPattern:
    """Helper to create test PersonalPattern instances."""
    return PersonalPattern(
        id=pat_id or uuid.uuid4(),
        user_id=str(user_id),
        description=description,
        domain=domain,
        confidence=confidence,
        status=status,
        evidence_ids=[uuid.uuid4() for _ in range(evidence_count)],
    )


# ==============================================================================
# 1. Duplicate Handling Tests
# ==============================================================================

def test_deduplicate_exact_experience_ids(quality_service, target_user_id):
    """Verify that multiple candidates sharing the exact same experience ID are collapsed."""
    exp = create_test_experience(target_user_id, "I love writing Python code.")
    candidates = [(exp, 0.75), (exp, 0.85), (exp, 0.60)]

    items, _ = quality_service.process_candidates(
        user_id=target_user_id,
        query="What programming language do I like?",
        experience_candidates=candidates,
    )

    assert len(items) == 1
    assert items[0].experience_id == exp.id
    # Highest similarity score is preserved
    assert items[0].similarity == 0.85


def test_deduplicate_identical_content(quality_service, target_user_id):
    """Verify that repeated/identical content memories are collapsed into 1 representative."""
    exp1 = create_test_experience(target_user_id, "I want to become an AI engineer.", importance=ExperienceImportance.LOW)
    exp2 = create_test_experience(target_user_id, "I want to become an AI engineer.", importance=ExperienceImportance.HIGH)
    exp3 = create_test_experience(target_user_id, "  i want to become an ai engineer!  ", importance=ExperienceImportance.MEDIUM)

    candidates = [(exp1, 0.80), (exp2, 0.82), (exp3, 0.79)]

    items, _ = quality_service.process_candidates(
        user_id=target_user_id,
        query="What are my career goals?",
        experience_candidates=candidates,
    )

    # Collapsed to single highest-quality candidate (exp2 has HIGH importance)
    assert len(items) == 1
    assert items[0].experience_id == exp2.id
    assert items[0].importance == "HIGH"


def test_preserve_meaningful_temporal_variants(quality_service, target_user_id):
    """Verify that candidates with distinct temporal markers (e.g. 2024 vs 2026) are NOT collapsed."""
    exp_2024 = create_test_experience(target_user_id, "I wanted to become an AI engineer in 2024.")
    exp_2026 = create_test_experience(target_user_id, "I am actively preparing for AI engineering roles in 2026.")

    candidates = [(exp_2024, 0.85), (exp_2026, 0.88)]

    items, _ = quality_service.process_candidates(
        user_id=target_user_id,
        query="Tell me about my AI career journey.",
        experience_candidates=candidates,
    )

    # Both temporal variants must be preserved
    assert len(items) == 2
    item_contents = [it.content for it in items]
    assert exp_2024.content in item_contents
    assert exp_2026.content in item_contents


def test_preserve_distinct_temporal_context_fields(quality_service, target_user_id):
    """Verify that candidates with differing explicit temporal_context fields are NOT collapsed."""
    exp1 = create_test_experience(target_user_id, "I completed a marathon", temporal_context="Summer 2023")
    exp2 = create_test_experience(target_user_id, "I completed a marathon", temporal_context="Spring 2025")

    candidates = [(exp1, 0.80), (exp2, 0.82)]

    items, _ = quality_service.process_candidates(
        user_id=target_user_id,
        query="Have I run marathons?",
        experience_candidates=candidates,
    )

    assert len(items) == 2


# ==============================================================================
# 2. Lifecycle & Contradiction Handling Tests
# ==============================================================================

def test_active_memory_included(quality_service, target_user_id):
    """Verify ACTIVE lifecycle status memories are included normally."""
    exp = create_test_experience(
        target_user_id,
        "I work at ACME Corp.",
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )

    items, _ = quality_service.process_candidates(
        user_id=target_user_id,
        query="Where do I work?",
        experience_candidates=[(exp, 0.90)],
        is_historical=False,
    )

    assert len(items) == 1
    assert items[0].lifecycle_status == "ACTIVE"


def test_superseded_memory_excluded_in_normal_query(quality_service, target_user_id):
    """Verify SUPERSEDED memories are excluded in normal (non-historical) conversational queries."""
    active_exp = create_test_experience(
        target_user_id,
        "I currently live in London.",
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )
    superseded_exp = create_test_experience(
        target_user_id,
        "I used to live in Paris.",
        lifecycle_status=ExperienceLifecycleStatus.SUPERSEDED,
    )

    candidates = [(active_exp, 0.88), (superseded_exp, 0.85)]

    items, _ = quality_service.process_candidates(
        user_id=target_user_id,
        query="Where do I live?",
        experience_candidates=candidates,
        is_historical=False,
    )

    # Only the active memory is selected
    assert len(items) == 1
    assert items[0].experience_id == active_exp.id
    assert items[0].content == "I currently live in London."


def test_superseded_memory_included_in_historical_query(quality_service, target_user_id):
    """Verify SUPERSEDED memories are permitted when is_historical=True."""
    superseded_exp = create_test_experience(
        target_user_id,
        "I used to live in Paris in 2020.",
        lifecycle_status=ExperienceLifecycleStatus.SUPERSEDED,
    )

    items, _ = quality_service.process_candidates(
        user_id=target_user_id,
        query="Where did I live previously?",
        experience_candidates=[(superseded_exp, 0.85)],
        is_historical=True,
    )

    assert len(items) == 1
    assert items[0].experience_id == superseded_exp.id
    assert items[0].lifecycle_status == "SUPERSEDED"


def test_expired_memory_excluded_in_normal_query(quality_service, target_user_id):
    """Verify EXPIRED memories are excluded in normal active context."""
    expired_exp = create_test_experience(
        target_user_id,
        "Flight booking discount coupon expires tonight.",
        lifecycle_status=ExperienceLifecycleStatus.EXPIRED,
    )

    items, _ = quality_service.process_candidates(
        user_id=target_user_id,
        query="Do I have flight discounts?",
        experience_candidates=[(expired_exp, 0.90)],
        is_historical=False,
    )

    assert len(items) == 0


def test_expired_memory_included_in_historical_query(quality_service, target_user_id):
    """Verify EXPIRED memories are permitted when is_historical=True."""
    expired_exp = create_test_experience(
        target_user_id,
        "Old project deadline was March 1st.",
        lifecycle_status=ExperienceLifecycleStatus.EXPIRED,
    )

    items, _ = quality_service.process_candidates(
        user_id=target_user_id,
        query="What was the old deadline in history?",
        experience_candidates=[(expired_exp, 0.80)],
        is_historical=True,
    )

    assert len(items) == 1


# ==============================================================================
# 3. Personal Patterns Quality & Lifecycle Tests
# ==============================================================================

def test_pattern_active_statuses_included(quality_service, target_user_id):
    """Verify HYPOTHESIS and CONFIRMED pattern statuses are admitted."""
    pat_hyp = create_test_pattern(target_user_id, "User prefers concise Python syntax", status=PatternStatus.HYPOTHESIS)
    pat_conf = create_test_pattern(target_user_id, "User codes late at night", status=PatternStatus.CONFIRMED)

    _, patterns = quality_service.process_candidates(
        user_id=target_user_id,
        query="How do I like to code?",
        experience_candidates=[],
        pattern_candidates=[pat_hyp, pat_conf],
        min_pattern_query_relevance=0.0,
    )

    assert len(patterns) == 2
    statuses = {p.status for p in patterns}
    assert statuses == {"HYPOTHESIS", "CONFIRMED"}


def test_pattern_inactive_statuses_excluded(quality_service, target_user_id):
    """Verify WEAKENED and SUPERSEDED patterns are excluded from active context."""
    pat_weak = create_test_pattern(target_user_id, "User might like C++", status=PatternStatus.WEAKENED)
    pat_sup = create_test_pattern(target_user_id, "User works weekends", status=PatternStatus.SUPERSEDED)

    _, patterns = quality_service.process_candidates(
        user_id=target_user_id,
        query="What do I like?",
        experience_candidates=[],
        pattern_candidates=[pat_weak, pat_sup],
        min_pattern_query_relevance=0.0,
    )

    assert len(patterns) == 0


def test_pattern_and_experience_separation(quality_service, target_user_id):
    """Verify patterns and experiences are never deduplicated against each other or mixed."""
    exp = create_test_experience(target_user_id, "User prefers concise Python syntax")
    pat = create_test_pattern(target_user_id, "User prefers concise Python syntax", status=PatternStatus.CONFIRMED)

    items, patterns = quality_service.process_candidates(
        user_id=target_user_id,
        query="Tell me about my Python coding preferences.",
        experience_candidates=[(exp, 0.88)],
        pattern_candidates=[pat],
        min_pattern_query_relevance=0.0,
    )

    assert len(items) == 1
    assert len(patterns) == 1
    assert isinstance(items[0], PersonalContextItem)
    assert isinstance(patterns[0], PersonalPatternContextItem)
    assert items[0].content == exp.content
    assert patterns[0].description == pat.description


# ==============================================================================
# 4. Freshness & Durability Tests (Stale Memory Handling)
# ==============================================================================

def test_permanent_fact_remains_fresh_regardless_of_age(quality_service, target_user_id):
    """Verify STABLE durability facts (e.g. name, core identity) never decay over time."""
    two_years_ago = datetime.now(timezone.utc) - timedelta(days=730)
    exp = create_test_experience(
        target_user_id,
        "My name is Shoaib.",
        lifecycle=ExperienceLifecycle.STABLE,
        created_at=two_years_ago,
    )

    items, _ = quality_service.process_candidates(
        user_id=target_user_id,
        query="What is my name?",
        experience_candidates=[(exp, 0.90)],
    )

    assert len(items) == 1
    # STABLE lifecycle experiences receive full recency/freshness boost (1.0)
    # Composite: 0.70 * 0.90 + 0.15 * 0 + 0.10 * 0.5 + 0.05 * 1.0 = 0.63 + 0.05 + 0.05 = 0.73
    assert items[0].score >= 0.70


def test_temporary_state_decays_with_age(quality_service, target_user_id):
    """Verify TEMPORARY state memories decay when old."""
    now = datetime.now(timezone.utc)
    recent_temp = create_test_experience(
        target_user_id,
        "I have a fever today.",
        lifecycle=ExperienceLifecycle.TEMPORARY,
        created_at=now - timedelta(days=1),
    )
    stale_temp = create_test_experience(
        target_user_id,
        "I have a mild fever.",
        lifecycle=ExperienceLifecycle.TEMPORARY,
        created_at=now - timedelta(days=60),
    )

    # Both have identical similarity
    items, _ = quality_service.process_candidates(
        user_id=target_user_id,
        query="How am I feeling?",
        experience_candidates=[(stale_temp, 0.80), (recent_temp, 0.80)],
    )

    assert len(items) == 2
    # Recent temporary state is ranked ahead of stale temporary state
    assert items[0].experience_id == recent_temp.id
    assert items[0].score > items[1].score


# ==============================================================================
# 5. Quality Scoring & Relevance Dominance Tests
# ==============================================================================

def test_relevance_dominance_over_importance(quality_service, target_user_id):
    """Verify that vector similarity/relevance is dominant over importance."""
    highly_relevant = create_test_experience(
        target_user_id,
        "I specialize in distributed backend microservices.",
        importance=ExperienceImportance.LOW,
    )
    weakly_relevant = create_test_experience(
        target_user_id,
        "I purchased a blue coffee mug.",
        importance=ExperienceImportance.HIGH,
    )

    candidates = [(weakly_relevant, 0.35), (highly_relevant, 0.88)]

    items, _ = quality_service.process_candidates(
        user_id=target_user_id,
        query="What is my backend architecture background?",
        experience_candidates=candidates,
    )

    assert len(items) == 2
    # Highly relevant low-importance memory easily beats weakly relevant high-importance memory
    assert items[0].experience_id == highly_relevant.id
    assert items[0].score > items[1].score


def test_importance_boost_when_similarity_is_equal(quality_service, target_user_id):
    """Verify HIGH importance ranks above LOW importance when similarity is identical."""
    high_imp = create_test_experience(
        target_user_id,
        "Critical production database credential policy.",
        importance=ExperienceImportance.HIGH,
    )
    low_imp = create_test_experience(
        target_user_id,
        "Casual database note.",
        importance=ExperienceImportance.LOW,
    )

    candidates = [(low_imp, 0.80), (high_imp, 0.80)]

    items, _ = quality_service.process_candidates(
        user_id=target_user_id,
        query="Database policies",
        experience_candidates=candidates,
    )

    assert items[0].experience_id == high_imp.id
    assert items[0].score > items[1].score


# ==============================================================================
# 6. Context Budget Limits Tests
# ==============================================================================

def test_context_budget_bounds_enforced(quality_service, target_user_id):
    """Verify final_limit and pattern_limit strictly bound returned context items."""
    exps = [
        (create_test_experience(target_user_id, f"Memory {i}"), 0.80 - (i * 0.01))
        for i in range(10)
    ]
    pats = [
        create_test_pattern(target_user_id, f"Pattern {i}", status=PatternStatus.CONFIRMED)
        for i in range(6)
    ]

    items, patterns = quality_service.process_candidates(
        user_id=target_user_id,
        query="General query",
        experience_candidates=exps,
        pattern_candidates=pats,
        final_limit=3,
        pattern_limit=2,
        min_pattern_query_relevance=0.0,
    )

    assert len(items) == 3
    assert len(patterns) == 2


# ==============================================================================
# 7. Strict User Isolation & Security Tests (Fail-Closed)
# ==============================================================================

def test_cross_user_candidates_rejected(quality_service, target_user_id, other_user_id):
    """Verify candidates belonging to another user are completely dropped."""
    own_exp = create_test_experience(target_user_id, "User A confidential memory.")
    other_exp = create_test_experience(other_user_id, "User B secret token.")

    candidates = [(own_exp, 0.85), (other_exp, 0.99)]

    items, _ = quality_service.process_candidates(
        user_id=target_user_id,
        query="Tell me my secrets",
        experience_candidates=candidates,
    )

    # Only own memory accepted; other user's candidate rejected fail-closed
    assert len(items) == 1
    assert items[0].experience_id == own_exp.id


def test_missing_or_malformed_user_id_candidate_rejected(quality_service, target_user_id):
    """Verify candidate experiences with missing or malformed user_id fail closed."""
    no_user_exp = create_test_experience(target_user_id, "Memory with no user.")
    no_user_exp.user_id = None

    malformed_user_exp = create_test_experience(target_user_id, "Memory with bad user.")
    malformed_user_exp.user_id = "not-a-valid-uuid"

    candidates = [(no_user_exp, 0.90), (malformed_user_exp, 0.90)]

    items, _ = quality_service.process_candidates(
        user_id=target_user_id,
        query="Test query",
        experience_candidates=candidates,
    )

    assert len(items) == 0


def test_cross_user_pattern_rejected(quality_service, target_user_id, other_user_id):
    """Verify pattern candidate belonging to another user is rejected fail-closed."""
    own_pat = create_test_pattern(target_user_id, "Own pattern hypothesis", status=PatternStatus.CONFIRMED)
    other_pat = create_test_pattern(other_user_id, "Other user pattern", status=PatternStatus.CONFIRMED)

    _, patterns = quality_service.process_candidates(
        user_id=target_user_id,
        query="Test patterns",
        experience_candidates=[],
        pattern_candidates=[own_pat, other_pat],
        min_pattern_query_relevance=0.0,
    )

    assert len(patterns) == 1
    assert patterns[0].pattern_id == own_pat.id


def test_invalid_target_user_id_returns_empty(quality_service):
    """Verify invalid/None target user_id fails closed and returns empty context."""
    items, patterns = quality_service.process_candidates(
        user_id=None,
        query="Query",
        experience_candidates=[],
    )

    assert items == []
    assert patterns == []


# ==============================================================================
# 8. Evidence Level & Quality Selection Tests
# ==============================================================================

def test_evidence_level_preference_in_duplicate_selection(quality_service, target_user_id):
    """Verify that EXPLICIT_USER evidence level is preferred over EXTRACTED/INFERRED when deduplicating."""
    exp_inferred = create_test_experience(
        target_user_id,
        "I prefer dark mode in IDEs.",
        evidence_level=ExperienceEvidenceLevel.INFERRED,
    )
    exp_explicit = create_test_experience(
        target_user_id,
        "I prefer dark mode in IDEs.",
        evidence_level=ExperienceEvidenceLevel.EXPLICIT_USER,
    )

    candidates = [(exp_inferred, 0.80), (exp_explicit, 0.80)]

    items, _ = quality_service.process_candidates(
        user_id=target_user_id,
        query="What theme do I prefer?",
        experience_candidates=candidates,
    )

    assert len(items) == 1
    assert items[0].experience_id == exp_explicit.id
    assert items[0].evidence_level == "EXPLICIT_USER"


def test_pattern_deduplication_keeps_higher_confidence(quality_service, target_user_id):
    """Verify near-duplicate patterns keep the variant with higher confidence / evidence."""
    pat_low_conf = create_test_pattern(
        target_user_id,
        "User prefers working in silence",
        confidence=0.60,
        evidence_count=2,
    )
    pat_high_conf = create_test_pattern(
        target_user_id,
        "  User prefers working in silence.  ",
        confidence=0.92,
        evidence_count=5,
    )

    _, patterns = quality_service.process_candidates(
        user_id=target_user_id,
        query="How do I work best?",
        experience_candidates=[],
        pattern_candidates=[pat_low_conf, pat_high_conf],
        min_pattern_query_relevance=0.0,
    )

    assert len(patterns) == 1
    assert patterns[0].pattern_id == pat_high_conf.id
    assert patterns[0].confidence == 0.92


def test_pattern_query_relevance_gating(quality_service, target_user_id):
    """Verify patterns with query relevance below threshold are filtered out."""
    pat_relevant = create_test_pattern(
        target_user_id,
        "User prefers Python and backend architecture",
        confidence=0.85,
    )
    pat_irrelevant = create_test_pattern(
        target_user_id,
        "User goes hiking in the mountains",
        confidence=0.85,
    )

    _, patterns = quality_service.process_candidates(
        user_id=target_user_id,
        query="Help me write a Python microservice.",
        experience_candidates=[],
        pattern_candidates=[pat_relevant, pat_irrelevant],
        min_pattern_query_relevance=0.20,
    )

    assert len(patterns) == 1
    assert patterns[0].pattern_id == pat_relevant.id


def test_recurring_memory_freshness_decay(quality_service, target_user_id):
    """Verify RECURRING lifecycle memories decay mildly over time."""
    now = datetime.now(timezone.utc)
    fresh_recurring = create_test_experience(
        target_user_id,
        "I attend a weekly team standup.",
        lifecycle=ExperienceLifecycle.RECURRING,
        created_at=now - timedelta(days=5),
    )
    stale_recurring = create_test_experience(
        target_user_id,
        "I attend a weekly team standup.",
        lifecycle=ExperienceLifecycle.RECURRING,
        created_at=now - timedelta(days=120),
    )

    factor_fresh = quality_service._calculate_freshness_factor(fresh_recurring, now)
    factor_stale = quality_service._calculate_freshness_factor(stale_recurring, now)

    assert factor_fresh == 1.0
    assert factor_stale == 0.6


def test_query_dimension_alignment_boost(quality_service, target_user_id):
    """Verify candidate matching detected query dimension receives scoring boost."""
    exp_matching = create_test_experience(
        target_user_id,
        "My goal is to learn Rust in 2026.",
    )
    exp_matching.type = "GOAL"

    exp_non_matching = create_test_experience(
        target_user_id,
        "I like coffee in the morning.",
    )
    exp_non_matching.type = "FACT"

    # Both have same similarity (0.80)
    candidates = [(exp_matching, 0.80), (exp_non_matching, 0.80)]

    items, _ = quality_service.process_candidates(
        user_id=target_user_id,
        query="What are my goals for next year?",
        experience_candidates=candidates,
        detected_dimensions=[RetrievalDimension.GOALS],
    )

    assert len(items) == 2
    assert items[0].experience_id == exp_matching.id
    assert items[0].score > items[1].score


# ==============================================================================
# 9. Failure Safety Tests
# ==============================================================================

def test_failure_safety_on_unexpected_exception(quality_service, target_user_id):
    """Verify that unexpected exceptions during processing fail safely without raising."""
    corrupt_candidate = ("not_an_experience_object", 0.90)

    items, patterns = quality_service.process_candidates(
        user_id=target_user_id,
        query="Query",
        experience_candidates=[corrupt_candidate],
    )

    assert items == []
    assert patterns == []

# ==============================================================================
# 10. Service Integration Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_personal_context_retrieval_service_integrates_quality_service(target_user_id):
    """Verify PersonalContextRetrievalService delegates to MemoryQualityService cleanly."""
    from unittest.mock import AsyncMock
    from personal_ai.application.memory.personal_context_service import PersonalContextRetrievalService
    from personal_ai.infrastructure.embedding import MockEmbeddingProvider

    provider = MockEmbeddingProvider()
    mock_exp_repo = AsyncMock()
    mock_pattern_repo = AsyncMock()

    active_exp = create_test_experience(target_user_id, "I love coding in Python.")
    superseded_exp = create_test_experience(
        target_user_id,
        "I love coding in Perl.",
        lifecycle_status=ExperienceLifecycleStatus.SUPERSEDED,
    )
    duplicate_exp = create_test_experience(target_user_id, "I love coding in Python.")

    mock_exp_repo.search_by_vector.return_value = [
        (active_exp, 0.90),
        (superseded_exp, 0.85),
        (duplicate_exp, 0.88),
    ]

    active_pat = create_test_pattern(
        target_user_id,
        "User prefers Python",
        status=PatternStatus.CONFIRMED,
    )
    weak_pat = create_test_pattern(
        target_user_id,
        "User prefers Perl",
        status=PatternStatus.WEAKENED,
    )
    mock_pattern_repo.get_active_patterns.return_value = [active_pat, weak_pat]

    service = PersonalContextRetrievalService(
        embedding_provider=provider,
        experience_repo=mock_exp_repo,
        pattern_repo=mock_pattern_repo,
    )

    ctx = await service.retrieve_context(
        user_id=target_user_id,
        query="What are my coding preferences?",
        include_historical=False,
    )

    # 1. Superseded memory dropped in non-historical query
    # 2. Duplicate memory collapsed
    # 3. Only active memory remains
    assert len(ctx.items) == 1
    assert ctx.items[0].content == "I love coding in Python."

    # 4. Weakened pattern dropped
    # 5. Confirmed pattern preserved
    assert len(ctx.patterns) == 1
    assert ctx.patterns[0].description == "User prefers Python"

