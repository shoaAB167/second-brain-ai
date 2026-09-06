from datetime import datetime, timedelta, timezone
from typing import List
import uuid

import pytest

from personal_ai.api.dependencies import get_reflection_service
from personal_ai.application.memory.quality_service import MemoryQualityService
from personal_ai.application.reflection import ReflectionService
from personal_ai.domain.experience import (
    EmotionalContext,
    Experience,
    ExperienceLifecycleStatus,
    ExperienceSource,
    ExperienceType,
)
from personal_ai.domain.pattern.entity import PersonalPattern
from personal_ai.domain.pattern.enums import PatternDomain, PatternStatus
from personal_ai.domain.reflection import Reflection, ReflectionStatus, ReflectionType


def _fixed_now() -> datetime:
    """Return fixed UTC reference time for deterministic testing."""
    return datetime(2026, 9, 6, 12, 0, 0, tzinfo=timezone.utc)


# ==============================================================================
# 1. Pattern Reinforcement Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_reflection_reinforcement_with_two_supporting_experiences():
    """Requirement: >= 2 distinct supporting experiences produce PATTERN_REINFORCEMENT."""
    service = ReflectionService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    id1 = uuid.uuid4()
    id2 = uuid.uuid4()
    pattern_id = uuid.uuid4()

    exp1 = Experience(
        id=id1,
        user_id=str(user_id),
        content="I studied AI at 10pm yesterday.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        temporal_context="10pm yesterday",
        created_at=now - timedelta(days=3),
    )
    exp2 = Experience(
        id=id2,
        user_id=str(user_id),
        content="I worked on my AI project last night.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        temporal_context="last night",
        created_at=now - timedelta(days=1),
    )

    pattern = PersonalPattern(
        id=pattern_id,
        user_id=str(user_id),
        description="You tend to study AI at night.",
        domain=PatternDomain.LEARNING,
        status=PatternStatus.CONFIRMED,
        confidence=0.80,
    )

    reflections = await service.analyze(
        user_id=user_id,
        experiences=[exp1, exp2],
        patterns=[pattern],
        time_window_days=30,
        reference_time=now,
    )

    assert len(reflections) == 1
    ref = reflections[0]
    assert ref.type == ReflectionType.PATTERN_REINFORCEMENT
    assert ref.user_id == user_id
    assert set(ref.evidence_ids) == {id1, id2}
    assert ref.pattern_ids == [pattern_id]
    assert "continues to support your established pattern" in ref.observation
    assert 0.55 <= ref.confidence <= 0.88


@pytest.mark.asyncio
async def test_reflection_reinforcement_requires_distinct_evidence_ids():
    """Requirement: Exact duplicate experience does not double-count; 1 distinct experience returns no reflection."""
    service = ReflectionService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    goal_id = uuid.uuid4()
    exp1 = Experience(
        id=goal_id,
        user_id=str(user_id),
        content="I studied AI at 10pm yesterday.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        temporal_context="10pm yesterday",
        created_at=now - timedelta(days=3),
    )
    exp2 = Experience(
        id=goal_id,
        user_id=str(user_id),
        content="I studied AI at 10pm yesterday.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        temporal_context="10pm yesterday",
        created_at=now - timedelta(days=3),
    )

    pattern = PersonalPattern(
        user_id=str(user_id),
        description="You tend to study AI at night.",
        domain=PatternDomain.LEARNING,
        status=PatternStatus.CONFIRMED,
        confidence=0.80,
    )

    reflections = await service.analyze(
        user_id=user_id,
        experiences=[exp1, exp2],
        patterns=[pattern],
        time_window_days=30,
        reference_time=now,
    )

    assert reflections == []


# ==============================================================================
# 2. Pattern Weakening Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_reflection_weakening_with_observed_inactivity():
    """Requirement: >= 2 relevant observations indicating stalled progress or pause produce PATTERN_WEAKENING."""
    service = ReflectionService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    id1 = uuid.uuid4()
    id2 = uuid.uuid4()
    pattern_id = uuid.uuid4()

    exp1 = Experience(
        id=id1,
        user_id=str(user_id),
        content="I haven't worked on my AI project for 5 days.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        temporal_context="for 5 days",
        created_at=now - timedelta(days=5),
    )
    exp2 = Experience(
        id=id2,
        user_id=str(user_id),
        content="AI project is on hold with zero progress recently.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        temporal_context="on hold",
        created_at=now - timedelta(days=1),
    )

    pattern = PersonalPattern(
        id=pattern_id,
        user_id=str(user_id),
        description="User tends to work on AI projects at night.",
        domain=PatternDomain.PROJECTS,
        status=PatternStatus.CONFIRMED,
        confidence=0.85,
    )

    reflections = await service.analyze(
        user_id=user_id,
        experiences=[exp1, exp2],
        patterns=[pattern],
        time_window_days=30,
        reference_time=now,
    )

    assert len(reflections) == 1
    ref = reflections[0]
    assert ref.type == ReflectionType.PATTERN_WEAKENING
    assert set(ref.evidence_ids) == {id1, id2}
    assert ref.pattern_ids == [pattern_id]
    assert "activity appears lower than the established pattern" in ref.observation
    assert 0.55 <= ref.confidence <= 0.80


# ==============================================================================
# 3. Pattern Inconsistency vs Pattern Change Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_reflection_single_conflicting_observation_produces_inconsistency_not_change():
    """Requirement: Exactly 1 conflicting observation produces PATTERN_INCONSISTENCY, NOT PATTERN_CHANGE."""
    service = ReflectionService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    id1 = uuid.uuid4()
    pattern_id = uuid.uuid4()

    # Single conflicting observation (studied in morning instead of established night pattern)
    exp1 = Experience(
        id=id1,
        user_id=str(user_id),
        content="I studied AI in the morning today at 8am.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        temporal_context="8am morning",
        created_at=now - timedelta(days=1),
    )

    pattern = PersonalPattern(
        id=pattern_id,
        user_id=str(user_id),
        description="You tend to study AI at night.",
        domain=PatternDomain.LEARNING,
        status=PatternStatus.CONFIRMED,
        confidence=0.80,
    )

    reflections = await service.analyze(
        user_id=user_id,
        experiences=[exp1],
        patterns=[pattern],
        time_window_days=30,
        reference_time=now,
    )

    assert len(reflections) == 1
    ref = reflections[0]
    assert ref.type == ReflectionType.PATTERN_INCONSISTENCY
    assert ref.evidence_ids == [id1]
    assert ref.pattern_ids == [pattern_id]
    assert "differs from your established pattern" in ref.observation
    assert ref.confidence <= 0.60


@pytest.mark.asyncio
async def test_reflection_change_requires_multiple_consistent_recent_observations():
    """Requirement: >= 2 consistent shifting observations produce PATTERN_CHANGE."""
    service = ReflectionService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    id1 = uuid.uuid4()
    id2 = uuid.uuid4()
    pattern_id = uuid.uuid4()

    exp1 = Experience(
        id=id1,
        user_id=str(user_id),
        content="I studied AI every morning at 7am this week.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        temporal_context="7am morning",
        created_at=now - timedelta(days=4),
    )
    exp2 = Experience(
        id=id2,
        user_id=str(user_id),
        content="Worked on AI project in morning sessions repeatedly.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        temporal_context="morning",
        created_at=now - timedelta(days=1),
    )

    pattern = PersonalPattern(
        id=pattern_id,
        user_id=str(user_id),
        description="You tend to study AI at night.",
        domain=PatternDomain.LEARNING,
        status=PatternStatus.CONFIRMED,
        confidence=0.80,
    )

    reflections = await service.analyze(
        user_id=user_id,
        experiences=[exp1, exp2],
        patterns=[pattern],
        time_window_days=30,
        reference_time=now,
    )

    assert len(reflections) == 1
    ref = reflections[0]
    assert ref.type == ReflectionType.PATTERN_CHANGE
    assert set(ref.evidence_ids) == {id1, id2}
    assert ref.pattern_ids == [pattern_id]
    assert "may be shifting from your established pattern" in ref.observation
    assert "morning" in ref.observation.lower()
    assert 0.55 <= ref.confidence <= 0.88


# ==============================================================================
# 4. Temporal Window & Durable Pattern Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_reflection_old_evidence_outside_window_is_ignored():
    """Requirement: Observations outside the time_window_days are excluded from reflection analysis."""
    service = ReflectionService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    # Evidence from 45 days ago (outside 30-day default window)
    old_exp1 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I studied AI at 10pm yesterday.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=45),
    )
    old_exp2 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I worked on my AI project last night.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=40),
    )

    pattern = PersonalPattern(
        user_id=str(user_id),
        description="You tend to study AI at night.",
        domain=PatternDomain.LEARNING,
        status=PatternStatus.CONFIRMED,
        confidence=0.80,
    )

    reflections = await service.analyze(
        user_id=user_id,
        experiences=[old_exp1, old_exp2],
        patterns=[pattern],
        time_window_days=30,
        reference_time=now,
    )

    assert reflections == []


@pytest.mark.asyncio
async def test_reflection_durable_pattern_older_than_window_remains_eligible():
    """Requirement: Patterns established months ago remain eligible to be compared against recent evidence."""
    service = ReflectionService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    # Pattern created 120 days ago
    old_pattern = PersonalPattern(
        id=uuid.uuid4(),
        user_id=str(user_id),
        description="You tend to study AI at night.",
        domain=PatternDomain.LEARNING,
        status=PatternStatus.CONFIRMED,
        confidence=0.85,
        created_at=now - timedelta(days=120),
    )

    # Recent evidence within 30 days
    exp1 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I studied AI at 10pm yesterday.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=3),
    )
    exp2 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I worked on my AI project last night.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=1),
    )

    reflections = await service.analyze(
        user_id=user_id,
        experiences=[exp1, exp2],
        patterns=[old_pattern],
        time_window_days=30,
        reference_time=now,
    )

    assert len(reflections) == 1
    assert reflections[0].type == ReflectionType.PATTERN_REINFORCEMENT


@pytest.mark.asyncio
async def test_reflection_custom_time_window_configuration():
    """Requirement: Custom time_window_days (e.g. 60 days) accepts evidence within that expanded window."""
    service = ReflectionService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    exp1 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I studied AI at 10pm yesterday.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=45),
    )
    exp2 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I worked on my AI project last night.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=40),
    )

    pattern = PersonalPattern(
        user_id=str(user_id),
        description="You tend to study AI at night.",
        domain=PatternDomain.LEARNING,
        status=PatternStatus.CONFIRMED,
        confidence=0.80,
    )

    # 1. 30 days window -> excluded
    ref_30 = await service.analyze(
        user_id=user_id,
        experiences=[exp1, exp2],
        patterns=[pattern],
        time_window_days=30,
        reference_time=now,
    )
    assert ref_30 == []

    # 2. 60 days window -> included
    ref_60 = await service.analyze(
        user_id=user_id,
        experiences=[exp1, exp2],
        patterns=[pattern],
        time_window_days=60,
        reference_time=now,
    )
    assert len(ref_60) == 1
    assert ref_60[0].type == ReflectionType.PATTERN_REINFORCEMENT


# ==============================================================================
# 5. Subject Specificity & Unrelated Keyword Filtering
# ==============================================================================

@pytest.mark.asyncio
async def test_reflection_unrelated_evidence_ignored():
    """Requirement: Unrelated activities (e.g. watched AI documentary) do not count toward study/work pattern."""
    service = ReflectionService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    exp1 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I watched an interesting AI documentary on television.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=2),
    )
    exp2 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I had breakfast at 8am.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=1),
    )

    pattern = PersonalPattern(
        user_id=str(user_id),
        description="User tends to work on AI coding projects at night.",
        domain=PatternDomain.PROJECTS,
        status=PatternStatus.CONFIRMED,
        confidence=0.85,
    )

    reflections = await service.analyze(
        user_id=user_id,
        experiences=[exp1, exp2],
        patterns=[pattern],
        time_window_days=30,
        reference_time=now,
    )

    assert reflections == []


# ==============================================================================
# 6. Observational Safety & Non-Diagnostic Language
# ==============================================================================

@pytest.mark.asyncio
async def test_reflection_safety_invariants_strictly_prevent_diagnoses_and_personality_claims():
    """Requirement: Generated observations strictly avoid psychological/medical diagnoses, personality traits, and motivation assumptions."""
    service = ReflectionService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    exp1 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I haven't worked on my AI project for 5 days.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=5),
    )
    exp2 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="AI project is on hold with zero progress recently.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=1),
    )

    pattern = PersonalPattern(
        user_id=str(user_id),
        description="User tends to work on AI projects at night.",
        domain=PatternDomain.PROJECTS,
        status=PatternStatus.CONFIRMED,
        confidence=0.85,
    )

    reflections = await service.analyze(
        user_id=user_id,
        experiences=[exp1, exp2],
        patterns=[pattern],
        time_window_days=30,
        reference_time=now,
    )

    assert len(reflections) == 1
    obs = reflections[0].observation.lower()

    forbidden_terms = [
        "lazy", "lost motivation", "unmotivated", "depressed", "depression",
        "disorder", "syndrome", "chronic", "personality", "you are always",
        "you never", "diagnos", "clinical", "inconsistent person", "failing",
    ]
    for term in forbidden_terms:
        assert term not in obs, f"Forbidden diagnostic or judgment term found: '{term}' in '{obs}'"


# ==============================================================================
# 7. Strict User Isolation (Fail Closed)
# ==============================================================================

@pytest.mark.asyncio
async def test_reflection_user_isolation_rejects_cross_user_and_invalid_user_ids():
    """Requirement: Strict user isolation drops foreign experiences, foreign patterns, and invalid user IDs."""
    service = ReflectionService()
    user_a = uuid.uuid4()
    user_b = uuid.uuid4()
    now = _fixed_now()

    # Experience belonging to user B
    exp_b = Experience(
        id=uuid.uuid4(),
        user_id=str(user_b),
        content="I studied AI at 10pm yesterday.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=3),
    )

    # Pattern belonging to user B
    pat_b = PersonalPattern(
        id=uuid.uuid4(),
        user_id=str(user_b),
        description="You tend to study AI at night.",
        domain=PatternDomain.LEARNING,
        status=PatternStatus.CONFIRMED,
        confidence=0.80,
    )

    # Analyzing for user A must drop user B's records -> NO REFLECTIONS
    assert await service.analyze(user_id=user_a, experiences=[exp_b], patterns=[pat_b], reference_time=now) == []

    # Invalid user_id must fail closed safely
    assert await service.analyze(user_id=None, experiences=[exp_b], patterns=[pat_b]) == []
    assert await service.analyze(user_id="invalid-uuid", experiences=[exp_b], patterns=[pat_b]) == []


# ==============================================================================
# 8. Pattern Status Filtering
# ==============================================================================

@pytest.mark.asyncio
async def test_reflection_superseded_and_weakened_patterns_are_ignored():
    """Requirement: Weakened or superseded patterns are excluded from reflection analysis."""
    service = ReflectionService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    exp1 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I studied AI at 10pm yesterday.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=3),
    )
    exp2 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I worked on my AI project last night.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=1),
    )

    superseded_pat = PersonalPattern(
        user_id=str(user_id),
        description="You tend to study AI at night.",
        domain=PatternDomain.LEARNING,
        status=PatternStatus.SUPERSEDED,
        confidence=0.80,
    )

    weakened_pat = PersonalPattern(
        user_id=str(user_id),
        description="You tend to study AI at night.",
        domain=PatternDomain.LEARNING,
        status=PatternStatus.WEAKENED,
        confidence=0.35,
    )

    reflections = await service.analyze(
        user_id=user_id,
        experiences=[exp1, exp2],
        patterns=[superseded_pat, weakened_pat],
        reference_time=now,
    )

    assert reflections == []


# ==============================================================================
# 9. Domain Model Validation & Serialization
# ==============================================================================

def test_reflection_domain_model_validation_and_serialization():
    """Requirement: Reflection model validates fields, bounds confidence, deduplicates IDs, and serializes cleanly."""
    user_id = uuid.uuid4()
    ev1 = uuid.uuid4()
    pat1 = uuid.uuid4()

    # 1. Valid Reflection
    ref = Reflection(
        user_id=user_id,
        type=ReflectionType.PATTERN_REINFORCEMENT,
        observation="Recent activity continues to support your established pattern.",
        confidence=0.74,
        evidence_ids=[ev1, ev1],  # Duplicate ID should be deduplicated
        pattern_ids=[pat1, pat1],
        time_window_days=30,
        domain="LEARNING",
    )

    assert ref.evidence_ids == [ev1]
    assert ref.pattern_ids == [pat1]
    assert ref.confidence == 0.74
    assert ref.type == ReflectionType.PATTERN_REINFORCEMENT

    # 2. Serialization roundtrip
    d = ref.to_dict()
    assert d["type"] == "PATTERN_REINFORCEMENT"
    assert d["user_id"] == str(user_id)
    assert d["confidence"] == 0.74
    assert d["evidence_ids"] == [str(ev1)]
    assert d["pattern_ids"] == [str(pat1)]

    reconstructed = Reflection.from_dict(d)
    assert reconstructed.id == ref.id
    assert reconstructed.user_id == ref.user_id
    assert reconstructed.type == ref.type
    assert reconstructed.confidence == ref.confidence

    # 3. Validation failures
    with pytest.raises(ValueError, match="user_id must be a UUID"):
        Reflection(user_id=None, type=ReflectionType.PATTERN_REINFORCEMENT, observation="Test", confidence=0.5)

    with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
        Reflection(user_id=user_id, type=ReflectionType.PATTERN_REINFORCEMENT, observation="Test", confidence=1.5)

    with pytest.raises(ValueError, match="observation must be a non-empty string"):
        Reflection(user_id=user_id, type=ReflectionType.PATTERN_REINFORCEMENT, observation="", confidence=0.5)

    with pytest.raises(ValueError, match="time_window_days must be a positive integer"):
        Reflection(user_id=user_id, type=ReflectionType.PATTERN_REINFORCEMENT, observation="Test", confidence=0.5, time_window_days=0)


# ==============================================================================
# 10. Dependency Wiring
# ==============================================================================

def test_get_reflection_service_dependency_wiring():
    """Requirement: get_reflection_service() constructs ReflectionService with MemoryQualityService."""
    quality_service = MemoryQualityService()
    service = get_reflection_service(memory_quality_service=quality_service)
    assert isinstance(service, ReflectionService)
    assert service._quality_service is quality_service
