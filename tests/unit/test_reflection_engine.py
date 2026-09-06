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
# 2. Temporary Inactivity vs Pattern Weakening Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_reflection_one_day_inactivity_does_not_create_weakening():
    """Requirement: A 1-day pause (e.g. 'haven't worked on AI project for 1 day') does NOT create PATTERN_WEAKENING."""
    service = ReflectionService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    one_day_pause = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I haven't worked on my AI project for 1 day.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=1),
    )

    pattern = PersonalPattern(
        user_id=str(user_id),
        description="You tend to work on AI projects regularly.",
        domain=PatternDomain.PROJECTS,
        status=PatternStatus.CONFIRMED,
        confidence=0.85,
    )

    reflections = await service.analyze(
        user_id=user_id,
        experiences=[one_day_pause],
        patterns=[pattern],
        time_window_days=30,
        reference_time=now,
    )

    assert reflections == []


@pytest.mark.asyncio
async def test_reflection_short_temporary_pause_does_not_create_weakening():
    """Requirement: Short temporary pauses (e.g. 'paused briefly', 'was busy yesterday') do NOT produce PATTERN_WEAKENING."""
    service = ReflectionService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    exp1 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I paused the project briefly.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=2),
    )
    exp2 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I was busy yesterday with chores.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=1),
    )

    pattern = PersonalPattern(
        user_id=str(user_id),
        description="You tend to work on AI projects regularly.",
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


@pytest.mark.asyncio
async def test_reflection_repeated_meaningful_inactivity_creates_weakening():
    """Requirement: Repeated sustained inactivity / stalled progress across multiple observations produces PATTERN_WEAKENING."""
    service = ReflectionService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    id1 = uuid.uuid4()
    id2 = uuid.uuid4()
    pattern_id = uuid.uuid4()

    exp1 = Experience(
        id=id1,
        user_id=str(user_id),
        content="I haven't worked on my AI project for weeks, progress is stalled.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        temporal_context="for weeks, stalled",
        created_at=now - timedelta(days=7),
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
        description="User tends to work on AI projects regularly.",
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
# 4. Temporal Window Validation & Bounds Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_reflection_time_window_validation_bounds():
    """Requirement: time_window_days must be an integer between 1 and 90; invalid values raise ValueError."""
    service = ReflectionService()
    user_id = uuid.uuid4()

    # Invalid: 0
    with pytest.raises(ValueError, match="time_window_days must be an integer between 1 and 90"):
        await service.analyze(user_id=user_id, experiences=[], patterns=[], time_window_days=0)

    # Invalid: -1
    with pytest.raises(ValueError, match="time_window_days must be an integer between 1 and 90"):
        await service.analyze(user_id=user_id, experiences=[], patterns=[], time_window_days=-1)

    # Invalid: 91
    with pytest.raises(ValueError, match="time_window_days must be an integer between 1 and 90"):
        await service.analyze(user_id=user_id, experiences=[], patterns=[], time_window_days=91)

    # Valid: 30, 60, 90 (empty inputs return empty list without error)
    assert await service.analyze(user_id=user_id, experiences=[], patterns=[], time_window_days=30) == []
    assert await service.analyze(user_id=user_id, experiences=[], patterns=[], time_window_days=60) == []
    assert await service.analyze(user_id=user_id, experiences=[], patterns=[], time_window_days=90) == []


@pytest.mark.asyncio
async def test_reflection_old_evidence_outside_window_is_ignored():
    """Requirement: Observations outside the time_window_days are excluded from reflection analysis."""
    service = ReflectionService()
    user_id = uuid.uuid4()
    now = _fixed_now()

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


# ==============================================================================
# 5. Confidence Calculation & Deduplication Fail-Closed Tests
# ==============================================================================

def test_reflection_confidence_helper_signature_and_scale():
    """Requirement: _calculate_confidence has signature (self, evidence_count: int) without unused base parameter."""
    service = ReflectionService()
    assert service._calculate_confidence(1) == 0.52
    assert service._calculate_confidence(2) == 0.55
    assert service._calculate_confidence(3) == 0.65
    assert service._calculate_confidence(4) == 0.72
    assert service._calculate_confidence(5) == 0.80
    assert service._calculate_confidence(6) == 0.82
    assert service._calculate_confidence(10) == 0.86


@pytest.mark.asyncio
async def test_reflection_deduplication_fails_closed_on_missing_pattern_id():
    """Requirement: Reflection without pattern_ids is rejected during deduplication and not assigned a random ID."""
    service = ReflectionService()
    user_id = uuid.uuid4()

    # Raw reflection without pattern_ids
    invalid_ref = Reflection(
        user_id=user_id,
        type=ReflectionType.PATTERN_REINFORCEMENT,
        observation="Test observation",
        confidence=0.70,
        pattern_ids=[],  # Missing pattern ID
    )

    # Calling reflect_on_pattern with valid inputs will always produce pattern_ids,
    # but let's test that the deduplication loop skips reflections with empty pattern_ids
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(service, "reflect_on_pattern", lambda *args, **kwargs: invalid_ref)

        pat = PersonalPattern(
            user_id=str(user_id),
            description="Test pattern",
            domain=PatternDomain.GENERAL,
        )
        exp = Experience(
            user_id=str(user_id),
            content="Test content",
            type=ExperienceType.EVENT,
            source=ExperienceSource.CHAT,
        )

        reflections = await service.analyze(user_id=user_id, experiences=[exp], patterns=[pat])
        # Invalid reflection with missing pattern_ids must be skipped/rejected
        assert reflections == []


# ==============================================================================
# 6. Safety Invariants & Non-Diagnostic Language
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
        content="I haven't worked on my AI project for weeks, stalled.",
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
# 8. Domain Model Validation & Serialization
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

    with pytest.raises(ValueError, match="time_window_days must be an integer between 1 and 90"):
        Reflection(user_id=user_id, type=ReflectionType.PATTERN_REINFORCEMENT, observation="Test", confidence=0.5, time_window_days=0)

    with pytest.raises(ValueError, match="time_window_days must be an integer between 1 and 90"):
        Reflection(user_id=user_id, type=ReflectionType.PATTERN_REINFORCEMENT, observation="Test", confidence=0.5, time_window_days=95)


# ==============================================================================
# 9. Dependency Wiring
# ==============================================================================

def test_get_reflection_service_dependency_wiring():
    """Requirement: get_reflection_service() constructs ReflectionService with MemoryQualityService."""
    quality_service = MemoryQualityService()
    service = get_reflection_service(memory_quality_service=quality_service)
    assert isinstance(service, ReflectionService)
    assert service._quality_service is quality_service
