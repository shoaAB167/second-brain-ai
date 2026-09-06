from datetime import datetime, timedelta, timezone
from typing import List
from unittest.mock import MagicMock, patch
import uuid

import pytest

from personal_ai.api.dependencies import get_proactive_intelligence_service
from personal_ai.application.memory.quality_service import MemoryQualityService
from personal_ai.application.proactive import ProactiveIntelligenceService
from personal_ai.domain.experience import (
    EmotionalContext,
    Experience,
    ExperienceLifecycle,
    ExperienceLifecycleStatus,
    ExperienceSource,
    ExperienceType,
    PersonalContext,
    PersonalContextItem,
    PersonalPatternContextItem,
    RetrievalDimension,
)
from personal_ai.domain.pattern.entity import PersonalPattern
from personal_ai.domain.pattern.enums import PatternDomain, PatternStatus
from personal_ai.domain.proactive import (
    ProactiveCandidate,
    ProactivePriority,
    ProactiveSignal,
    ProactiveSignalType,
)


def _fixed_now() -> datetime:
    """Return fixed UTC time for deterministic test execution."""
    return datetime(2026, 9, 6, 12, 0, 0, tzinfo=timezone.utc)


# ==============================================================================
# 1. MemoryQualityService Integration & Canonical Usage
# ==============================================================================

@pytest.mark.asyncio
async def test_proactive_analysis_actually_uses_memory_quality_service():
    """Requirement: ProactiveIntelligenceService actually uses MemoryQualityService for filtering/deduplication."""
    spy_quality_service = MemoryQualityService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    goal_exp = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="Build personal AI second brain project (haven't worked on this for weeks, on hold).",
        type=ExperienceType.GOAL,
        source=ExperienceSource.CHAT,
        temporal_context="no progress recently, on hold",
        created_at=now - timedelta(days=20),
    )

    pattern = PersonalPattern(
        id=uuid.uuid4(),
        user_id=str(user_id),
        description="User tends to work on AI coding projects at night.",
        domain=PatternDomain.PROJECTS,
        status=PatternStatus.CONFIRMED,
        confidence=0.85,
    )

    # Use spy on MemoryQualityService methods
    with patch.object(
        spy_quality_service,
        "filter_and_deduplicate_experiences",
        wraps=spy_quality_service.filter_and_deduplicate_experiences,
    ) as mock_exp_filter, patch.object(
        spy_quality_service,
        "filter_and_deduplicate_patterns",
        wraps=spy_quality_service.filter_and_deduplicate_patterns,
    ) as mock_pat_filter:

        service = ProactiveIntelligenceService(quality_service=spy_quality_service)
        candidates = await service.analyze(
            user_id=user_id,
            experiences=[goal_exp],
            patterns=[pattern],
            reference_time=now,
        )

        assert mock_exp_filter.called
        assert mock_pat_filter.called
        assert len(candidates) == 1


# ==============================================================================
# 2. Anti-Double-Counting & Deduplication with Temporal Preservation
# ==============================================================================

@pytest.mark.asyncio
async def test_proactive_duplicate_experiences_do_not_double_count_evidence():
    """Requirement: Duplicate experience IDs or identical normalized content within an observation set
    are collapsed into 1 record via MemoryQualityService.
    """
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    goal_id = uuid.uuid4()
    exp1 = Experience(
        id=goal_id,
        user_id=str(user_id),
        content="Build personal AI second brain project (haven't worked on this for weeks, on hold).",
        type=ExperienceType.GOAL,
        source=ExperienceSource.CHAT,
        temporal_context="no progress recently, on hold",
        created_at=now - timedelta(days=20),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )
    exp2 = Experience(
        id=goal_id,
        user_id=str(user_id),
        content="Build personal AI second brain project (haven't worked on this for weeks, on hold).",
        type=ExperienceType.GOAL,
        source=ExperienceSource.CHAT,
        temporal_context="no progress recently, on hold",
        created_at=now - timedelta(days=20),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )

    candidates = await service.analyze(
        user_id=user_id,
        experiences=[exp1, exp2],
        reference_time=now,
    )

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.signal_type == ProactiveSignalType.GOAL_INACTIVITY
    assert cand.evidence_count == 1
    assert cand.related_experience_ids == [goal_id]
    assert cand.evidence_summary == "Based on 1 observation."


@pytest.mark.asyncio
async def test_proactive_temporal_variants_are_preserved_by_memory_quality():
    """Requirement: Experiences with distinct temporal markers (e.g. in 2024 vs in 2026) are preserved."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    id1 = uuid.uuid4()
    id2 = uuid.uuid4()

    exp1 = Experience(
        id=id1,
        user_id=str(user_id),
        content="Feeling drained after budget reviews in 2024.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        emotional_context=EmotionalContext(emotion="tired", intensity=0.7),
        temporal_context="in 2024",
        created_at=now - timedelta(days=5),
    )
    exp2 = Experience(
        id=id2,
        user_id=str(user_id),
        content="Feeling drained after budget reviews in 2026.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        emotional_context=EmotionalContext(emotion="tired", intensity=0.8),
        temporal_context="in 2026",
        created_at=now - timedelta(days=1),
    )

    candidates = await service.analyze(
        user_id=user_id,
        experiences=[exp1, exp2],
        reference_time=now,
    )

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.signal_type == ProactiveSignalType.REPEATED_STATE
    assert cand.evidence_count == 2
    assert set(cand.related_experience_ids) == {id1, id2}


# ==============================================================================
# 3. Lifecycle Filtering & Stale Evidence
# ==============================================================================

@pytest.mark.asyncio
async def test_proactive_expired_and_superseded_experiences_are_filtered_out():
    """Requirement: Expired and superseded experiences are excluded and do not increase evidence strength."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    expired_goal = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="Pass AWS certification (haven't worked on this, paused).",
        type=ExperienceType.GOAL,
        source=ExperienceSource.CHAT,
        lifecycle_status=ExperienceLifecycleStatus.EXPIRED,
        created_at=now - timedelta(days=40),
    )
    superseded_goal = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="Complete React tutorial (no progress, stalled).",
        type=ExperienceType.GOAL,
        source=ExperienceSource.CHAT,
        lifecycle_status=ExperienceLifecycleStatus.SUPERSEDED,
        created_at=now - timedelta(days=40),
    )

    candidates = await service.analyze(
        user_id=user_id,
        experiences=[expired_goal, superseded_goal],
        reference_time=now,
    )

    assert candidates == []


# ==============================================================================
# 4. Conservative Subject-Specific Pattern Support
# ==============================================================================

@pytest.mark.asyncio
async def test_proactive_unrelated_pattern_does_not_support_signal():
    """Requirement: An unrelated pattern (e.g., 'I exercise in the morning') must NOT support an AI project signal."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    goal_id = uuid.uuid4()
    goal_exp = Experience(
        id=goal_id,
        user_id=str(user_id),
        content="Build personal AI second brain project (haven't worked on this for weeks, on hold).",
        type=ExperienceType.GOAL,
        source=ExperienceSource.CHAT,
        temporal_context="no progress recently, on hold",
        created_at=now - timedelta(days=20),
    )

    # Unrelated pattern sharing generic words like 'morning' / 'exercise'
    unrelated_pattern = PersonalPattern(
        id=uuid.uuid4(),
        user_id=str(user_id),
        description="I exercise in the morning.",
        domain=PatternDomain.FITNESS,
        status=PatternStatus.CONFIRMED,
        confidence=0.90,
    )

    candidates = await service.analyze(
        user_id=user_id,
        experiences=[goal_exp],
        patterns=[unrelated_pattern],
        reference_time=now,
    )

    assert len(candidates) == 1
    cand = candidates[0]
    # Unrelated pattern MUST NOT support the AI project inactivity signal
    assert cand.supporting_pattern_ids == []
    assert cand.evidence_summary == "Based on 1 observation."
    assert "exercise" not in cand.suggested_action.lower()


@pytest.mark.asyncio
async def test_proactive_relevant_pattern_supports_signal():
    """Requirement: A subject-compatible pattern (e.g. 'I usually work on AI coding projects at night')
    MAY support an AI project inactivity signal.
    """
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    goal_id = uuid.uuid4()
    pattern_id = uuid.uuid4()

    goal_exp = Experience(
        id=goal_id,
        user_id=str(user_id),
        content="Build personal AI second brain project (haven't worked on this for weeks, on hold).",
        type=ExperienceType.GOAL,
        source=ExperienceSource.CHAT,
        temporal_context="no progress recently, on hold",
        created_at=now - timedelta(days=20),
    )

    relevant_pattern = PersonalPattern(
        id=pattern_id,
        user_id=str(user_id),
        description="I usually work on AI coding projects at night.",
        domain=PatternDomain.PROJECTS,
        status=PatternStatus.CONFIRMED,
        confidence=0.85,
    )

    candidates = await service.analyze(
        user_id=user_id,
        experiences=[goal_exp],
        patterns=[relevant_pattern],
        reference_time=now,
    )

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.supporting_pattern_ids == [pattern_id]
    assert cand.evidence_summary == "Based on 1 observation and 1 supporting pattern."
    assert cand.evidence_count == 1
    # Candidate language is hypothesis-based and optional
    assert "Your project appears inactive recently." in cand.suggested_action
    assert "past activity suggests" in cand.suggested_action
    assert "Would you like to revisit it tonight?" in cand.suggested_action


# ==============================================================================
# 5. Subject-Specific Contradiction Precedence
# ==============================================================================

@pytest.mark.asyncio
async def test_proactive_company_keyword_does_not_falsely_suppress_pattern():
    """Requirement: 'I switched companies and I'm still working on the project' does NOT suppress night project pattern."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    goal_id = uuid.uuid4()
    pattern_id = uuid.uuid4()

    goal_exp = Experience(
        id=goal_id,
        user_id=str(user_id),
        content="Build personal AI second brain project (haven't worked on this for weeks, on hold).",
        type=ExperienceType.GOAL,
        source=ExperienceSource.CHAT,
        temporal_context="no progress recently, on hold",
        created_at=now - timedelta(days=20),
    )

    night_pattern = PersonalPattern(
        id=pattern_id,
        user_id=str(user_id),
        description="I usually work on AI coding projects at night.",
        domain=PatternDomain.PROJECTS,
        status=PatternStatus.CONFIRMED,
        confidence=0.85,
    )

    # Message contains 'switched' and 'project', but switches companies, NOT the night routine!
    candidates = await service.analyze(
        user_id=user_id,
        experiences=[goal_exp],
        patterns=[night_pattern],
        current_message="I switched companies and I'm still working on the project.",
        reference_time=now,
    )

    assert len(candidates) == 1
    cand = candidates[0]
    # Night pattern must NOT be falsely suppressed!
    assert cand.supporting_pattern_ids == [pattern_id]
    assert cand.evidence_summary == "Based on 1 observation and 1 supporting pattern."


@pytest.mark.asyncio
async def test_proactive_genuine_same_subject_routine_change_suppresses_pattern():
    """Requirement: 'I've switched to doing my project work in the morning now' DOES suppress night pattern."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    goal_id = uuid.uuid4()
    pattern_id = uuid.uuid4()

    goal_exp = Experience(
        id=goal_id,
        user_id=str(user_id),
        content="Build personal AI second brain project (haven't worked on this for weeks, on hold).",
        type=ExperienceType.GOAL,
        source=ExperienceSource.CHAT,
        temporal_context="no progress recently, on hold",
        created_at=now - timedelta(days=20),
    )

    night_pattern = PersonalPattern(
        id=pattern_id,
        user_id=str(user_id),
        description="I usually work on AI coding projects at night.",
        domain=PatternDomain.PROJECTS,
        status=PatternStatus.CONFIRMED,
        confidence=0.85,
    )

    # Genuine routine change to morning for project work
    candidates = await service.analyze(
        user_id=user_id,
        experiences=[goal_exp],
        patterns=[night_pattern],
        current_message="I've switched to doing my project work in the morning now.",
        reference_time=now,
    )

    assert len(candidates) == 1
    cand = candidates[0]
    # Night pattern MUST be suppressed
    assert cand.supporting_pattern_ids == []
    assert cand.evidence_summary == "Based on 1 observation."
    assert "tonight" not in cand.suggested_action.lower()


# ==============================================================================
# 6. Evidence Qualification (Single Observation + Pattern != 2 Observations)
# ==============================================================================

@pytest.mark.asyncio
async def test_proactive_one_observation_plus_pattern_does_not_pretend_to_be_two_observations():
    """Requirement: 1 observation + 1 pattern enriches metadata, but evidence_count remains strictly 1."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    goal_id = uuid.uuid4()
    pattern_id = uuid.uuid4()

    goal_exp = Experience(
        id=goal_id,
        user_id=str(user_id),
        content="Build personal AI second brain project (haven't worked on this for weeks, on hold).",
        type=ExperienceType.GOAL,
        source=ExperienceSource.CHAT,
        temporal_context="no progress recently, on hold",
        created_at=now - timedelta(days=20),
    )

    pattern = PersonalPattern(
        id=pattern_id,
        user_id=str(user_id),
        description="I usually work on AI coding projects at night.",
        domain=PatternDomain.PROJECTS,
        status=PatternStatus.CONFIRMED,
        confidence=0.85,
    )

    candidates = await service.analyze(
        user_id=user_id,
        experiences=[goal_exp],
        patterns=[pattern],
        reference_time=now,
    )

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.evidence_count == 1
    assert len(cand.supporting_pattern_ids) == 1
    assert cand.evidence_summary == "Based on 1 observation and 1 supporting pattern."


@pytest.mark.asyncio
async def test_proactive_multiple_distinct_valid_observations_increase_evidence_count():
    """Requirement: Multiple distinct valid observations increase evidence_count and scale repeated state confidence."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    id1 = uuid.uuid4()
    id2 = uuid.uuid4()
    id3 = uuid.uuid4()

    exp1 = Experience(
        id=id1,
        user_id=str(user_id),
        content="Feeling completely drained after team sync.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        emotional_context=EmotionalContext(emotion="tired", intensity=0.7),
        created_at=now - timedelta(days=5),
    )
    exp2 = Experience(
        id=id2,
        user_id=str(user_id),
        content="Woke up exhausted and low energy today.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        emotional_context=EmotionalContext(emotion="tired", intensity=0.8),
        created_at=now - timedelta(days=3),
    )
    exp3 = Experience(
        id=id3,
        user_id=str(user_id),
        content="Extremely fatigued after late deployment.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        emotional_context=EmotionalContext(emotion="tired", intensity=0.9),
        created_at=now - timedelta(days=1),
    )

    candidates = await service.analyze(
        user_id=user_id,
        experiences=[exp1, exp2, exp3],
        reference_time=now,
    )

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.signal_type == ProactiveSignalType.REPEATED_STATE
    assert cand.evidence_count == 3
    assert cand.confidence >= 0.78
    assert cand.evidence_summary == "Based on 3 recent observations."


# ==============================================================================
# 7. Strict User Isolation (Fail Closed)
# ==============================================================================

@pytest.mark.asyncio
async def test_proactive_user_isolation_for_experiences_and_patterns():
    """Requirement: Strict user isolation drops cross-user experiences, cross-user patterns, and invalid user IDs."""
    service = ProactiveIntelligenceService()
    user_a = uuid.uuid4()
    user_b = uuid.uuid4()
    now = _fixed_now()

    exp_b = Experience(
        id=uuid.uuid4(),
        user_id=str(user_b),
        content="Build project (haven't worked on this for weeks, on hold).",
        type=ExperienceType.GOAL,
        source=ExperienceSource.CHAT,
        temporal_context="no progress recently, on hold",
        created_at=now - timedelta(days=20),
    )

    exp_invalid_user = Experience(
        id=uuid.uuid4(),
        user_id="invalid-uuid-string",
        content="Build project (haven't worked on this for weeks, on hold).",
        type=ExperienceType.GOAL,
        source=ExperienceSource.CHAT,
        temporal_context="no progress recently, on hold",
        created_at=now - timedelta(days=20),
    )

    pat_b = PersonalPattern(
        id=uuid.uuid4(),
        user_id=str(user_b),
        description="I usually work on projects at night.",
        domain=PatternDomain.PROJECTS,
        status=PatternStatus.CONFIRMED,
        confidence=0.85,
    )

    # Analyzing for user A must drop all foreign and invalid records -> NO SIGNAL
    candidates = await service.analyze(
        user_id=user_a,
        experiences=[exp_b, exp_invalid_user],
        patterns=[pat_b],
        reference_time=now,
    )
    assert candidates == []


# ==============================================================================
# 8. Dependency Injection & Serialization
# ==============================================================================

def test_get_proactive_intelligence_service_dependency_wiring():
    """Requirement: get_proactive_intelligence_service() creates service with injected MemoryQualityService."""
    quality_service = MemoryQualityService()
    service = get_proactive_intelligence_service(memory_quality_service=quality_service)
    assert isinstance(service, ProactiveIntelligenceService)
    assert service._quality_service is quality_service


def test_proactive_candidate_to_dict_serialization():
    """Requirement: ProactiveCandidate.to_dict() serializes evidence fields and UUID strings."""
    exp_id = uuid.uuid4()
    pat_id = uuid.uuid4()

    cand = ProactiveCandidate(
        signal_type=ProactiveSignalType.GOAL_INACTIVITY,
        reason="The goal appears inactive based on available context.",
        confidence=0.78,
        priority=ProactivePriority.LOW,
        suggested_action="Ask whether user wants to revisit goal.",
        related_experience_ids=[exp_id],
        evidence_count=1,
        supporting_pattern_ids=[pat_id],
        evidence_summary="Based on 1 observation and 1 supporting pattern.",
    )

    d = cand.to_dict()
    assert d["signal_type"] == "GOAL_INACTIVITY"
    assert d["confidence"] == 0.78
    assert d["priority"] == "LOW"
    assert d["evidence_count"] == 1
    assert d["supporting_pattern_ids"] == [str(pat_id)]
    assert d["evidence_summary"] == "Based on 1 observation and 1 supporting pattern."
    assert d["related_experience_ids"] == [str(exp_id)]
