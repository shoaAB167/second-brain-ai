from datetime import datetime, timedelta, timezone
from typing import List
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
# 1. Anti-Double-Counting & Deduplication Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_proactive_duplicate_experiences_do_not_double_count_evidence():
    """Requirement: Duplicate experience IDs or identical normalized content within an observation set
    are collapsed into 1 record before signal detection and evidence evaluation.
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
    # Exact duplicate experience
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
async def test_proactive_identical_content_with_different_ids_is_deduplicated():
    """Requirement: Near-duplicate content with different IDs is deduplicated to prevent artificial evidence inflation."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    exp1 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I feel exhausted and drained after work.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        emotional_context=EmotionalContext(emotion="tired", intensity=0.8),
        created_at=now - timedelta(days=2),
    )
    # Identical content duplicate under different ID
    exp2 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I feel exhausted and drained after work.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        emotional_context=EmotionalContext(emotion="tired", intensity=0.8),
        created_at=now - timedelta(days=2),
    )

    candidates = await service.analyze(
        user_id=user_id,
        experiences=[exp1, exp2],
        reference_time=now,
    )

    # Since exp2 is deduplicated against exp1, effective count = 1 -> REPEATED_STATE requires >= 2 distinct observations -> NO SIGNAL
    assert candidates == []


# ==============================================================================
# 2. Lifecycle Filtering Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_proactive_expired_and_superseded_experiences_are_filtered_out():
    """Requirement: Expired and superseded experiences are excluded from proactive observation."""
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
# 3. Repeated State Evidence & Observational Non-Diagnostic Language
# ==============================================================================

@pytest.mark.asyncio
async def test_proactive_repeated_state_requires_at_least_two_distinct_observations():
    """Requirement: Single state observation produces NO signal; >= 2 distinct observations produce REPEATED_STATE signal."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    id1 = uuid.uuid4()
    id2 = uuid.uuid4()

    exp1 = Experience(
        id=id1,
        user_id=str(user_id),
        content="Feeling completely drained after meetings.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        emotional_context=EmotionalContext(emotion="tired", intensity=0.7),
        created_at=now - timedelta(days=3),
    )
    exp2 = Experience(
        id=id2,
        user_id=str(user_id),
        content="Woke up with very low energy and fatigued.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        emotional_context=EmotionalContext(emotion="tired", intensity=0.8),
        created_at=now - timedelta(days=1),
    )

    # 1. Single observation -> NO SIGNAL
    cand_single = await service.analyze(
        user_id=user_id,
        experiences=[exp1],
        reference_time=now,
    )
    assert cand_single == []

    # 2. Two distinct observations -> REPEATED_STATE
    cand_multi = await service.analyze(
        user_id=user_id,
        experiences=[exp1, exp2],
        reference_time=now,
    )
    assert len(cand_multi) == 1
    cand = cand_multi[0]
    assert cand.signal_type == ProactiveSignalType.REPEATED_STATE
    assert cand.evidence_count == 2
    assert set(cand.related_experience_ids) == {id1, id2}
    assert "similar state ('tired') was recorded more than once recently" in cand.reason
    assert "Ask how the user is feeling" in cand.suggested_action
    # Must preserve uncertainty and be purely observational (no clinical diagnosis, no personality labels)
    for forbidden in ["diagnos", "clinical", "depression", "disorder", "syndrome", "chronic", "you are always", "personality"]:
        assert forbidden not in cand.reason.lower()
        assert forbidden not in cand.suggested_action.lower()


# ==============================================================================
# 4. Pattern Support Integration Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_proactive_candidate_enriched_by_supporting_pattern():
    """Requirement: Active supporting pattern enriches proactive candidate with supporting_pattern_ids,
    bounded confidence bump, and contextual suggested action.
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

    active_pattern = PersonalPattern(
        id=pattern_id,
        user_id=str(user_id),
        description="User tends to work on AI coding projects at night.",
        domain=PatternDomain.PROJECTS,
        status=PatternStatus.CONFIRMED,
        confidence=0.85,
        evidence_ids=[uuid.uuid4(), uuid.uuid4()],
    )

    candidates = await service.analyze(
        user_id=user_id,
        experiences=[goal_exp],
        patterns=[active_pattern],
        reference_time=now,
    )

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.signal_type == ProactiveSignalType.GOAL_INACTIVITY
    assert cand.evidence_count == 1
    assert cand.supporting_pattern_ids == [pattern_id]
    assert "Based on 1 observation and 1 supporting pattern." in cand.evidence_summary
    assert "work on AI projects at night" in cand.suggested_action or "tonight" in cand.suggested_action
    assert 0.60 <= cand.confidence <= 0.86


@pytest.mark.asyncio
async def test_proactive_inactive_or_rejected_patterns_are_ignored():
    """Requirement: Inactive, weakened, or superseded patterns are not linked as supporting patterns."""
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

    superseded_pattern = PersonalPattern(
        id=pattern_id,
        user_id=str(user_id),
        description="User tends to work on AI coding projects at night.",
        domain=PatternDomain.PROJECTS,
        status=PatternStatus.SUPERSEDED,
        confidence=0.85,
        evidence_ids=[uuid.uuid4(), uuid.uuid4()],
    )

    candidates = await service.analyze(
        user_id=user_id,
        experiences=[goal_exp],
        patterns=[superseded_pattern],
        reference_time=now,
    )

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.supporting_pattern_ids == []
    assert cand.evidence_summary == "Based on 1 observation."


# ==============================================================================
# 5. Current User Statement Precedence Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_proactive_current_user_statement_contradicting_pattern_suppresses_pattern():
    """Requirement: Current explicit user statement strictly supersedes contradictory historical pattern.
    If pattern says 'works at night' but current message says 'switched to morning now', pattern is excluded.
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

    night_pattern = PersonalPattern(
        id=pattern_id,
        user_id=str(user_id),
        description="User tends to work on AI coding projects at night.",
        domain=PatternDomain.PROJECTS,
        status=PatternStatus.CONFIRMED,
        confidence=0.85,
        evidence_ids=[uuid.uuid4(), uuid.uuid4()],
    )

    # User message states routine change / contradiction
    candidates = await service.analyze(
        user_id=user_id,
        experiences=[goal_exp],
        patterns=[night_pattern],
        current_message="I have switched to morning now for all my deep work.",
        reference_time=now,
    )

    assert len(candidates) == 1
    cand = candidates[0]
    # The contradictory night pattern must NOT be linked
    assert cand.supporting_pattern_ids == []
    assert cand.evidence_summary == "Based on 1 observation."
    assert "tonight" not in cand.suggested_action.lower()


# ==============================================================================
# 6. Missed Commitment Evidence Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_proactive_missed_commitment_requires_all_three_evidence_conditions():
    """Requirement: Missed commitment requires (1) explicit commitment, (2) past deadline, (3) missed evidence.
    Future deadline or generic intention produces NO signal.
    """
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    # 1. Future deadline -> NO SIGNAL
    future_commitment = Experience(
        user_id=str(user_id),
        content="Committed to submit the budget report.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        temporal_context="tomorrow at 5 PM",
        created_at=now - timedelta(days=1),
    )
    assert await service.analyze(user_id=user_id, experiences=[future_commitment], reference_time=now) == []

    # 2. Generic intention without commitment phrasing -> NO SIGNAL
    generic_plan = Experience(
        user_id=str(user_id),
        content="I plan to read chapter 4 of the book.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        temporal_context="yesterday, didn't do it",
        created_at=now - timedelta(days=1),
    )
    assert await service.analyze(user_id=user_id, experiences=[generic_plan], reference_time=now) == []

    # 3. Explicit commitment + past deadline + missed evidence -> COMMITMENT_MISSED signal
    valid_id = uuid.uuid4()
    missed_commitment = Experience(
        id=valid_id,
        user_id=str(user_id),
        content="Promised to submit the architecture review proposal.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        temporal_context="yesterday, couldn't finish it in time",
        created_at=now - timedelta(days=2),
    )
    candidates = await service.analyze(user_id=user_id, experiences=[missed_commitment], reference_time=now)
    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.signal_type == ProactiveSignalType.COMMITMENT_MISSED
    assert cand.evidence_count == 1
    assert cand.related_experience_ids == [valid_id]


# ==============================================================================
# 7. Strict User Isolation (Fail Closed)
# ==============================================================================

@pytest.mark.asyncio
async def test_proactive_user_isolation_rejects_cross_user_and_malformed_ids():
    """Requirement: Strict user isolation fails closed on cross-user experiences, patterns, missing or malformed user IDs."""
    service = ProactiveIntelligenceService()
    user_a = uuid.uuid4()
    user_b = uuid.uuid4()
    now = _fixed_now()

    # Experience belonging to user B
    exp_b = Experience(
        id=uuid.uuid4(),
        user_id=str(user_b),
        content="Build project (haven't worked on this for weeks, on hold).",
        type=ExperienceType.GOAL,
        source=ExperienceSource.CHAT,
        temporal_context="no progress recently, on hold",
        created_at=now - timedelta(days=20),
    )

    # Experience with missing user_id
    exp_missing = Experience(
        id=uuid.uuid4(),
        user_id=None,
        content="Build project (haven't worked on this for weeks, on hold).",
        type=ExperienceType.GOAL,
        source=ExperienceSource.CHAT,
        temporal_context="no progress recently, on hold",
        created_at=now - timedelta(days=20),
    )

    # Pattern belonging to user B
    pat_b = PersonalPattern(
        id=uuid.uuid4(),
        user_id=str(user_b),
        description="User works at night.",
        domain=PatternDomain.PROJECTS,
        status=PatternStatus.CONFIRMED,
        confidence=0.85,
    )

    # When analyzing for user A, all foreign/invalid records are dropped -> NO SIGNAL
    candidates = await service.analyze(
        user_id=user_a,
        experiences=[exp_b, exp_missing],
        patterns=[pat_b],
        reference_time=now,
    )
    assert candidates == []


# ==============================================================================
# 8. PersonalContext Integration with Patterns
# ==============================================================================

@pytest.mark.asyncio
async def test_proactive_analysis_via_personal_context_with_patterns():
    """Requirement: ProactiveIntelligenceService accepts PersonalContext with both items and patterns."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    goal_id = uuid.uuid4()
    pat_id = uuid.uuid4()

    ctx_item = PersonalContextItem(
        experience_id=goal_id,
        content="Build personal AI second brain project (haven't worked on this for weeks, on hold).",
        matched_dimensions=[RetrievalDimension.PROJECTS],
        score=0.9,
        type="GOAL",
        domain="projects",
        temporal_context="no progress recently, on hold",
        created_at=now - timedelta(days=20),
    )

    pat_item = PersonalPatternContextItem(
        pattern_id=pat_id,
        description="User tends to work on AI coding projects at night.",
        domain="projects",
        confidence=0.82,
        status="CONFIRMED",
        evidence_count=3,
        score=0.88,
    )

    personal_ctx = PersonalContext(
        user_id=user_id,
        query="What should I work on?",
        items=[ctx_item],
        patterns=[pat_item],
    )

    candidates = await service.analyze(
        user_id=user_id,
        personal_context=personal_ctx,
        reference_time=now,
    )

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.signal_type == ProactiveSignalType.GOAL_INACTIVITY
    assert cand.evidence_count == 1
    assert cand.supporting_pattern_ids == [pat_id]
    assert cand.evidence_summary == "Based on 1 observation and 1 supporting pattern."


# ==============================================================================
# 9. Serialization & Domain Models
# ==============================================================================

def test_proactive_candidate_to_dict_includes_evidence_fields():
    """Requirement: ProactiveCandidate.to_dict() serializes evidence_count, supporting_pattern_ids, and evidence_summary."""
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


# ==============================================================================
# 10. Dependency Provider Wiring
# ==============================================================================

def test_get_proactive_intelligence_service_dependency_wiring():
    """Requirement: get_proactive_intelligence_service() creates service with MemoryQualityService dependency."""
    quality_service = MemoryQualityService()
    service = get_proactive_intelligence_service(memory_quality_service=quality_service)
    assert isinstance(service, ProactiveIntelligenceService)
    assert service._quality_service is quality_service
