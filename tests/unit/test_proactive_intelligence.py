from datetime import datetime, timedelta, timezone
from typing import List
import uuid

import pytest

from personal_ai.application.proactive import ProactiveIntelligenceService
from personal_ai.domain.experience import (
    EmotionalContext,
    Experience,
    ExperienceLifecycleStatus,
    ExperienceSource,
    ExperienceType,
    PersonalContext,
    PersonalContextItem,
    RetrievalDimension,
)
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
# 1. No Meaningful Signal Test
# ==============================================================================

@pytest.mark.asyncio
async def test_proactive_no_meaningful_signal_returns_empty_list():
    """Requirement 1: Neutral or unrelated observations produce no proactive candidate."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()

    experiences = [
        Experience(
            user_id=str(user_id),
            content="I like drinking espresso in the morning.",
            type=ExperienceType.PREFERENCE,
            source=ExperienceSource.CHAT,
        ),
        Experience(
            user_id=str(user_id),
            content="Python 3.13 was released recently.",
            type=ExperienceType.FACT,
            source=ExperienceSource.CHAT,
        ),
    ]

    candidates = await service.analyze(
        user_id=user_id,
        experiences=experiences,
        reference_time=_fixed_now(),
    )

    assert candidates == []


# ==============================================================================
# 2. Goal Inactivity Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_proactive_goal_inactivity_detected_for_aged_goal_without_activity():
    """Requirement 2: Active goal with evidence of prolonged inactivity generates GOAL_INACTIVITY."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    goal_id = uuid.uuid4()
    goal_exp = Experience(
        id=goal_id,
        user_id=str(user_id),
        content="Launch the personal AI Second Brain open-source project.",
        type=ExperienceType.GOAL,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=25),  # 25 days old
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )

    candidates = await service.analyze(
        user_id=user_id,
        experiences=[goal_exp],
        reference_time=now,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.signal_type == ProactiveSignalType.GOAL_INACTIVITY
    assert candidate.priority == ProactivePriority.LOW
    assert 0.0 <= candidate.confidence <= 1.0
    assert candidate.confidence == 0.72
    assert "goal appears inactive" in candidate.reason.lower()
    assert "revisit or work on this goal" in candidate.suggested_action.lower()
    assert candidate.related_experience_ids == [goal_id]


@pytest.mark.asyncio
async def test_proactive_active_goal_with_recent_progress_produces_no_inactivity_signal():
    """Requirement 3: Active goal with recent activity/progress produces NO inactivity signal."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    goal_id = uuid.uuid4()
    goal_exp = Experience(
        id=goal_id,
        user_id=str(user_id),
        content="Build personal AI second brain project.",
        type=ExperienceType.GOAL,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=20),
    )

    recent_activity_exp = Experience(
        user_id=str(user_id),
        content="Implemented tool calling framework for personal AI second brain.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=2),  # Active 2 days ago
    )

    candidates = await service.analyze(
        user_id=user_id,
        experiences=[goal_exp, recent_activity_exp],
        reference_time=now,
    )

    # Inactivity signal should NOT be produced
    goal_inactivity_candidates = [c for c in candidates if c.signal_type == ProactiveSignalType.GOAL_INACTIVITY]
    assert len(goal_inactivity_candidates) == 0


@pytest.mark.asyncio
async def test_proactive_active_goal_mentioned_in_current_message_produces_no_inactivity_signal():
    """Requirement 3b: Active goal mentioned in current conversation produces NO inactivity signal."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    goal_exp = Experience(
        user_id=str(user_id),
        content="Complete my AWS certification exam.",
        type=ExperienceType.GOAL,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=30),
    )

    # Current user message references the certification goal
    candidates = await service.analyze(
        user_id=user_id,
        experiences=[goal_exp],
        current_message="I studied 2 hours today for the AWS certification exam.",
        reference_time=now,
    )

    goal_inactivity_candidates = [c for c in candidates if c.signal_type == ProactiveSignalType.GOAL_INACTIVITY]
    assert len(goal_inactivity_candidates) == 0


@pytest.mark.asyncio
async def test_proactive_goal_insufficient_evidence_produces_no_signal():
    """Requirement 4: Brand new goal or goal without inactivity evidence produces NO signal."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    # Goal created yesterday
    recent_goal = Experience(
        user_id=str(user_id),
        content="Learn Rust programming language.",
        type=ExperienceType.GOAL,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=1),
    )

    candidates = await service.analyze(
        user_id=user_id,
        experiences=[recent_goal],
        reference_time=now,
    )

    assert candidates == []


# ==============================================================================
# 3. Missed Commitment Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_proactive_commitment_with_insufficient_temporal_evidence_produces_no_signal():
    """Requirement 5: Commitment with future deadline or no evidence of being missed produces NO signal."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    future_commitment = Experience(
        user_id=str(user_id),
        content="I promised to deliver the quarterly roadmap by next month.",
        type=ExperienceType.GOAL,
        source=ExperienceSource.CHAT,
        temporal_context="next month",
        created_at=now - timedelta(days=2),
    )

    candidates = await service.analyze(
        user_id=user_id,
        experiences=[future_commitment],
        reference_time=now,
    )

    commitment_candidates = [c for c in candidates if c.signal_type == ProactiveSignalType.COMMITMENT_MISSED]
    assert len(commitment_candidates) == 0


@pytest.mark.asyncio
async def test_proactive_missed_commitment_detected_when_temporal_evidence_shows_missed():
    """Requirement 6: Commitment with past deadline / missed evidence produces COMMITMENT_MISSED."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    commitment_id = uuid.uuid4()
    missed_commitment = Experience(
        id=commitment_id,
        user_id=str(user_id),
        content="I committed to finish the client proposal by yesterday, but I didn't manage to complete it.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        temporal_context="yesterday",
        created_at=now - timedelta(hours=12),
    )

    candidates = await service.analyze(
        user_id=user_id,
        experiences=[missed_commitment],
        reference_time=now,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.signal_type == ProactiveSignalType.COMMITMENT_MISSED
    assert candidate.priority == ProactivePriority.MEDIUM
    assert 0.0 <= candidate.confidence <= 1.0
    assert "planned commitment may have been missed" in candidate.reason
    assert "reschedule, adjust, or check in" in candidate.suggested_action
    assert candidate.related_experience_ids == [commitment_id]


# ==============================================================================
# 4. Repeated State Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_proactive_repeated_state_detected_for_multiple_occurrences():
    """Requirement 7: Repeated state (count >= 2) produces REPEATED_STATE signal."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    id1 = uuid.uuid4()
    id2 = uuid.uuid4()

    exp1 = Experience(
        id=id1,
        user_id=str(user_id),
        content="Feeling very tired and drained after meetings.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        emotional_context=EmotionalContext(emotion="tired", intensity=0.8),
        created_at=now - timedelta(days=3),
    )
    exp2 = Experience(
        id=id2,
        user_id=str(user_id),
        content="Exhausted again today, low energy all afternoon.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        emotional_context=EmotionalContext(emotion="tired", intensity=0.7),
        created_at=now - timedelta(days=1),
    )

    candidates = await service.analyze(
        user_id=user_id,
        experiences=[exp1, exp2],
        reference_time=now,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.signal_type == ProactiveSignalType.REPEATED_STATE
    assert candidate.priority == ProactivePriority.MEDIUM
    assert 0.0 <= candidate.confidence <= 1.0
    assert "similar state" in candidate.reason.lower()
    assert "how the user is feeling" in candidate.suggested_action.lower()
    assert set(candidate.related_experience_ids) == {id1, id2}


@pytest.mark.asyncio
async def test_proactive_single_temporary_state_produces_no_repeated_state_signal():
    """Requirement 8: Single temporary state produces NO repeated-state signal."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    single_state = Experience(
        user_id=str(user_id),
        content="I feel a bit tired today after the flight.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        emotional_context=EmotionalContext(emotion="tired", intensity=0.5),
        created_at=now,
    )

    candidates = await service.analyze(
        user_id=user_id,
        experiences=[single_state],
        reference_time=now,
    )

    state_candidates = [c for c in candidates if c.signal_type == ProactiveSignalType.REPEATED_STATE]
    assert len(state_candidates) == 0


# ==============================================================================
# 5. Emotional Intelligence & Safety Boundaries Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_proactive_emotional_context_preserves_uncertainty_and_avoids_diagnosis():
    """Requirement 9 & 10: State signals preserve uncertainty and NEVER diagnose medical conditions or personality traits."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    exp1 = Experience(
        user_id=str(user_id),
        content="Feeling anxious about the presentation.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        emotional_context=EmotionalContext(emotion="anxious", intensity=0.7),
        created_at=now - timedelta(days=4),
    )
    exp2 = Experience(
        user_id=str(user_id),
        content="Woke up feeling nervous and anxious about the demo.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        emotional_context=EmotionalContext(emotion="anxious", intensity=0.8),
        created_at=now - timedelta(days=1),
    )

    candidates = await service.analyze(
        user_id=user_id,
        experiences=[exp1, exp2],
        reference_time=now,
    )

    assert len(candidates) == 1
    cand = candidates[0]

    # Must preserve uncertainty: only mentions observed repetition
    assert "similar state" in cand.reason.lower()

    # Safety checks: Must NOT diagnose or label personality
    forbidden_terms = [
        "depression",
        "chronic anxiety",
        "anxious person",
        "personality trait",
        "disorder",
        "mental illness",
        "you are anxious",
        "lazy",
    ]
    for term in forbidden_terms:
        assert term not in cand.reason.lower(), f"Forbidden diagnostic term '{term}' found in reason"
        assert term not in cand.suggested_action.lower(), f"Forbidden diagnostic term '{term}' found in action"


# ==============================================================================
# 6. Domain Model Invariants & Validation Tests
# ==============================================================================

def test_proactive_confidence_must_be_strictly_bounded_between_0_and_1():
    """Requirement 11: Confidence must be bounded between 0.0 and 1.0; invalid values are rejected."""
    # Valid
    cand = ProactiveCandidate(
        signal_type=ProactiveSignalType.GOAL_INACTIVITY,
        reason="Test reason",
        confidence=0.75,
        priority=ProactivePriority.LOW,
        suggested_action="Test action",
    )
    assert cand.confidence == 0.75

    # Negative confidence rejected
    with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
        ProactiveCandidate(
            signal_type=ProactiveSignalType.GOAL_INACTIVITY,
            reason="Test",
            confidence=-0.1,
            priority=ProactivePriority.LOW,
            suggested_action="Test",
        )

    # Confidence > 1.0 rejected
    with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
        ProactiveCandidate(
            signal_type=ProactiveSignalType.GOAL_INACTIVITY,
            reason="Test",
            confidence=1.05,
            priority=ProactivePriority.LOW,
            suggested_action="Test",
        )

    # Same for ProactiveSignal
    with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
        ProactiveSignal(
            type=ProactiveSignalType.GOAL_INACTIVITY,
            reason="Test",
            confidence=1.5,
        )


def test_proactive_priority_and_signal_type_validation():
    """Requirement 12: Priority must be valid enum or valid string corresponding to enum."""
    cand = ProactiveCandidate(
        signal_type="GOAL_INACTIVITY",  # type: ignore[arg-type]
        reason="Test reason",
        confidence=0.5,
        priority="HIGH",  # type: ignore[arg-type]
        suggested_action="Test action",
    )
    assert cand.priority == ProactivePriority.HIGH
    assert cand.signal_type == ProactiveSignalType.GOAL_INACTIVITY

    with pytest.raises(ValueError, match="Invalid priority"):
        ProactiveCandidate(
            signal_type=ProactiveSignalType.GOAL_INACTIVITY,
            reason="Test",
            confidence=0.5,
            priority="CRITICAL",  # type: ignore[arg-type]
            suggested_action="Test",
        )


def test_proactive_candidate_serialization():
    """Requirement 13: to_dict preserves all fields and stringifies UUIDs."""
    exp_id = uuid.uuid4()
    cand = ProactiveCandidate(
        signal_type=ProactiveSignalType.COMMITMENT_MISSED,
        reason="Deadline passed",
        confidence=0.8,
        priority=ProactivePriority.MEDIUM,
        suggested_action="Ask user",
        related_experience_ids=[exp_id],
    )
    d = cand.to_dict()
    assert d["signal_type"] == "COMMITMENT_MISSED"
    assert d["reason"] == "Deadline passed"
    assert d["confidence"] == 0.8
    assert d["priority"] == "MEDIUM"
    assert d["suggested_action"] == "Ask user"
    assert d["related_experience_ids"] == [str(exp_id)]


# ==============================================================================
# 7. User Isolation Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_proactive_user_isolation_user_a_never_receives_signals_from_user_b():
    """Requirement 14: Analysis strictly ignores experiences belonging to other users."""
    service = ProactiveIntelligenceService()
    user_a = uuid.uuid4()
    user_b = uuid.uuid4()
    now = _fixed_now()

    # User B has an inactive goal and repeated tiredness
    user_b_goal = Experience(
        user_id=str(user_b),
        content="User B inactive personal goal.",
        type=ExperienceType.GOAL,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=30),
    )
    user_b_tired1 = Experience(
        user_id=str(user_b),
        content="User B feeling tired.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        emotional_context=EmotionalContext(emotion="tired"),
    )
    user_b_tired2 = Experience(
        user_id=str(user_b),
        content="User B feeling exhausted.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        emotional_context=EmotionalContext(emotion="tired"),
    )

    # User A only has a neutral fact
    user_a_fact = Experience(
        user_id=str(user_a),
        content="User A reading documentation.",
        type=ExperienceType.FACT,
        source=ExperienceSource.CHAT,
    )

    # Analyzing for User A with mixed experiences
    candidates_a = await service.analyze(
        user_id=user_a,
        experiences=[user_b_goal, user_b_tired1, user_b_tired2, user_a_fact],
        reference_time=now,
    )

    # User A must get NO candidates from User B's data
    assert candidates_a == []


@pytest.mark.asyncio
async def test_proactive_user_isolation_with_personal_context_container():
    """Requirement 14b: PersonalContext belonging to a different user is rejected safely."""
    service = ProactiveIntelligenceService()
    user_a = uuid.uuid4()
    user_b = uuid.uuid4()
    now = _fixed_now()

    # PersonalContext belonging to user B
    context_b = PersonalContext(
        user_id=user_b,
        query="goals",
        detected_dimensions=[RetrievalDimension.GOALS],
        items=[
            PersonalContextItem(
                experience_id=uuid.uuid4(),
                content="User B long inactive goal.",
                type="GOAL",
                created_at=now - timedelta(days=40),
            )
        ],
    )

    # Analyzing for User A with User B's PersonalContext
    candidates = await service.analyze(
        user_id=user_a,
        context=context_b,
        reference_time=now,
    )

    assert candidates == []


# ==============================================================================
# 8. Non-Autonomous & Deterministic Execution Invariants
# ==============================================================================

@pytest.mark.asyncio
async def test_proactive_candidates_do_not_execute_tools_and_no_llm_required():
    """Requirement 15 & 16: Service executes purely synchronously/in-memory with NO tool or LLM dependencies."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    goal_id = uuid.uuid4()
    experiences = [
        Experience(
            id=goal_id,
            user_id=str(user_id),
            content="Run a half marathon by summer.",
            type=ExperienceType.GOAL,
            source=ExperienceSource.CHAT,
            created_at=now - timedelta(days=20),
        )
    ]

    # No LLM client, no tool registry, no network or database calls
    candidates = await service.analyze(
        user_id=user_id,
        experiences=experiences,
        reference_time=now,
    )

    assert len(candidates) == 1
    cand = candidates[0]
    # Pure data container
    assert isinstance(cand, ProactiveCandidate)
    assert cand.signal_type == ProactiveSignalType.GOAL_INACTIVITY


@pytest.mark.asyncio
async def test_proactive_empty_and_invalid_context_handled_safely():
    """Requirement 17: Empty context, None user_id, or empty lists return empty list safely."""
    service = ProactiveIntelligenceService()

    # None user_id
    assert await service.analyze(user_id=None) == []  # type: ignore[arg-type]

    # None context
    user_id = uuid.uuid4()
    assert await service.analyze(user_id=user_id, context=None) == []

    # Empty list
    assert await service.analyze(user_id=user_id, experiences=[]) == []

    # Empty PersonalContext
    empty_ctx = PersonalContext(user_id=user_id, query="test", items=[])
    assert await service.analyze(user_id=user_id, context=empty_ctx) == []


# ==============================================================================
# 9. Deduplication Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_proactive_duplicate_evidence_does_not_create_duplicate_candidates():
    """Requirement 18: In-memory per-analysis deduplication prevents identical candidates from duplicate evidence."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    goal_id = uuid.uuid4()
    # Same goal supplied twice in context
    goal_exp = Experience(
        id=goal_id,
        user_id=str(user_id),
        content="Complete the master thesis defense.",
        type=ExperienceType.GOAL,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=30),
    )

    candidates = await service.analyze(
        user_id=user_id,
        experiences=[goal_exp, goal_exp],
        reference_time=now,
    )

    # Exactly 1 candidate produced, not 2
    assert len(candidates) == 1
    assert candidates[0].signal_type == ProactiveSignalType.GOAL_INACTIVITY
