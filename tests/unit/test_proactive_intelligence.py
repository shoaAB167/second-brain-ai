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
# 1. Goal Inactivity Tests (Conservative Evidence vs False Positives)
# ==============================================================================

@pytest.mark.asyncio
async def test_proactive_old_goal_alone_without_inactivity_evidence_produces_no_signal():
    """Requirement: Goal age alone (e.g. 30 days old) does NOT constitute inactivity; returns NO SIGNAL."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    old_goal = Experience(
        user_id=str(user_id),
        content="I want to become an AI engineer.",
        type=ExperienceType.GOAL,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=35),  # 35 days old, but no inactivity notes
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )

    candidates = await service.analyze(
        user_id=user_id,
        experiences=[old_goal],
        reference_time=now,
    )

    # Must prefer NO SIGNAL over FALSE POSITIVE
    assert candidates == []


@pytest.mark.asyncio
async def test_proactive_long_term_goal_without_inactivity_evidence_produces_no_signal():
    """Requirement: Long-term life goals legitimately remain untouched without generating false inactivity."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    long_term_goal = Experience(
        user_id=str(user_id),
        content="Buy a house in Pune within the next 3 years.",
        type=ExperienceType.GOAL,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=60),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )

    candidates = await service.analyze(
        user_id=user_id,
        experiences=[long_term_goal],
        reference_time=now,
    )

    assert candidates == []


@pytest.mark.asyncio
async def test_proactive_explicit_inactivity_language_generates_goal_inactivity_signal():
    """Requirement: Explicit evidence of stalled progress or pause generates GOAL_INACTIVITY signal."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    goal_id = uuid.uuid4()
    stalled_goal = Experience(
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
        experiences=[stalled_goal],
        reference_time=now,
    )

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.signal_type == ProactiveSignalType.GOAL_INACTIVITY
    assert cand.priority == ProactivePriority.LOW
    assert 0.0 <= cand.confidence <= 1.0
    assert "goal appears inactive" in cand.reason.lower()
    assert "revisit or work on this goal" in cand.suggested_action.lower()
    assert cand.related_experience_ids == [goal_id]


@pytest.mark.asyncio
async def test_proactive_active_goal_with_meaningful_progress_produces_no_inactivity_signal():
    """Requirement: Active goal with recent progress in observations produces NO inactivity signal."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    goal_exp = Experience(
        user_id=str(user_id),
        content="Build personal AI second brain project.",
        type=ExperienceType.GOAL,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=20),
    )
    progress_exp = Experience(
        user_id=str(user_id),
        content="Implemented tool calling framework for personal AI second brain.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=2),
    )

    candidates = await service.analyze(
        user_id=user_id,
        experiences=[goal_exp, progress_exp],
        reference_time=now,
    )

    assert candidates == []


@pytest.mark.asyncio
async def test_proactive_weak_keyword_overlap_does_not_falsely_count_as_progress_or_inactivity():
    """Requirement: Generic single word overlap (e.g. 'AI' in 'AI documentary') does not count as goal progress,
    and without explicit inactivity evidence, produces NO SIGNAL.
    """
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    goal_exp = Experience(
        user_id=str(user_id),
        content="Build my personal AI second brain codebase.",
        type=ExperienceType.GOAL,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=20),
    )
    # Generic memory sharing the word "AI"
    generic_memory = Experience(
        user_id=str(user_id),
        content="I watched an interesting AI documentary on television.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=1),
    )

    candidates = await service.analyze(
        user_id=user_id,
        experiences=[goal_exp, generic_memory],
        reference_time=now,
    )

    # Without explicit inactivity evidence, conservative policy returns NO SIGNAL
    assert candidates == []


@pytest.mark.asyncio
async def test_proactive_explicit_inactivity_in_current_message_overrides_topical_overlap():
    """Requirement: Explicit inactivity in current message ('haven't worked on for weeks') generates GOAL_INACTIVITY,
    and is NOT suppressed by keyword overlap with the goal.
    """
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    goal_id = uuid.uuid4()
    goal_exp = Experience(
        id=goal_id,
        user_id=str(user_id),
        content="Build my personal AI second brain project.",
        type=ExperienceType.GOAL,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=20),
    )

    # Current message states explicit inactivity on the goal
    candidates = await service.analyze(
        user_id=user_id,
        experiences=[goal_exp],
        current_message="I haven't worked on my personal AI second brain project for weeks.",
        reference_time=now,
    )

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.signal_type == ProactiveSignalType.GOAL_INACTIVITY
    assert cand.priority == ProactivePriority.LOW
    assert "goal appears inactive" in cand.reason.lower()
    assert cand.related_experience_ids == [goal_id]


# ==============================================================================
# 2. Plan vs Commitment Semantics Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_proactive_generic_plan_without_missed_evidence_produces_no_signal():
    """Requirement: Generic plans ('I plan to study AI tonight') are NOT commitments and produce NO signal."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    plan_exp = Experience(
        user_id=str(user_id),
        content="I plan to study AI architectures tonight.",
        type=ExperienceType.DECISION,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(hours=3),
    )

    candidates = await service.analyze(
        user_id=user_id,
        experiences=[plan_exp],
        reference_time=now,
    )

    assert candidates == []


@pytest.mark.asyncio
async def test_proactive_vague_intention_produces_no_signal():
    """Requirement: Vague intentions ('I will learn Rust someday') produce NO commitment signal."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    vague_intention = Experience(
        user_id=str(user_id),
        content="I will learn Rust programming someday.",
        type=ExperienceType.GOAL,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=5),
    )

    candidates = await service.analyze(
        user_id=user_id,
        experiences=[vague_intention],
        reference_time=now,
    )

    assert candidates == []


@pytest.mark.asyncio
async def test_proactive_future_commitment_produces_no_signal():
    """Requirement: Explicit commitment with upcoming/future deadline produces NO missed signal."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    future_commitment = Experience(
        user_id=str(user_id),
        content="I committed to delivering the client roadmap by next month.",
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

    assert candidates == []


@pytest.mark.asyncio
async def test_proactive_commitment_passed_deadline_with_missed_evidence_produces_signal():
    """Requirement: Commitment + passed deadline + missed evidence produces COMMITMENT_MISSED."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    commitment_id = uuid.uuid4()
    missed_commitment = Experience(
        id=commitment_id,
        user_id=str(user_id),
        content="I committed to submit the financial report yesterday, but didn't finish it.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        temporal_context="yesterday",
        created_at=now - timedelta(hours=10),
    )

    candidates = await service.analyze(
        user_id=user_id,
        experiences=[missed_commitment],
        reference_time=now,
    )

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.signal_type == ProactiveSignalType.COMMITMENT_MISSED
    assert cand.priority == ProactivePriority.MEDIUM
    assert 0.0 <= cand.confidence <= 1.0
    assert "planned commitment may have been missed" in cand.reason.lower()
    assert "reschedule, adjust, or check in" in cand.suggested_action.lower()
    assert cand.related_experience_ids == [commitment_id]


@pytest.mark.asyncio
async def test_proactive_commitment_with_insufficient_temporal_evidence_produces_no_signal():
    """Requirement: Commitment without temporal evidence or without evidence of being missed produces NO signal."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    commitment_exp = Experience(
        user_id=str(user_id),
        content="I promised to help my colleague review their code.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=1),
    )

    candidates = await service.analyze(
        user_id=user_id,
        experiences=[commitment_exp],
        reference_time=now,
    )

    assert candidates == []


@pytest.mark.asyncio
async def test_proactive_commitment_missing_past_deadline_even_with_missed_word_produces_no_signal():
    """Requirement: 'I promised to help my colleague review their code, but I didn't' without explicit past deadline
    produces NO SIGNAL.
    """
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    no_deadline_commitment = Experience(
        user_id=str(user_id),
        content="I promised to help my colleague review their code, but I didn't.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=1),
    )

    candidates = await service.analyze(
        user_id=user_id,
        experiences=[no_deadline_commitment],
        reference_time=now,
    )

    # Missing condition 2 (no past deadline / timeframe) -> NO SIGNAL
    assert candidates == []


@pytest.mark.asyncio
async def test_proactive_future_commitment_with_not_started_produces_no_signal():
    """Requirement: 'I committed to submit the report next month, but haven't started' has future deadline -> NO SIGNAL."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    future_commitment = Experience(
        user_id=str(user_id),
        content="I committed to submit the report next month, but haven't started.",
        type=ExperienceType.GOAL,
        source=ExperienceSource.CHAT,
        temporal_context="next month",
        created_at=now - timedelta(days=1),
    )

    candidates = await service.analyze(
        user_id=user_id,
        experiences=[future_commitment],
        reference_time=now,
    )

    assert candidates == []



# ==============================================================================
# 3. Repeated State Tests (Observational & Non-Diagnostic)
# ==============================================================================

@pytest.mark.asyncio
async def test_proactive_single_state_produces_no_repeated_state_signal():
    """Requirement: Single temporary state produces NO signal."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    single_state = Experience(
        user_id=str(user_id),
        content="Feeling tired after a long flight.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        emotional_context=EmotionalContext(emotion="tired", intensity=0.6),
        created_at=now,
    )

    candidates = await service.analyze(
        user_id=user_id,
        experiences=[single_state],
        reference_time=now,
    )

    assert candidates == []


@pytest.mark.asyncio
async def test_proactive_repeated_recent_states_produces_observational_signal():
    """Requirement: Two similar recent states produce REPEATED_STATE with purely observational wording."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    id1 = uuid.uuid4()
    id2 = uuid.uuid4()

    exp1 = Experience(
        id=id1,
        user_id=str(user_id),
        content="Feeling very tired after long meetings.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        emotional_context=EmotionalContext(emotion="tired", intensity=0.7),
        created_at=now - timedelta(days=3),
    )
    exp2 = Experience(
        id=id2,
        user_id=str(user_id),
        content="Exhausted again today, low energy all afternoon.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        emotional_context=EmotionalContext(emotion="tired", intensity=0.8),
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
    assert cand.priority == ProactivePriority.MEDIUM
    assert 0.0 <= cand.confidence <= 1.0
    assert "was recorded more than once recently" in cand.reason.lower() or "appeared repeatedly" in cand.reason.lower()
    assert set(cand.related_experience_ids) == {id1, id2}


@pytest.mark.asyncio
async def test_proactive_different_unrelated_states_produce_no_signal():
    """Requirement: Distinct, non-matching emotional states (e.g. 1 tired, 1 excited) do NOT produce a signal."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    exp_tired = Experience(
        user_id=str(user_id),
        content="Feeling tired after workout.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        emotional_context=EmotionalContext(emotion="tired"),
        created_at=now - timedelta(days=2),
    )
    exp_happy = Experience(
        user_id=str(user_id),
        content="Feeling excited and energetic about the project demo.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        emotional_context=EmotionalContext(emotion="joy"),
        created_at=now - timedelta(days=1),
    )

    candidates = await service.analyze(
        user_id=user_id,
        experiences=[exp_tired, exp_happy],
        reference_time=now,
    )

    assert candidates == []


@pytest.mark.asyncio
async def test_proactive_repeated_state_strictly_avoids_diagnosis_and_personality_inference():
    """Requirement: Emotional state signals must NEVER diagnose medical conditions or label personality traits."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    exp1 = Experience(
        user_id=str(user_id),
        content="Feeling anxious before the board presentation.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        emotional_context=EmotionalContext(emotion="anxious"),
        created_at=now - timedelta(days=3),
    )
    exp2 = Experience(
        user_id=str(user_id),
        content="Nervous and anxious about the client feedback.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        emotional_context=EmotionalContext(emotion="anxious"),
        created_at=now - timedelta(days=1),
    )

    candidates = await service.analyze(
        user_id=user_id,
        experiences=[exp1, exp2],
        reference_time=now,
    )

    assert len(candidates) == 1
    cand = candidates[0]

    # Prohibited medical/diagnostic/personality terms
    forbidden_terms = [
        "depression",
        "chronic anxiety",
        "anxious person",
        "personality trait",
        "disorder",
        "mental illness",
        "clinical",
        "you are an anxious",
        "lazy",
    ]
    for term in forbidden_terms:
        assert term not in cand.reason.lower(), f"Diagnostic term '{term}' found in reason"
        assert term not in cand.suggested_action.lower(), f"Diagnostic term '{term}' found in action"


# ==============================================================================
# 4. Intermediate ProactiveSignal Layer & Architecture Tests
# ==============================================================================

def test_proactive_detect_signals_returns_intermediate_signals_layer():
    """Requirement: detect_signals() produces ProactiveSignal objects before generate_candidates() converts them."""
    service = ProactiveIntelligenceService()
    now = _fixed_now()

    goal_id = uuid.uuid4()
    records = [
        {
            "id": goal_id,
            "content": "Master the piano (on hold, haven't touched in weeks).",
            "type": "GOAL",
            "domain": "hobbies",
            "lifecycle_status": "ACTIVE",
            "temporal_context": "haven't touched",
            "emotion": None,
            "created_at": now - timedelta(days=30),
        }
    ]

    # 1. Test detect_signals returns ProactiveSignal
    signals = service.detect_signals(records=records, now=now)
    assert len(signals) == 1
    signal = signals[0]
    assert isinstance(signal, ProactiveSignal)
    assert signal.type == ProactiveSignalType.GOAL_INACTIVITY
    assert signal.related_experience_ids == [goal_id]
    assert 0.0 <= signal.confidence <= 1.0

    # 2. Test generate_candidates converts ProactiveSignal to ProactiveCandidate
    candidates = service.generate_candidates(signals)
    assert len(candidates) == 1
    cand = candidates[0]
    assert isinstance(cand, ProactiveCandidate)
    assert cand.signal_type == ProactiveSignalType.GOAL_INACTIVITY
    assert cand.priority == ProactivePriority.LOW
    assert cand.suggested_action == "Ask whether the user wants to revisit or work on this goal."
    assert cand.related_experience_ids == [goal_id]


# ==============================================================================
# 5. User Isolation & Safety Boundary Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_proactive_user_isolation_user_a_never_receives_signals_from_user_b():
    """Requirement: Analysis strictly ignores experiences belonging to other users."""
    service = ProactiveIntelligenceService()
    user_a = uuid.uuid4()
    user_b = uuid.uuid4()
    now = _fixed_now()

    # User B has an inactive goal and repeated tiredness
    user_b_goal = Experience(
        user_id=str(user_b),
        content="User B inactive personal goal (on hold, haven't worked on).",
        type=ExperienceType.GOAL,
        source=ExperienceSource.CHAT,
        temporal_context="on hold",
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

    candidates_a = await service.analyze(
        user_id=user_a,
        experiences=[user_b_goal, user_b_tired1, user_b_tired2, user_a_fact],
        reference_time=now,
    )

    assert candidates_a == []


@pytest.mark.asyncio
async def test_proactive_candidates_do_not_execute_tools_and_no_llm_required():
    """Requirement: Service executes purely synchronously/in-memory with NO tool or LLM dependencies."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    goal_id = uuid.uuid4()
    experiences = [
        Experience(
            id=goal_id,
            user_id=str(user_id),
            content="Run a half marathon by summer (paused, no progress).",
            type=ExperienceType.GOAL,
            source=ExperienceSource.CHAT,
            temporal_context="paused",
        )
    ]

    candidates = await service.analyze(
        user_id=user_id,
        experiences=experiences,
        reference_time=now,
    )

    assert len(candidates) == 1
    cand = candidates[0]
    assert isinstance(cand, ProactiveCandidate)
    assert cand.signal_type == ProactiveSignalType.GOAL_INACTIVITY


@pytest.mark.asyncio
async def test_proactive_empty_and_invalid_context_handled_safely():
    """Requirement: Empty context, None user_id, or empty lists return empty list safely."""
    service = ProactiveIntelligenceService()

    assert await service.analyze(user_id=None) == []  # type: ignore[arg-type]
    user_id = uuid.uuid4()
    assert await service.analyze(user_id=user_id, context=None) == []
    assert await service.analyze(user_id=user_id, experiences=[]) == []


@pytest.mark.asyncio
async def test_proactive_duplicate_evidence_does_not_create_duplicate_candidates():
    """Requirement: In-memory per-analysis deduplication prevents identical candidates from duplicate evidence."""
    service = ProactiveIntelligenceService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    goal_id = uuid.uuid4()
    goal_exp = Experience(
        id=goal_id,
        user_id=str(user_id),
        content="Complete the master thesis defense (on hold, no progress).",
        type=ExperienceType.GOAL,
        source=ExperienceSource.CHAT,
        temporal_context="on hold",
    )

    candidates = await service.analyze(
        user_id=user_id,
        experiences=[goal_exp, goal_exp],
        reference_time=now,
    )

    assert len(candidates) == 1
    assert candidates[0].signal_type == ProactiveSignalType.GOAL_INACTIVITY
