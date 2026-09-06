from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import uuid

import pytest

from personal_ai.application.pattern.service import PersonalPatternService
from personal_ai.domain.experience import (
    EmotionalContext,
    Experience,
    ExperienceLifecycleStatus,
    ExperienceSource,
    ExperienceType,
)
from personal_ai.domain.pattern.entity import PersonalPattern
from personal_ai.domain.pattern.enums import PatternDomain, PatternStatus
from personal_ai.domain.pattern.repository import PersonalPatternRepository


def _fixed_now() -> datetime:
    """Return fixed UTC time for deterministic test execution."""
    return datetime(2026, 9, 6, 12, 0, 0, tzinfo=timezone.utc)


class InMemoryPersonalPatternRepository(PersonalPatternRepository):
    """In-memory implementation of PersonalPatternRepository for testing."""

    def __init__(self) -> None:
        self.patterns: Dict[uuid.UUID, PersonalPattern] = {}

    async def create(self, pattern: PersonalPattern) -> PersonalPattern:
        self.patterns[pattern.id] = pattern
        return pattern

    async def save(self, pattern: PersonalPattern) -> PersonalPattern:
        return await self.create(pattern)

    async def update(self, pattern: PersonalPattern) -> PersonalPattern:
        self.patterns[pattern.id] = pattern
        return pattern

    async def get_by_id(
        self,
        pattern_id: uuid.UUID,
        user_id: Optional[uuid.UUID] = None,
    ) -> Optional[PersonalPattern]:
        pattern = self.patterns.get(pattern_id)
        if pattern:
            if user_id is not None and pattern.user_id != user_id:
                return None
            return pattern
        return None

    async def list_by_user(
        self,
        user_id: uuid.UUID,
        status: Optional[PatternStatus] = None,
        domain: Optional[str] = None,
    ) -> List[PersonalPattern]:
        result = [p for p in self.patterns.values() if p.user_id == user_id]
        if status is not None:
            result = [p for p in result if p.status == status]
        if domain is not None:
            dom_str = domain.value if hasattr(domain, "value") else str(domain)
            result = [p for p in result if p.domain == dom_str.upper().strip()]
        return result

    async def get_active_patterns(
        self,
        user_id: uuid.UUID,
        domain: Optional[str] = None,
    ) -> List[PersonalPattern]:
        result = [
            p
            for p in self.patterns.values()
            if p.user_id == user_id
            and p.status in (PatternStatus.HYPOTHESIS, PatternStatus.CONFIRMED)
        ]
        if domain is not None:
            dom_str = domain.value if hasattr(domain, "value") else str(domain)
            result = [p for p in result if p.domain == dom_str.upper().strip()]
        return result

    async def get_active_by_user(
        self,
        user_id: uuid.UUID,
        domain: Optional[str] = None,
    ) -> List[PersonalPattern]:
        return await self.get_active_patterns(user_id=user_id, domain=domain)

    async def get_all_by_user(
        self,
        user_id: uuid.UUID,
        domain: Optional[str] = None,
    ) -> List[PersonalPattern]:
        return await self.list_by_user(user_id=user_id, domain=domain)


# ==============================================================================
# 1. Single experience does not create a pattern
# ==============================================================================

@pytest.mark.asyncio
async def test_single_experience_does_not_create_pattern():
    """Requirement 1: A single observation is insufficient evidence to form any pattern."""
    service = PersonalPatternService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    single_exp = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I procrastinated on my AI project today.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now,
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )

    patterns = await service.detect_patterns(
        user_id=user_id,
        experiences=[single_exp],
        reference_time=now,
    )

    assert patterns == []


# ==============================================================================
# 2. Two distinct relevant experiences can create a hypothesis
# ==============================================================================

@pytest.mark.asyncio
async def test_two_distinct_relevant_experiences_can_create_hypothesis():
    """Requirement 2: 2 distinct relevant experiences create a HYPOTHESIS pattern."""
    service = PersonalPatternService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    exp1 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I procrastinated on my AI project.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=2),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )
    exp2 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I haven't worked on my project for 4 days.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=1),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )

    patterns = await service.detect_patterns(
        user_id=user_id,
        experiences=[exp1, exp2],
        reference_time=now,
    )

    assert len(patterns) == 1
    pattern = patterns[0]
    assert pattern.status == PatternStatus.HYPOTHESIS
    assert pattern.domain == PatternDomain.PROJECTS.value
    assert pattern.user_id == user_id
    assert len(pattern.evidence_ids) == 2
    assert exp1.id in pattern.evidence_ids
    assert exp2.id in pattern.evidence_ids
    assert 0.50 <= pattern.confidence <= 0.60
    assert "appears to" in pattern.description.lower()


# ==============================================================================
# 3. Duplicate evidence does not increase evidence count
# ==============================================================================

@pytest.mark.asyncio
async def test_duplicate_evidence_does_not_increase_evidence_count():
    """Requirement 3: Duplicate experience IDs or identical content do not count as multiple evidence."""
    service = PersonalPatternService()
    user_id = uuid.uuid4()
    now = _fixed_now()
    exp_id = uuid.uuid4()

    # Same ID passed twice
    exp1 = Experience(
        id=exp_id,
        user_id=str(user_id),
        content="I procrastinated on my AI project.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=2),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )
    exp2 = Experience(
        id=exp_id,
        user_id=str(user_id),
        content="I procrastinated on my AI project.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=2),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )

    patterns = await service.detect_patterns(
        user_id=user_id,
        experiences=[exp1, exp2],
        reference_time=now,
    )

    # Since deduplication leaves only 1 distinct experience, no pattern should be formed
    assert patterns == []


# ==============================================================================
# 4. Repeated consistent experiences increase confidence
# ==============================================================================

@pytest.mark.asyncio
async def test_repeated_consistent_experiences_increase_confidence():
    """Requirement 4: Additional distinct experiences progressively increase confidence."""
    service = PersonalPatternService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    def make_exp(text: str, days_ago: int) -> Experience:
        return Experience(
            id=uuid.uuid4(),
            user_id=str(user_id),
            content=text,
            type=ExperienceType.EVENT,
            source=ExperienceSource.CHAT,
            created_at=now - timedelta(days=days_ago),
            lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
        )

    exps = [
        make_exp("I procrastinated on my AI project.", 5),
        make_exp("Haven't worked on my project for days.", 4),
        make_exp("I keep delaying my AI study.", 3),
        make_exp("I delay working on the second brain code.", 2),
        make_exp("Postponed my project tasks again.", 1),
    ]

    # 2 experiences -> ~0.55 confidence, HYPOTHESIS
    p2 = await service.detect_patterns(user_id, exps[:2], reference_time=now)
    assert len(p2) == 1
    assert p2[0].status == PatternStatus.HYPOTHESIS
    assert p2[0].confidence == pytest.approx(0.55, abs=0.01)

    # 3 experiences -> ~0.65 confidence, HYPOTHESIS
    p3 = await service.detect_patterns(user_id, exps[:3], reference_time=now)
    assert len(p3) == 1
    assert p3[0].status == PatternStatus.HYPOTHESIS
    assert p3[0].confidence == pytest.approx(0.65, abs=0.01)

    # 4 experiences -> ~0.72 confidence, HYPOTHESIS
    p4 = await service.detect_patterns(user_id, exps[:4], reference_time=now)
    assert len(p4) == 1
    assert p4[0].status == PatternStatus.HYPOTHESIS
    assert p4[0].confidence == pytest.approx(0.72, abs=0.01)

    # 5 experiences -> >= 0.80 confidence, CONFIRMED
    p5 = await service.detect_patterns(user_id, exps[:5], reference_time=now)
    assert len(p5) == 1
    assert p5[0].status == PatternStatus.CONFIRMED
    assert p5[0].confidence == pytest.approx(0.80, abs=0.01)


# ==============================================================================
# 5. Weak/inconsistent evidence does not create a strong pattern
# ==============================================================================

@pytest.mark.asyncio
async def test_weak_or_inconsistent_evidence_does_not_create_strong_pattern():
    """Requirement 5: Mismatched/unrelated experiences do not create patterns or inflate confidence."""
    service = PersonalPatternService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    exp1 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I bought some groceries at the store today.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=2),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )
    exp2 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="The weather was nice and sunny yesterday.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=1),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )

    patterns = await service.detect_patterns(
        user_id=user_id,
        experiences=[exp1, exp2],
        reference_time=now,
    )

    # Completely disparate facts without recurring behavioral pattern
    assert patterns == []


# ==============================================================================
# 6. Old unrelated experiences do not create a recent pattern
# ==============================================================================

@pytest.mark.asyncio
async def test_old_unrelated_experiences_do_not_create_recent_pattern():
    """Requirement 6: Experiences outside the temporal observation window are ignored."""
    service = PersonalPatternService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    old_exp1 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I procrastinated on my project.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=150),  # 150 days ago
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )
    old_exp2 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I delayed my study session.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=140),  # 140 days ago
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )

    # Max window is 90 days
    patterns = await service.detect_patterns(
        user_id=user_id,
        experiences=[old_exp1, old_exp2],
        reference_time=now,
    )

    assert patterns == []


# ==============================================================================
# 7. Emotional single-event experience does not create a pattern
# ==============================================================================

@pytest.mark.asyncio
async def test_emotional_single_event_experience_does_not_create_pattern():
    """Requirement 7: Single emotional occurrence ('I feel stressed/anxious today') does NOT form a pattern."""
    service = PersonalPatternService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    emotional_exp = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I feel really anxious and stressed about my work today.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        created_at=now,
        emotional_context=EmotionalContext(
            emotion="anxious",
            intensity=0.9,
        ),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )

    patterns = await service.detect_patterns(
        user_id=user_id,
        experiences=[emotional_exp],
        reference_time=now,
    )

    assert patterns == []


# ==============================================================================
# 8. Multiple emotional observations can produce a cautious hypothesis
# ==============================================================================

@pytest.mark.asyncio
async def test_multiple_emotional_observations_produce_cautious_hypothesis():
    """Requirement 8: Multiple recurring emotional observations produce a cautious behavioral hypothesis."""
    service = PersonalPatternService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    exp1 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I am feeling very stressed as the project deadline approaches.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=5),
        emotional_context=EmotionalContext(emotion="stressed", intensity=0.7),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )
    exp2 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="Upcoming deadline is making me feel overwhelmed and stressed.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=2),
        emotional_context=EmotionalContext(emotion="stressed", intensity=0.8),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )

    patterns = await service.detect_patterns(
        user_id=user_id,
        experiences=[exp1, exp2],
        reference_time=now,
    )

    assert len(patterns) >= 1
    stress_pattern = next(
        (p for p in patterns if "stressed" in p.description.lower()),
        None,
    )
    assert stress_pattern is not None
    assert stress_pattern.status == PatternStatus.HYPOTHESIS
    assert "tends to report feeling stressed" in stress_pattern.description.lower()
    assert stress_pattern.confidence <= 0.65


# ==============================================================================
# 9. Pattern evidence IDs are preserved
# ==============================================================================

@pytest.mark.asyncio
async def test_pattern_evidence_ids_are_preserved():
    """Requirement 9: Pattern evidence IDs strictly match the source experiences."""
    service = PersonalPatternService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    id1 = uuid.uuid4()
    id2 = uuid.uuid4()

    exp1 = Experience(
        id=id1,
        user_id=str(user_id),
        content="Whenever I have an upcoming deadline, my work intensity goes up.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=3),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )
    exp2 = Experience(
        id=id2,
        user_id=str(user_id),
        content="Working much harder now to finish before the deadline tomorrow.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=1),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )

    patterns = await service.detect_patterns(
        user_id=user_id,
        experiences=[exp1, exp2],
        reference_time=now,
    )

    assert len(patterns) == 1
    assert set(patterns[0].evidence_ids) == {id1, id2}


# ==============================================================================
# 10. Existing pattern gets updated when new supporting evidence arrives
# ==============================================================================

@pytest.mark.asyncio
async def test_existing_pattern_updated_when_new_evidence_arrives():
    """Requirement 10: Existing pattern evolves when new supporting experience is processed."""
    repo = InMemoryPersonalPatternRepository()
    service = PersonalPatternService(pattern_repo=repo)
    user_id = uuid.uuid4()
    now = _fixed_now()

    exp1 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I procrastinated on my AI project.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=4),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )
    exp2 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="Haven't worked on the project for days.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=3),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )

    # Initial detection & persistence
    initial_patterns = await service.detect_and_persist(
        user_id=user_id,
        experiences=[exp1, exp2],
        reference_time=now,
    )
    assert len(initial_patterns) == 1
    p_id = initial_patterns[0].id
    assert initial_patterns[0].confidence == pytest.approx(0.55, abs=0.01)

    # Later new experience arrives
    exp3 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="Delay working on the second brain code again.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now,
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )

    updated_patterns = await service.detect_and_persist(
        user_id=user_id,
        experiences=[exp1, exp2, exp3],
        reference_time=now,
    )

    assert len(updated_patterns) == 1
    updated = updated_patterns[0]
    assert updated.id == p_id  # Same pattern entity updated
    assert len(updated.evidence_ids) == 3
    assert exp3.id in updated.evidence_ids
    assert updated.confidence == pytest.approx(0.65, abs=0.01)
    assert updated.last_observed_at == now


# ==============================================================================
# 11. Historical pattern is not deleted
# ==============================================================================

@pytest.mark.asyncio
async def test_historical_pattern_is_not_deleted():
    """Requirement 11: Patterns are never deleted from history, even when superseded."""
    repo = InMemoryPersonalPatternRepository()
    service = PersonalPatternService(pattern_repo=repo)
    user_id = uuid.uuid4()
    now = _fixed_now()

    pattern = PersonalPattern(
        user_id=user_id,
        description="Initial project consistency pattern.",
        domain=PatternDomain.PROJECTS.value,
        evidence_ids=[uuid.uuid4(), uuid.uuid4()],
        confidence=0.55,
        status=PatternStatus.HYPOTHESIS,
        first_observed_at=now - timedelta(days=10),
        last_observed_at=now - timedelta(days=5),
    )
    await repo.save(pattern)

    # Supersede with a newer pattern
    new_pattern_id = uuid.uuid4()
    await service.supersede_pattern(
        pattern_id=pattern.id,
        user_id=user_id,
        superseded_by_id=new_pattern_id,
    )

    # Verify old pattern still exists in repository with SUPERSEDED status
    old_pattern = await repo.get_by_id(pattern.id, user_id=user_id)
    assert old_pattern is not None
    assert old_pattern.status == PatternStatus.SUPERSEDED
    assert old_pattern.superseded_by_id == new_pattern_id

    # Active patterns query filters it out, but list_by_user returns it
    active_patterns = await repo.get_active_patterns(user_id=user_id)
    all_patterns = await repo.list_by_user(user_id=user_id)
    assert old_pattern not in active_patterns
    assert old_pattern in all_patterns


# ==============================================================================
# 12. Weakened pattern can reduce confidence
# ==============================================================================

@pytest.mark.asyncio
async def test_weakened_pattern_reduces_confidence():
    """Requirement 12: Calling weaken() reduces confidence and transitions to WEAKENED if appropriate."""
    user_id = uuid.uuid4()
    now = _fixed_now()

    pattern = PersonalPattern(
        user_id=user_id,
        description="Project activity appears to increase near deadlines.",
        domain=PatternDomain.PROJECTS.value,
        evidence_ids=[uuid.uuid4(), uuid.uuid4()],
        confidence=0.55,
        status=PatternStatus.HYPOTHESIS,
        first_observed_at=now - timedelta(days=10),
        last_observed_at=now - timedelta(days=2),
    )

    # Weaken once: 0.55 - 0.15 = 0.40
    pattern.weaken(confidence_reduction=0.15)
    assert pattern.confidence == pytest.approx(0.40, abs=0.01)
    assert pattern.status == PatternStatus.HYPOTHESIS

    # Weaken second time: 0.40 - 0.15 = 0.25 (drops below 0.40 threshold -> WEAKENED)
    pattern.weaken(confidence_reduction=0.15)
    assert pattern.confidence == pytest.approx(0.25, abs=0.01)
    assert pattern.status == PatternStatus.WEAKENED


# ==============================================================================
# 13. User isolation: User A cannot use User B's experiences as evidence
# ==============================================================================

@pytest.mark.asyncio
async def test_cross_user_isolation():
    """Requirement 13: Strict isolation by authenticated user_id. Cross-user experiences are rejected."""
    service = PersonalPatternService()
    user_a = uuid.uuid4()
    user_b = uuid.uuid4()
    now = _fixed_now()

    # User B's experience
    user_b_exp = Experience(
        id=uuid.uuid4(),
        user_id=str(user_b),
        content="I procrastinated on my AI project.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=2),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )
    # User A's experience
    user_a_exp = Experience(
        id=uuid.uuid4(),
        user_id=str(user_a),
        content="I haven't worked on my project for days.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=1),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )

    # When running analysis for User A, User B's experience must be filtered out
    patterns = await service.detect_patterns(
        user_id=user_a,
        experiences=[user_a_exp, user_b_exp],
        reference_time=now,
    )

    # Only 1 experience belongs to User A; since >= 2 required, NO pattern is created
    assert patterns == []


# ==============================================================================
# 14. Empty experience set produces no patterns
# ==============================================================================

@pytest.mark.asyncio
async def test_empty_experience_set_produces_no_patterns():
    """Requirement 14: Empty experience list produces empty pattern list."""
    service = PersonalPatternService()
    user_id = uuid.uuid4()

    patterns = await service.detect_patterns(
        user_id=user_id,
        experiences=[],
        reference_time=_fixed_now(),
    )

    assert patterns == []


# ==============================================================================
# 15. Invalid confidence is rejected
# ==============================================================================

def test_invalid_confidence_is_rejected():
    """Requirement 15: Out-of-bounds confidence (<0.0 or >1.0) raises ValueError."""
    user_id = uuid.uuid4()

    # Negative confidence
    with pytest.raises(ValueError, match="Pattern confidence must be between 0.0 and 1.0"):
        PersonalPattern(
            user_id=user_id,
            description="Test pattern",
            domain=PatternDomain.GENERAL.value,
            confidence=-0.1,
        )

    # Confidence > 1.0
    with pytest.raises(ValueError, match="Pattern confidence must be between 0.0 and 1.0"):
        PersonalPattern(
            user_id=user_id,
            description="Test pattern",
            domain=PatternDomain.GENERAL.value,
            confidence=1.05,
        )


# ==============================================================================
# 16. Pattern descriptions do not contain diagnostic/personality claims
# ==============================================================================

@pytest.mark.asyncio
async def test_pattern_descriptions_contain_no_diagnostic_or_personality_claims():
    """Requirement 16: Patterns describe observable behavior with uncertainty language, avoiding diagnoses."""
    service = PersonalPatternService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    exps = [
        Experience(
            id=uuid.uuid4(),
            user_id=str(user_id),
            content="I procrastinated on my AI project.",
            type=ExperienceType.EVENT,
            source=ExperienceSource.CHAT,
            created_at=now - timedelta(days=3),
            lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
        ),
        Experience(
            id=uuid.uuid4(),
            user_id=str(user_id),
            content="Haven't worked on my project for 4 days.",
            type=ExperienceType.EVENT,
            source=ExperienceSource.CHAT,
            created_at=now - timedelta(days=1),
            lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
        ),
    ]

    patterns = await service.detect_patterns(user_id=user_id, experiences=exps, reference_time=now)
    assert len(patterns) >= 1

    forbidden_terms = [
        "lazy", "adhd", "depression", "disorder", "personality", "unstable",
        "incompetent", "pathological", "illness", "defect", "lacks discipline"
    ]

    for p in patterns:
        desc_lower = p.description.lower()
        for forbidden in forbidden_terms:
            assert forbidden not in desc_lower, f"Pattern description contained forbidden term: '{forbidden}'"
        # Must contain cautious uncertainty phrasing
        assert any(term in desc_lower for term in ["appears to", "tends to", "repeatedly", "may", "observed"])


# ==============================================================================
# 17. No single event creates personality traits
# ==============================================================================

@pytest.mark.asyncio
async def test_no_single_event_creates_personality_traits():
    """Requirement 17: Explicit checks that single isolated events never produce traits."""
    service = PersonalPatternService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    single_events = [
        "I didn't work today.",
        "I isolated myself once.",
        "I felt anxious today.",
        "I ate junk food today.",
    ]

    for event_text in single_events:
        exp = Experience(
            id=uuid.uuid4(),
            user_id=str(user_id),
            content=event_text,
            type=ExperienceType.EVENT,
            source=ExperienceSource.CHAT,
            created_at=now,
            lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
        )
        patterns = await service.detect_patterns(user_id, [exp], reference_time=now)
        assert patterns == [], f"Single event '{event_text}' unexpectedly produced a pattern!"


# ==============================================================================
# 18. No LLM/tool/autonomous action is required for pattern detection
# ==============================================================================

@pytest.mark.asyncio
async def test_deterministic_execution_no_llm_or_tools_needed():
    """Requirement 18: Detection executes deterministically without any external network or LLM dependency."""
    service = PersonalPatternService(pattern_repo=None)
    user_id = uuid.uuid4()
    now = _fixed_now()

    exp1 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="Maintaining my daily 5k running routine every morning.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=3),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )
    exp2 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="Completed my regular gym workout routine on schedule today.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=1),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )

    patterns = await service.detect_patterns(user_id, [exp1, exp2], reference_time=now)
    assert len(patterns) == 1
    assert patterns[0].domain == PatternDomain.FITNESS.value
    assert patterns[0].status == PatternStatus.HYPOTHESIS


# ==============================================================================
# 19. Serialization and Repository CRUD operations
# ==============================================================================

@pytest.mark.asyncio
async def test_entity_serialization_and_repository_crud():
    """Requirement 19: Test PersonalPattern serialization/deserialization and full repository lifecycle."""
    repo = InMemoryPersonalPatternRepository()
    user_id = uuid.uuid4()
    now = _fixed_now()

    pattern = PersonalPattern(
        user_id=user_id,
        description="Learning activity appears to occur consistently during weekends.",
        domain=PatternDomain.LEARNING.value,
        evidence_ids=[uuid.uuid4(), uuid.uuid4()],
        confidence=0.65,
        status=PatternStatus.HYPOTHESIS,
        first_observed_at=now - timedelta(days=7),
        last_observed_at=now,
    )

    # Test serialization round-trip
    p_dict = pattern.to_dict()
    restored = PersonalPattern.from_dict(p_dict)
    assert restored.id == pattern.id
    assert restored.user_id == pattern.user_id
    assert restored.description == pattern.description
    assert restored.domain == pattern.domain
    assert restored.confidence == pattern.confidence
    assert restored.status == pattern.status
    assert restored.evidence_ids == pattern.evidence_ids

    # Save to repository
    await repo.save(pattern)

    # Query active
    active = await repo.get_active_patterns(user_id=user_id, domain=PatternDomain.LEARNING.value)
    assert len(active) == 1
    assert active[0].id == pattern.id

    # Add evidence
    new_evidence_id = uuid.uuid4()
    pattern.add_evidence(new_evidence_id, observed_at=now)
    await repo.update(pattern)

    fetched = await repo.get_by_id(pattern.id, user_id=user_id)
    assert fetched is not None
    assert new_evidence_id in fetched.evidence_ids
    assert len(fetched.evidence_ids) == 3


# ==============================================================================
# 20. Idempotency and Non-Inflation Tests for detect_and_persist()
# ==============================================================================

@pytest.mark.asyncio
async def test_detect_and_persist_is_idempotent_when_called_twice():
    """Requirement: Reprocessing the exact same experience set does not change evidence_ids or confidence."""
    repo = InMemoryPersonalPatternRepository()
    service = PersonalPatternService(pattern_repo=repo)
    user_id = uuid.uuid4()
    now = _fixed_now()

    exp1 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I procrastinated on my AI project.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=2),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )
    exp2 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="Haven't worked on my project for 4 days.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=1),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )

    # First run
    first_run = await service.detect_and_persist(
        user_id=user_id,
        experiences=[exp1, exp2],
        reference_time=now,
    )
    assert len(first_run) == 1
    p1 = first_run[0]
    p1_id = p1.id
    p1_evidence = list(p1.evidence_ids)
    p1_conf = p1.confidence

    # Second run with identical experiences
    second_run = await service.detect_and_persist(
        user_id=user_id,
        experiences=[exp1, exp2],
        reference_time=now,
    )
    assert len(second_run) == 1
    p2 = second_run[0]

    # Must be completely identical and idempotent
    assert p2.id == p1_id
    assert p2.evidence_ids == p1_evidence
    assert p2.confidence == p1_conf
    assert len(await repo.get_active_patterns(user_id)) == 1


@pytest.mark.asyncio
async def test_detect_and_persist_new_evidence_added_exactly_once():
    """Requirement: New evidence is added exactly once and confidence does not inflate on re-runs."""
    repo = InMemoryPersonalPatternRepository()
    service = PersonalPatternService(pattern_repo=repo)
    user_id = uuid.uuid4()
    now = _fixed_now()

    exp1 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I procrastinated on my AI project.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=3),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )
    exp2 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="Haven't worked on my project for 4 days.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=2),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )
    exp3 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="Delayed my AI study again today.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=1),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )

    # Initial 2 experiences -> 2 evidence, 0.55 confidence
    await service.detect_and_persist(user_id, [exp1, exp2], reference_time=now)

    # Add 3rd experience -> 3 evidence, 0.65 confidence
    updated = await service.detect_and_persist(user_id, [exp1, exp2, exp3], reference_time=now)
    assert len(updated[0].evidence_ids) == 3
    assert updated[0].confidence == pytest.approx(0.65, abs=0.01)

    # Repeatedly process 5 more times with the exact same 3 experiences
    for _ in range(5):
        re_run = await service.detect_and_persist(user_id, [exp1, exp2, exp3], reference_time=now)
        assert len(re_run[0].evidence_ids) == 3
        assert re_run[0].confidence == pytest.approx(0.65, abs=0.01)


# ==============================================================================
# 21. Evidence Validation in evaluate_evidence()
# ==============================================================================

def test_evaluate_evidence_rejects_unrelated_arbitrary_experience():
    """Requirement: Unrelated experience cannot be blindly attached as supporting evidence."""
    service = PersonalPatternService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    pattern = PersonalPattern(
        user_id=user_id,
        description="Project activity appears to become inconsistent after periods of initial activity.",
        domain=PatternDomain.PROJECTS.value,
        evidence_ids=[uuid.uuid4(), uuid.uuid4()],
        confidence=0.55,
        status=PatternStatus.HYPOTHESIS,
        first_observed_at=now - timedelta(days=5),
        last_observed_at=now - timedelta(days=2),
    )
    initial_evidence = list(pattern.evidence_ids)
    initial_conf = pattern.confidence

    # Completely unrelated experience (groceries)
    unrelated_exp = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I went to the supermarket and bought milk and eggs today.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now,
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )

    result = service.evaluate_evidence(
        pattern=pattern,
        new_experience=unrelated_exp,
        is_supporting=True,
    )

    # Must NOT attach unrelated experience
    assert result.evidence_ids == initial_evidence
    assert result.confidence == initial_conf
    assert unrelated_exp.id not in result.evidence_ids


def test_evaluate_evidence_accepts_matching_supporting_experience():
    """Requirement: Matching experience is attached and evolves pattern confidence."""
    service = PersonalPatternService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    pattern = PersonalPattern(
        user_id=user_id,
        description="Project activity appears to become inconsistent after periods of initial activity.",
        domain=PatternDomain.PROJECTS.value,
        evidence_ids=[uuid.uuid4(), uuid.uuid4()],
        confidence=0.55,
        status=PatternStatus.HYPOTHESIS,
        first_observed_at=now - timedelta(days=5),
        last_observed_at=now - timedelta(days=2),
    )

    matching_exp = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="Struggling to continue and delayed working on my second brain project.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now,
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )

    result = service.evaluate_evidence(
        pattern=pattern,
        new_experience=matching_exp,
        is_supporting=True,
    )

    assert matching_exp.id in result.evidence_ids
    assert len(result.evidence_ids) == 3
    assert result.confidence == pytest.approx(0.65, abs=0.01)


# ==============================================================================
# 22. Fail-Closed Deserialization Tests
# ==============================================================================

def test_repository_model_to_domain_fails_closed_on_corrupted_status():
    """Requirement: Corrupted or unknown PatternStatus in DB model raises ValueError (fails closed)."""
    from personal_ai.db.models import PersonalPatternModel
    from personal_ai.db.repositories.sqlalchemy_personal_pattern_repository import (
        SQLAlchemyPersonalPatternRepository,
    )

    corrupted_model = PersonalPatternModel(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        description="Some pattern description",
        domain="PROJECTS",
        evidence_ids=[str(uuid.uuid4())],
        confidence=0.6,
        status="CORRUPTED_OR_UNKNOWN_STATUS",
        first_observed_at=_fixed_now(),
        last_observed_at=_fixed_now(),
        created_at=_fixed_now(),
        updated_at=_fixed_now(),
    )

    with pytest.raises(ValueError, match="Invalid or corrupted PatternStatus"):
        SQLAlchemyPersonalPatternRepository._model_to_domain(corrupted_model)


def test_domain_entity_from_dict_fails_closed_on_corrupted_status():
    """Requirement: PersonalPattern.from_dict raises ValueError on invalid status."""
    corrupted_dict = {
        "id": str(uuid.uuid4()),
        "user_id": str(uuid.uuid4()),
        "description": "Some pattern description",
        "domain": "PROJECTS",
        "evidence_ids": [str(uuid.uuid4())],
        "confidence": 0.6,
        "status": "INVALID_STATUS_STRING",
        "first_observed_at": _fixed_now().isoformat(),
        "last_observed_at": _fixed_now().isoformat(),
    }

    with pytest.raises(ValueError):
        PersonalPattern.from_dict(corrupted_dict)


# ==============================================================================
# 23. Intra-Experience Component Relational Validation (Negative & Positive Tests)
# ==============================================================================

@pytest.mark.asyncio
async def test_negative_poor_sleep_alone_and_low_energy_alone_produces_no_pattern():
    """Negative Requirement 1: Poor sleep alone + low energy separately -> NO pattern."""
    service = PersonalPatternService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    # Exp 1 mentions sleep, but NOT low energy
    exp1 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I had poor sleep and insomnia last night.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=2),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )
    # Exp 2 mentions low energy, but NOT sleep
    exp2 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I have low energy and feel tired today.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=1),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )

    patterns = await service.detect_patterns(user_id, [exp1, exp2], reference_time=now)
    # Must NOT form relational sleep_energy_correlation pattern
    assert patterns == []


@pytest.mark.asyncio
async def test_negative_deadline_alone_and_stress_alone_produces_no_pattern():
    """Negative Requirement 2: Deadline alone + stress separately -> NO pattern."""
    service = PersonalPatternService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    # Exp 1 mentions deadline, but NO stress
    exp1 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="The project deadline and presentation is scheduled for next Friday.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=2),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )
    # Exp 2 mentions stress, but NO deadline/milestone
    exp2 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I am feeling anxious, nervous, and stressed today.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=1),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )

    patterns = await service.detect_patterns(user_id, [exp1, exp2], reference_time=now)
    # Must NOT form relational milestone_stress_response pattern
    assert patterns == []


@pytest.mark.asyncio
async def test_negative_deadline_alone_and_working_harder_alone_produces_no_pattern():
    """Negative Requirement 3: Deadline alone + working harder separately -> NO pattern."""
    service = PersonalPatternService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    # Exp 1 mentions deadline, but NO intensity surge
    exp1 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="The upcoming deadline for the assignment is next week.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=2),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )
    # Exp 2 mentions working harder, but NO deadline
    exp2 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I was hyperfocused and working harder today on general tasks.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=1),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )

    patterns = await service.detect_patterns(user_id, [exp1, exp2], reference_time=now)
    # Must NOT form relational deadline_intensity_surge pattern
    assert patterns == []


@pytest.mark.asyncio
async def test_negative_two_unrelated_gym_exercise_events_produces_no_fitness_consistency():
    """Negative Requirement 4: Two isolated gym/exercise events -> NO fitness consistency pattern without routine evidence."""
    service = PersonalPatternService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    exp1 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I went for a 5k run today.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=3),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )
    exp2 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I went to the gym for a workout session.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=1),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )

    patterns = await service.detect_patterns(user_id, [exp1, exp2], reference_time=now)
    assert patterns == []


@pytest.mark.asyncio
async def test_positive_same_experience_connects_poor_sleep_and_low_energy():
    """Positive Requirement 1: Same experience connects poor sleep with low energy across 2+ experiences -> pattern."""
    service = PersonalPatternService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    exp1 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="Poor sleep last night left me feeling very low energy and tired today.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=3),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )
    exp2 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="Slept poorly and woke up completely exhausted and groggy this morning.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=1),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )

    patterns = await service.detect_patterns(user_id, [exp1, exp2], reference_time=now)
    assert len(patterns) == 1
    assert patterns[0].domain == PatternDomain.HEALTH.value
    assert patterns[0].status == PatternStatus.HYPOTHESIS
    assert "low energy after poor sleep" in patterns[0].description.lower()


@pytest.mark.asyncio
async def test_positive_same_experience_connects_deadline_and_stress():
    """Positive Requirement 2: Same experience connects deadline with stress across 2+ experiences -> pattern."""
    service = PersonalPatternService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    exp1 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I am feeling very stressed as the project deadline approaches.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=4),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )
    exp2 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="Upcoming deadline presentation is making me feel overwhelmed and anxious.",
        type=ExperienceType.STATE,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=2),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )

    patterns = await service.detect_patterns(user_id, [exp1, exp2], reference_time=now)
    assert len(patterns) >= 1
    career_pat = next((p for p in patterns if p.domain == PatternDomain.CAREER.value), None)
    assert career_pat is not None
    assert "feeling stressed" in career_pat.description.lower()


@pytest.mark.asyncio
async def test_positive_same_experience_connects_deadline_and_increased_work_intensity():
    """Positive Requirement 3: Same experience connects deadline with work intensity across 2+ experiences -> pattern."""
    service = PersonalPatternService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    exp1 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="Whenever I have an upcoming deadline, my work intensity goes up.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=3),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )
    exp2 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="Working much harder now to finish before the deadline tomorrow.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=1),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )

    patterns = await service.detect_patterns(user_id, [exp1, exp2], reference_time=now)
    assert len(patterns) == 1
    assert patterns[0].domain == PatternDomain.PROJECTS.value
    assert "work intensity appears to increase near deadlines" in patterns[0].description.lower()


@pytest.mark.asyncio
async def test_positive_explicit_recurring_fitness_routine():
    """Positive Requirement 4: Explicit recurring fitness routine across 2+ experiences -> pattern."""
    service = PersonalPatternService()
    user_id = uuid.uuid4()
    now = _fixed_now()

    exp1 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="I have maintained my daily 5k running routine every morning.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=3),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )
    exp2 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="Going to the gym consistently three times a week on schedule.",
        type=ExperienceType.EVENT,
        source=ExperienceSource.CHAT,
        created_at=now - timedelta(days=1),
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
    )

    patterns = await service.detect_patterns(user_id, [exp1, exp2], reference_time=now)
    assert len(patterns) == 1
    assert patterns[0].domain == PatternDomain.FITNESS.value
    assert "consistent routine" in patterns[0].description.lower()
