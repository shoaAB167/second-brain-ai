from datetime import datetime, timedelta, timezone
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

from personal_ai.core.logger import get_logger
from personal_ai.domain.experience import (
    Experience,
    ExperienceLifecycleStatus,
    ExperienceType,
)
from personal_ai.domain.pattern.entity import PersonalPattern, utc_now
from personal_ai.domain.pattern.enums import PatternDomain, PatternStatus
from personal_ai.domain.pattern.repository import PersonalPatternRepository

logger = get_logger(__name__)


class PersonalPatternService:
    """Deterministic, conservative service for detecting and evolving personal behavioral patterns.

    Architecture:
        Observation -> Repeated Evidence -> Pattern Hypothesis -> Confidence -> Personal Model

    Safety Invariants:
        1. Minimum Evidence Rule: A single experience NEVER creates a pattern (requires >= 2 distinct pieces of evidence).
        2. Conservative Preference: NO PATTERN > FALSE POSITIVE when evidence is weak, single, or ambiguous.
        3. Observable Behavior Only: Descriptions use uncertainty language ('appears to', 'tends to').
        4. Zero Clinical/Medical Diagnosis: Never infers depression, ADHD, anxiety disorders, etc.
        5. Zero Personality Trait Labeling: Never brands user as lazy, undisciplined, or disorganized.
        6. Strict User Isolation: All operations are strictly bounded to the authenticated user_id.
        7. History Preservation: Historical patterns are never deleted (status transitions to WEAKENED or SUPERSEDED).
    """

    # Conservative thematic keyword clusters for observable behavioral patterns
    _PATTERN_THEMES = [
        {
            "id": "project_inconsistency",
            "domain": PatternDomain.PROJECTS.value,
            "description": "Project activity appears to become inconsistent after periods of initial activity.",
            "keywords": [
                r"\b(procrastinated|procrastinating|procrastination)\b",
                r"\b(haven'?t worked on|delayed|delaying|stalled|put off|postponed)\b",
                r"\b(struggling to continue|lost momentum|skipping study|skipping project|delaying my ai study)\b",
                r"\b(hard to stay consistent|inconsistent with project|delay working on)\b",
            ],
            "relevant_types": {"GOAL", "PROJECT", "EVENT", "HABIT", "STATE", "DECISION", "FACT"},
        },
        {
            "id": "deadline_intensity_surge",
            "domain": PatternDomain.PROJECTS.value,
            "description": "Work intensity appears to increase near deadlines.",
            "keywords": [
                r"\b(near deadline|close to deadline|deadline approaching|before the deadline|upcoming deadline|deadline tomorrow)\b",
                r"\b(working harder|work intensity|cramming|hyperfocused before|rush to finish|finish before the deadline)\b",
                r"\b(productive under pressure|last minute burst)\b",
            ],
            "relevant_types": {"GOAL", "PROJECT", "EVENT", "HABIT", "DECISION", "STATE", "FACT"},
        },
        {
            "id": "sleep_energy_correlation",
            "domain": PatternDomain.HEALTH.value,
            "description": "Repeatedly reports feeling low energy after poor sleep or late nights.",
            "keywords": [
                r"\b(poor sleep|bad sleep|late night|slept late|lack of sleep|insomnia)\b",
                r"\b(tired today|low energy|exhausted morning|groggy|drained afternoon)\b",
            ],
            "relevant_types": {"STATE", "EMOTION_STATE", "EVENT", "HABIT", "FACT"},
        },
        {
            "id": "milestone_stress_response",
            "domain": PatternDomain.CAREER.value,
            "description": "Tends to report feeling stressed or anxious around major project milestones or deadlines.",
            "keywords": [
                r"\b(presentation|demo|board meeting|interview|milestone|review|project deadline|upcoming deadline)\b",
                r"\b(nervous|anxious|stressed|anxiety|under pressure|overwhelmed)\b",
            ],
            "relevant_types": {"STATE", "EMOTION_STATE", "EVENT", "FACT"},
        },
        {
            "id": "morning_focus_preference",
            "domain": PatternDomain.LEARNING.value,
            "description": "Productivity tends to be higher when focused work is scheduled in the morning.",
            "keywords": [
                r"\b(morning focus|productive morning|early morning routine|morning study)\b",
                r"\b(more focused early|accomplished a lot before noon)\b",
            ],
            "relevant_types": {"HABIT", "PROJECT", "EVENT", "STATE", "FACT"},
        },
        {
            "id": "fitness_consistency",
            "domain": PatternDomain.FITNESS.value,
            "description": "Fitness and exercise activities appear to follow a consistent routine.",
            "keywords": [
                r"\b(run|running|5k run|workout|gym|exercise|fitness|lifting|cardio)\b",
            ],
            "relevant_types": {"HABIT", "EVENT", "GOAL", "STATE", "FACT"},
        },
    ]

    def __init__(
        self,
        pattern_repository: Optional[PersonalPatternRepository] = None,
        pattern_repo: Optional[PersonalPatternRepository] = None,
    ) -> None:
        """Initialize PersonalPatternService with optional repository dependency."""
        self._pattern_repo = pattern_repository or pattern_repo

    def _normalize_user_id(self, user_id: Any) -> Optional[uuid.UUID]:
        """Convert any user_id representation to UUID safely."""
        if isinstance(user_id, uuid.UUID):
            return user_id
        if isinstance(user_id, str):
            try:
                return uuid.UUID(user_id)
            except ValueError:
                return None
        return None

    def calculate_confidence(self, evidence_count: int) -> float:
        """Deterministic confidence calculation strategy based on evidence volume and consistency.

        Confidence scale:
            - 0-1 evidence: 0.00 (Insufficient evidence)
            - 2 evidence: ~0.55 (Initial hypothesis)
            - 3 evidence: ~0.65 (Emerging pattern)
            - 4 evidence: ~0.72 (Consistent pattern)
            - 5+ evidence: ~0.80 - 0.85 (Strong pattern, capped at 0.86; never 0.99)
        """
        if evidence_count < 2:
            return 0.0

        if evidence_count == 2:
            return 0.55
        elif evidence_count == 3:
            return 0.65
        elif evidence_count == 4:
            return 0.72
        elif evidence_count == 5:
            return 0.80
        else:
            return round(min(0.80 + (evidence_count - 5) * 0.02, 0.86), 4)

    def _filter_user_experiences(
        self,
        user_id: uuid.UUID,
        experiences: List[Experience],
        reference_time: Optional[datetime] = None,
        max_lookback_days: int = 90,
    ) -> List[Experience]:
        """Filter experiences to guarantee strict user isolation, deduplicate, and bound temporally."""
        isolated: List[Experience] = []
        seen_ids: Set[uuid.UUID] = set()
        seen_contents: Set[str] = set()

        ref = reference_time or utc_now()
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
        cutoff_date = ref - timedelta(days=max_lookback_days)

        for exp in experiences:
            # Enforce user isolation: User A cannot use User B's experiences
            exp_user = self._normalize_user_id(exp.user_id)
            if exp_user is not None and exp_user != user_id:
                logger.warning(
                    "User isolation violation prevented: experience user_id mismatch [expected=%s, got=%s]",
                    user_id,
                    exp.user_id,
                )
                continue

            # Deduplicate by experience ID
            if exp.id in seen_ids:
                continue

            # Deduplicate by exact content hash
            clean_content = exp.content.strip().lower()
            if clean_content in seen_contents:
                continue

            # Skip expired or superseded experiences
            if exp.lifecycle_status in (ExperienceLifecycleStatus.EXPIRED, ExperienceLifecycleStatus.SUPERSEDED):
                continue

            # Temporal bounding: skip experiences older than lookback window
            if isinstance(exp.created_at, datetime):
                exp_dt = exp.created_at.replace(tzinfo=timezone.utc) if exp.created_at.tzinfo is None else exp.created_at
                if exp_dt < cutoff_date:
                    continue

            seen_ids.add(exp.id)
            seen_contents.add(clean_content)
            isolated.append(exp)

        return isolated

    def find_candidate_patterns(
        self,
        user_id: uuid.UUID,
        experiences: List[Experience],
        reference_time: Optional[datetime] = None,
    ) -> List[PersonalPattern]:
        """Detect candidate personal pattern hypotheses from an authenticated user's experiences.

        Args:
            user_id: Authenticated user UUID.
            experiences: List of raw or structured Experience entities.
            reference_time: Optional reference time for temporal boundedness.

        Returns:
            List[PersonalPattern]: Detected pattern hypotheses (empty if insufficient evidence).
        """
        if not user_id:
            logger.warning("find_candidate_patterns called without user_id.")
            return []

        if not experiences or len(experiences) < 2:
            # Minimum evidence invariant: < 2 experiences can NEVER create a pattern
            return []

        ref_time = reference_time or utc_now()
        if ref_time.tzinfo is None:
            ref_time = ref_time.replace(tzinfo=timezone.utc)

        # 1. Enforce user isolation, deduplication, and temporal bounding
        valid_experiences = self._filter_user_experiences(
            user_id=user_id,
            experiences=experiences,
            reference_time=ref_time,
        )
        if len(valid_experiences) < 2:
            return []

        candidate_patterns: List[PersonalPattern] = []

        # 2. Evaluate each behavioral theme against user experiences
        for theme in self._PATTERN_THEMES:
            matched_experiences: List[Experience] = []

            for exp in valid_experiences:
                exp_content = exp.content.lower()
                exp_temporal = (exp.temporal_context or "").lower()
                exp_emotion = (exp.emotional_context.emotion if exp.emotional_context else "") or ""

                combined_text = f"{exp_content} {exp_temporal} {exp_emotion}".lower()

                # Check if this experience matches theme patterns
                is_match = False
                for pattern_regex in theme["keywords"]:
                    if re.search(pattern_regex, combined_text):
                        is_match = True
                        break

                if is_match:
                    matched_experiences.append(exp)

            # Minimum evidence rule: require >= 2 distinct matched experiences
            if len(matched_experiences) >= 2:
                evidence_ids = [exp.id for exp in matched_experiences]
                evidence_count = len(evidence_ids)
                confidence = self.calculate_confidence(evidence_count)

                # Determine first and last observed timestamps
                timestamps = [
                    (exp.created_at.replace(tzinfo=timezone.utc) if exp.created_at.tzinfo is None else exp.created_at)
                    for exp in matched_experiences
                    if isinstance(exp.created_at, datetime)
                ]
                first_obs = min(timestamps) if timestamps else ref_time
                last_obs = max(timestamps) if timestamps else ref_time

                status = PatternStatus.CONFIRMED if (evidence_count >= 5 and confidence >= 0.80) else PatternStatus.HYPOTHESIS

                pattern = PersonalPattern(
                    user_id=user_id,
                    description=theme["description"],
                    domain=theme["domain"],
                    evidence_ids=evidence_ids,
                    confidence=confidence,
                    status=status,
                    first_observed_at=first_obs,
                    last_observed_at=last_obs,
                    created_at=ref_time,
                    updated_at=ref_time,
                )
                candidate_patterns.append(pattern)

        return candidate_patterns

    async def detect_patterns(
        self,
        user_id: uuid.UUID,
        experiences: List[Experience],
        reference_time: Optional[datetime] = None,
    ) -> List[PersonalPattern]:
        """Alias for find_candidate_patterns for convenience."""
        return self.find_candidate_patterns(
            user_id=user_id,
            experiences=experiences,
            reference_time=reference_time,
        )

    async def detect_and_persist(
        self,
        user_id: uuid.UUID,
        experiences: List[Experience],
        reference_time: Optional[datetime] = None,
    ) -> List[PersonalPattern]:
        """Detect patterns and evolve or persist them into the configured repository.

        Args:
            user_id: Authenticated user UUID.
            experiences: List of Experience entities.
            reference_time: Reference time for evaluation.

        Returns:
            List[PersonalPattern]: Persisted / updated PersonalPattern entities.
        """
        candidates = self.find_candidate_patterns(
            user_id=user_id,
            experiences=experiences,
            reference_time=reference_time,
        )

        if not self._pattern_repo:
            return candidates

        existing_patterns = await self._pattern_repo.get_active_patterns(user_id=user_id)
        result: List[PersonalPattern] = []

        for candidate in candidates:
            # Find matching active pattern by domain and description
            matching_existing = next(
                (p for p in existing_patterns if p.domain == candidate.domain and p.description == candidate.description),
                None,
            )

            if matching_existing:
                # Evolve existing pattern with all new evidence
                for eid in candidate.evidence_ids:
                    if eid not in matching_existing.evidence_ids:
                        matching_existing.add_evidence(eid, observed_at=candidate.last_observed_at)

                matching_existing.update_confidence(self.calculate_confidence(len(matching_existing.evidence_ids)))
                if len(matching_existing.evidence_ids) >= 5 and matching_existing.confidence >= 0.80:
                    matching_existing.status = PatternStatus.CONFIRMED

                updated = await self._pattern_repo.update(matching_existing)
                result.append(updated)
            else:
                created = await self._pattern_repo.create(candidate)
                result.append(created)

        return result

    def evaluate_evidence(
        self,
        pattern: PersonalPattern,
        new_experience: Experience,
        is_supporting: bool = True,
    ) -> PersonalPattern:
        """Evolve an existing pattern hypothesis with new incoming evidence.

        Args:
            pattern: The existing PersonalPattern domain entity.
            new_experience: Incoming new experience observation.
            is_supporting: True if experience reinforces pattern, False if it contradicts/weakens.

        Returns:
            PersonalPattern: Updated pattern entity.
        """
        # User isolation invariant
        exp_user = self._normalize_user_id(new_experience.user_id)
        if exp_user is not None and exp_user != pattern.user_id:
            logger.warning(
                "User isolation violation prevented during evidence evaluation: pattern user_id %s != experience user_id %s",
                pattern.user_id,
                new_experience.user_id,
            )
            return pattern

        if is_supporting:
            exp_time = (
                new_experience.created_at.replace(tzinfo=timezone.utc)
                if new_experience.created_at.tzinfo is None
                else new_experience.created_at
            ) if isinstance(new_experience.created_at, datetime) else utc_now()

            pattern.add_evidence(experience_id=new_experience.id, observed_at=exp_time)
            new_confidence = self.calculate_confidence(len(pattern.evidence_ids))
            pattern.update_confidence(new_confidence)

            # Transition to CONFIRMED if evidence is strong (e.g. >= 5 distinct evidence items)
            if len(pattern.evidence_ids) >= 5 and pattern.confidence >= 0.80:
                pattern.status = PatternStatus.CONFIRMED
        else:
            # Weaken confidence when contradictory evidence arrives
            pattern.weaken(confidence_reduction=0.15)

        return pattern

    async def supersede_pattern(
        self,
        pattern_id: Union[uuid.UUID, PersonalPattern],
        user_id: Optional[uuid.UUID] = None,
        superseded_by_id: Optional[Union[uuid.UUID, PersonalPattern]] = None,
    ) -> Optional[PersonalPattern]:
        """Mark an older pattern as superseded by a newer pattern without deleting history."""
        if isinstance(pattern_id, PersonalPattern):
            old_pattern = pattern_id
            target_sup_id = superseded_by_id.id if isinstance(superseded_by_id, PersonalPattern) else superseded_by_id
            if target_sup_id:
                old_pattern.supersede_with(new_pattern_id=target_sup_id)
            if self._pattern_repo:
                await self._pattern_repo.update(old_pattern)
            return old_pattern

        if not self._pattern_repo or not user_id:
            return None

        old_pat = await self._pattern_repo.get_by_id(pattern_id=pattern_id, user_id=user_id)
        if not old_pat:
            return None

        target_id = superseded_by_id.id if isinstance(superseded_by_id, PersonalPattern) else (superseded_by_id or uuid.uuid4())
        old_pat.supersede_with(new_pattern_id=target_id)
        return await self._pattern_repo.update(old_pat)

    async def save_pattern(self, pattern: PersonalPattern) -> PersonalPattern:
        """Persist a pattern via the repository if configured."""
        if self._pattern_repo is not None:
            return await self._pattern_repo.create(pattern)
        return pattern

    async def update_pattern(self, pattern: PersonalPattern) -> PersonalPattern:
        """Update an existing pattern via the repository if configured."""
        if self._pattern_repo is not None:
            return await self._pattern_repo.update(pattern)
        return pattern
