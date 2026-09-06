from datetime import datetime, timedelta, timezone
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

from personal_ai.application.memory.quality_service import MemoryQualityService
from personal_ai.core.logger import get_logger
from personal_ai.domain.experience import Experience
from personal_ai.domain.pattern.entity import PersonalPattern
from personal_ai.domain.pattern.enums import PatternStatus
from personal_ai.domain.reflection.enums import ReflectionStatus, ReflectionType
from personal_ai.domain.reflection.models import Reflection

logger = get_logger(__name__)


def _utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


def _normalize_user_id(user_id: Any) -> Optional[uuid.UUID]:
    """Convert any user_id representation to UUID safely. Fails closed on missing or malformed input."""
    if user_id is None:
        return None
    if isinstance(user_id, uuid.UUID):
        return user_id
    if isinstance(user_id, str):
        cleaned = user_id.strip()
        if not cleaned:
            return None
        try:
            return uuid.UUID(cleaned)
        except (ValueError, AttributeError):
            return None
    return None


class ReflectionService:
    """Deterministic, evidence-grounded Reflection Engine for Second Brain AI (PR #28).

    Core Architecture:
        Experiences (what happened)
            +
        Personal Patterns (behavioral hypotheses)
            +
        MemoryQualityService (validation, lifecycle filtering & deduplication)
            ↓
        Reflection Engine
            ↓
        Traceable Reflection (observational change, reinforcement, weakening, or inconsistency)

    Safety Invariants:
        1. Purely Observational: Describes evidence-backed shifts, continuations, or inconsistencies.
        2. Strictly Non-Diagnostic: Never asserts personality traits, medical/clinical labels, or motivation/intent assumptions.
        3. Strict User Isolation: All evidence and patterns are validated and isolated to the authenticated user.
        4. Full Traceability: Every reflection links directly to the specific evidence_ids and pattern_ids that produced it.
    """

    # Generic stop words that cannot independently establish subject compatibility
    _GENERIC_STOP_WORDS: Set[str] = {
        "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", "from",
        "by", "for", "with", "about", "against", "between", "into", "through", "during",
        "before", "after", "above", "below", "to", "in", "on", "off", "over", "under",
        "again", "further", "then", "once", "here", "there", "all", "any", "both", "each",
        "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "can", "will", "just", "don", "should", "now",
        "i", "me", "my", "myself", "we", "our", "ours", "you", "your", "he", "him", "his",
        "she", "her", "they", "them", "what", "which", "who", "whom", "this", "that", "these",
        "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "having", "do", "does", "did", "doing", "would", "could", "want", "need", "like",
        "work", "project", "task", "goal", "habit", "routine", "exercise", "morning", "night",
        "time", "plan", "activity", "feel", "feeling", "recently", "today", "yesterday",
        "tomorrow", "hold", "pause", "paused", "stalled", "stuck", "touch", "touched",
        "progress", "weeks", "days", "months", "started", "working", "done", "make", "build",
        "complete", "finish", "create", "user", "tends", "usually", "often", "always",
        "suggests", "revisit", "schedule", "track", "check", "looking", "trying",
    }

    # Explicit phrases indicating inactivity or stalled progress
    _INACTIVITY_PHRASES: List[str] = [
        "inactive",
        "no progress",
        "haven't touched",
        "haven't worked on",
        "have not worked on",
        "on hold",
        "paused",
        "not worked on",
        "no progress recently",
        "stalled",
        "zero progress",
        "haven't studied",
        "haven't coded",
        "haven't practiced",
        "haven't exercised",
        "not touched for",
        "haven't done",
    ]

    # Time slot clusters
    _SLOT_DEFINITIONS: Dict[str, Set[str]] = {
        "night": {"night", "evening", "late night", "10pm", "11pm", "midnight", "tonight", "last night"},
        "morning": {"morning", "early morning", "6am", "7am", "8am", "9am"},
        "afternoon": {"afternoon", "midday", "lunchtime", "2pm", "3pm", "4pm"},
        "weekend": {"weekend", "weekends", "saturday", "sunday"},
        "weekday": {"weekday", "weekdays", "monday", "tuesday", "wednesday", "thursday", "friday"},
    }

    def __init__(
        self,
        quality_service: Optional[MemoryQualityService] = None,
    ) -> None:
        """Initialize ReflectionService with canonical MemoryQualityService."""
        self._quality_service = quality_service or MemoryQualityService()

    def _extract_subject_words(self, text: str) -> Set[str]:
        """Extract meaningful, non-generic subject keywords (length >= 2) from text."""
        if not text:
            return set()
        words = re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text.lower())
        return {w for w in words if w not in self._GENERIC_STOP_WORDS}

    def _detect_time_slot(self, text: str) -> Optional[str]:
        """Detect which time slot (if any) is mentioned in the text."""
        text_lower = text.lower()
        for slot_name, keywords in self._SLOT_DEFINITIONS.items():
            if any(re.search(r"\b" + re.escape(kw) + r"\b", text_lower) for kw in keywords):
                return slot_name
        return None

    def _calculate_confidence(self, evidence_count: int, base: float = 0.55) -> float:
        """Calculate deterministic bounded confidence from evidence count.

        Count scale:
            1 -> 0.52
            2 -> 0.55
            3 -> 0.65
            4 -> 0.72
            5 -> 0.80
            6+ -> 0.86 (capped strictly below 0.90)
        """
        if evidence_count <= 1:
            return 0.52
        elif evidence_count == 2:
            return 0.55
        elif evidence_count == 3:
            return 0.65
        elif evidence_count == 4:
            return 0.72
        elif evidence_count == 5:
            return 0.80
        else:
            return min(0.86, 0.80 + (evidence_count - 5) * 0.02)

    def reflect_on_pattern(
        self,
        user_id: uuid.UUID,
        pattern: PersonalPattern,
        experiences: List[Experience],
        time_window_days: int = 30,
        reference_time: Optional[datetime] = None,
    ) -> Optional[Reflection]:
        """Compare recent evidence against a single personal pattern and produce a reflection if justified.

        Args:
            user_id: Authenticated user UUID.
            pattern: PersonalPattern entity hypothesis.
            experiences: List of quality-filtered, user-isolated experiences.
            time_window_days: Number of days in the reflection lookback window.
            reference_time: Optional reference UTC timestamp (defaults to current UTC time).

        Returns:
            Optional[Reflection]: Generated reflection if evidence threshold is met, else None.
        """
        norm_user = _normalize_user_id(user_id)
        if not norm_user:
            return None

        # Verify pattern user_id matches authenticated user
        pat_user = _normalize_user_id(pattern.user_id)
        if pat_user is None or pat_user != norm_user:
            logger.warning("User isolation violation prevented: pattern user_id mismatch.")
            return None

        # Filter pattern status (HYPOTHESIS and CONFIRMED only)
        status_val = pattern.status.value if hasattr(pattern.status, "value") else str(pattern.status).upper()
        if status_val not in (PatternStatus.HYPOTHESIS.value, PatternStatus.CONFIRMED.value):
            return None

        now = reference_time or _utc_now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        # 1. Filter experiences within time_window_days
        window_start = now - timedelta(days=time_window_days)
        recent_experiences: List[Experience] = []
        for exp in experiences:
            exp_user = _normalize_user_id(exp.user_id)
            if exp_user is None or exp_user != norm_user:
                continue

            created = exp.created_at if isinstance(exp.created_at, datetime) else None
            if created:
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                if created < window_start:
                    continue
            recent_experiences.append(exp)

        if not recent_experiences:
            return None

        # 2. Extract pattern subject keywords and pattern slot
        pat_desc = pattern.description.strip()
        pat_subjects = self._extract_subject_words(pat_desc)
        pat_slot = self._detect_time_slot(pat_desc)

        # 3. Categorize matching recent experiences
        supporting_exps: List[Experience] = []
        conflicting_exps: List[Experience] = []
        inactivity_exps: List[Experience] = []

        for exp in recent_experiences:
            exp_content = exp.content.strip()
            exp_temporal = (exp.temporal_context or "").strip()
            combined_text = f"{exp_content} {exp_temporal}".lower()

            exp_subjects = self._extract_subject_words(combined_text)

            # Check conservative subject compatibility
            if not pat_subjects.intersection(exp_subjects):
                continue

            # Check for explicit inactivity / stall
            if any(phrase in combined_text for phrase in self._INACTIVITY_PHRASES):
                inactivity_exps.append(exp)
                continue

            # Check time slot compatibility
            exp_slot = self._detect_time_slot(combined_text)

            if pat_slot and exp_slot:
                if exp_slot == pat_slot:
                    supporting_exps.append(exp)
                else:
                    conflicting_exps.append(exp)
            else:
                # If no specific slot in pattern or experience, general subject progress supports pattern
                supporting_exps.append(exp)

        # 4. Evaluate evidence thresholds deterministically
        subject_display = " ".join(sorted(list(pat_subjects))) or "activity"

        # Case A: PATTERN_CHANGE (>= 2 consistent shifting observations)
        if len(conflicting_exps) >= 2:
            new_slots = {self._detect_time_slot(e.content) for e in conflicting_exps if self._detect_time_slot(e.content)}
            slot_str = " / ".join(filter(None, new_slots)) or "a new schedule"
            obs_text = (
                f"Recent activity suggests your {subject_display} routine may be shifting from your "
                f"established pattern toward {slot_str}."
            )
            ev_ids = [e.id for e in conflicting_exps]
            conf = self._calculate_confidence(len(conflicting_exps))
            return Reflection(
                user_id=norm_user,
                type=ReflectionType.PATTERN_CHANGE,
                observation=obs_text,
                confidence=conf,
                evidence_ids=ev_ids,
                pattern_ids=[pattern.id],
                time_window_days=time_window_days,
                domain=pattern.domain,
            )

        # Case B: PATTERN_REINFORCEMENT (>= 2 supporting observations)
        if len(supporting_exps) >= 2 and len(supporting_exps) > len(conflicting_exps):
            obs_text = f"Recent activity continues to support your established pattern of '{pat_desc}'."
            ev_ids = [e.id for e in supporting_exps]
            conf = self._calculate_confidence(len(supporting_exps))
            # Minor adjustment based on pattern confidence
            conf = round(min(conf + min(0.04 * pattern.confidence, 0.04), 0.88), 4)
            return Reflection(
                user_id=norm_user,
                type=ReflectionType.PATTERN_REINFORCEMENT,
                observation=obs_text,
                confidence=conf,
                evidence_ids=ev_ids,
                pattern_ids=[pattern.id],
                time_window_days=time_window_days,
                domain=pattern.domain,
            )

        # Case C: PATTERN_WEAKENING (>= 2 inactivity / stall observations)
        if len(inactivity_exps) >= 2:
            obs_text = f"Recent {subject_display} activity appears lower than the established pattern."
            ev_ids = [e.id for e in inactivity_exps]
            conf = min(self._calculate_confidence(len(inactivity_exps)), 0.80)
            return Reflection(
                user_id=norm_user,
                type=ReflectionType.PATTERN_WEAKENING,
                observation=obs_text,
                confidence=conf,
                evidence_ids=ev_ids,
                pattern_ids=[pattern.id],
                time_window_days=time_window_days,
                domain=pattern.domain,
            )

        # Case D: PATTERN_INCONSISTENCY (exactly 1 conflicting observation)
        if len(conflicting_exps) == 1:
            conflict_exp = conflicting_exps[0]
            obs_text = (
                f"Recent activity includes an observation ('{conflict_exp.content}') that differs "
                f"from your established pattern."
            )
            return Reflection(
                user_id=norm_user,
                type=ReflectionType.PATTERN_INCONSISTENCY,
                observation=obs_text,
                confidence=0.52,
                evidence_ids=[conflict_exp.id],
                pattern_ids=[pattern.id],
                time_window_days=time_window_days,
                domain=pattern.domain,
            )

        return None

    async def analyze(
        self,
        user_id: uuid.UUID,
        experiences: List[Experience],
        patterns: List[PersonalPattern],
        time_window_days: int = 30,
        reference_time: Optional[datetime] = None,
    ) -> List[Reflection]:
        """Analyze quality-filtered experiences against established patterns to generate reflections.

        Pipeline:
            Valid/Deduped Experiences + Active Patterns
                ↓
            Filter Lookback Window (Experiences)
                ↓
            Compare Pattern vs Recent Evidence
                ↓
            Generate Traceable Reflections
                ↓
            Deduplicate Reflections

        Args:
            user_id: Authenticated user UUID for strict isolation.
            experiences: List of candidate Experience entities.
            patterns: List of candidate PersonalPattern entities.
            time_window_days: Window lookback in days (default 30, max 90).
            reference_time: Optional reference UTC timestamp (defaults to current UTC time).

        Returns:
            List[Reflection]: Traceable, user-isolated reflection entities.
        """
        norm_user = _normalize_user_id(user_id)
        if not norm_user:
            logger.warning("ReflectionService.analyze called without valid user_id: %s", user_id)
            return []

        if not experiences or not patterns:
            return []

        # Bound time_window_days
        window_days = max(1, min(int(time_window_days), 90))
        now = reference_time or _utc_now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        # 1. Delegate experience validation, lifecycle filtering, and deduplication to MemoryQualityService
        filtered_experiences = self._quality_service.filter_and_deduplicate_experiences(
            user_id=norm_user,
            experiences=experiences,
            is_historical=False,
        )

        # 2. Delegate pattern validation, lifecycle filtering, and deduplication to MemoryQualityService
        filtered_patterns = self._quality_service.filter_and_deduplicate_patterns(
            user_id=norm_user,
            patterns=patterns,
        )

        if not filtered_experiences or not filtered_patterns:
            return []

        logger.info(
            "Analyzing reflections [user_id=%s, experiences_in=%d, patterns_in=%d, time_window_days=%d]",
            norm_user,
            len(filtered_experiences),
            len(filtered_patterns),
            window_days,
        )

        # 3. Generate reflections for each pattern
        raw_reflections: List[Reflection] = []
        for pat in filtered_patterns:
            if not isinstance(pat, PersonalPattern):
                continue
            ref = self.reflect_on_pattern(
                user_id=norm_user,
                pattern=pat,
                experiences=filtered_experiences,
                time_window_days=window_days,
                reference_time=now,
            )
            if ref:
                raw_reflections.append(ref)

        # 4. In-memory per-analysis deduplication
        deduped_reflections: List[Reflection] = []
        seen_keys: Set[Tuple[uuid.UUID, uuid.UUID, ReflectionType, int]] = set()

        for ref in raw_reflections:
            pid = ref.pattern_ids[0] if ref.pattern_ids else uuid.uuid4()
            dedup_key = (ref.user_id, pid, ref.type, ref.time_window_days)
            if dedup_key not in seen_keys:
                seen_keys.add(dedup_key)
                deduped_reflections.append(ref)

        logger.info(
            "Reflection analysis complete [user_id=%s, generated_count=%d]",
            norm_user,
            len(deduped_reflections),
        )

        return deduped_reflections
