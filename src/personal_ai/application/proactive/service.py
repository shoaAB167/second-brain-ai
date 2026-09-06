from datetime import datetime, timedelta, timezone
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

from personal_ai.application.memory.quality_service import MemoryQualityService
from personal_ai.core.logger import get_logger
from personal_ai.domain.experience import (
    Experience,
    PersonalContext,
    PersonalContextItem,
    PersonalPatternContextItem,
)
from personal_ai.domain.pattern.entity import PersonalPattern
from personal_ai.domain.pattern.enums import PatternStatus
from personal_ai.domain.proactive.enums import ProactivePriority, ProactiveSignalType
from personal_ai.domain.proactive.models import ProactiveCandidate, ProactiveSignal

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


class ProactiveIntelligenceService:
    """Deterministic, evidence-evaluated proactive intelligence service for PR #22 & PR #27.

    Philosophy:
        Observation -> Signal -> Supporting Evidence -> Pattern/Context -> Proactive Candidate

    Invariants:
        1. ProactiveCandidate != Action: Produces passive proposal containers only.
           Does NOT execute tools, send notifications, modify memory, or make LLM calls.
        2. Preserves Uncertainty: Patterns are hypotheses, not facts or personality claims.
           Never diagnoses conditions, makes causal claims, or turns temporary states into personality traits.
        3. Strict User Isolation: Every analysis is scoped exclusively to authenticated user_id.
        4. Canonical Memory Quality: Reuses MemoryQualityService for lifecycle eligibility,
           conservative deduplication with temporal preservation, and user-scoped validation.
        5. Conservative Signal Generation: Prefers NO SIGNAL over FALSE POSITIVE when evidence is insufficient.
        6. Subject-Specific Pattern Support: Patterns only support signals with matching specific subjects.
           Generic words (work, project, morning, night, routine, exercise) never independently establish relevance.
        7. Subject-Specific Precedence: Current explicit user statements override historical patterns only
           when the message genuinely contradicts the specific behavior/routine.
    """

    # Generic stop words and generic domain terms that cannot independently establish subject compatibility
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

    # Emotional/Mental state keywords for observational repeated state detection
    _STATE_KEYWORDS = {
        "tired": ["tired", "exhausted", "fatigued", "drained", "weary"],
        "low energy": ["low energy", "lethargic", "no energy", "sluggish", "depleted"],
        "overwhelmed": ["overwhelmed", "too much going on", "swamped", "drowning in work"],
        "stressed": ["stressed", "stress", "tense", "under pressure", "strained"],
        "anxious": ["anxious", "anxiety", "worried", "nervous", "uneasy"],
        "demotivated": ["demotivated", "unmotivated", "lack of motivation", "struggling to start"],
        "frustrated": ["frustrated", "annoyed", "irritated", "stuck"],
        "sad": ["sad", "down", "low mood", "feeling blue"],
    }

    # Explicit phrases indicating goal inactivity (goal age alone is never sufficient)
    _EXPLICIT_INACTIVITY_PHRASES = [
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
    ]

    # Explicit commitment indicator phrases
    _EXPLICIT_COMMITMENT_INDICATORS = [
        r"\b(committed to|promised to|agreed to|solemnly promised)\b",
        r"\b(my commitment is|commitment to)\b",
    ]

    # Missed commitment evidence keywords
    _MISSED_INDICATORS = [
        r"\b(didn'?t|did not|couldn'?t|could not|failed to|missed|forgot to|haven'?t|have not)\b",
        r"\b(fell behind|delayed|unable to finish|skipped|postponed|never completed)\b",
    ]

    def __init__(
        self,
        quality_service: Optional[MemoryQualityService] = None,
        memory_quality_service: Optional[MemoryQualityService] = None,
    ) -> None:
        """Initialize ProactiveIntelligenceService with canonical MemoryQualityService."""
        self._quality_service = memory_quality_service or quality_service or MemoryQualityService()

    def _extract_items(
        self,
        user_id: uuid.UUID,
        context: Optional[Union[PersonalContext, List[Experience], List[PersonalContextItem]]] = None,
        personal_context: Optional[PersonalContext] = None,
        experiences: Optional[List[Experience]] = None,
    ) -> List[Dict[str, Any]]:
        """Normalize, quality-filter, and deduplicate observations using MemoryQualityService.

        Delegates candidate validation, lifecycle filtering, and conservative deduplication
        with temporal preservation directly to MemoryQualityService.
        """
        raw_experiences: List[Experience] = []
        raw_context_items: List[PersonalContextItem] = []

        # 1. From context argument
        if isinstance(context, PersonalContext):
            personal_context = context
        elif isinstance(context, list):
            for item in context:
                if isinstance(item, Experience):
                    raw_experiences.append(item)
                elif isinstance(item, PersonalContextItem):
                    raw_context_items.append(item)

        # 2. From personal_context
        if personal_context and not personal_context.is_empty:
            # Enforce fail-closed user isolation at container level
            if _normalize_user_id(personal_context.user_id) == user_id:
                raw_context_items.extend(personal_context.items)
            else:
                logger.warning(
                    "User isolation violation prevented: personal_context user_id mismatch [expected=%s, got=%s]",
                    user_id,
                    personal_context.user_id,
                )

        # 3. From experiences list
        if experiences:
            raw_experiences.extend(experiences)

        # 4. Delegate experience validation, lifecycle filtering, and deduplication to MemoryQualityService
        quality_filtered_experiences: List[Experience] = []
        if raw_experiences:
            quality_filtered_experiences = self._quality_service.filter_and_deduplicate_experiences(
                user_id=user_id,
                experiences=raw_experiences,
                is_historical=False,
            )

        # 5. Format standardized observation dicts
        records: List[Dict[str, Any]] = []
        seen_ids: Set[uuid.UUID] = set()

        for exp in quality_filtered_experiences:
            if exp.id in seen_ids:
                continue
            seen_ids.add(exp.id)

            emotion_val = None
            if exp.emotional_context:
                emotion_val = exp.emotional_context.emotion

            created = exp.created_at if isinstance(exp.created_at, datetime) else None
            if created and created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)

            type_str = exp.type.value if hasattr(exp.type, "value") else str(exp.type or "")
            lifecycle_str = (
                exp.lifecycle.value
                if hasattr(exp.lifecycle, "value")
                else str(exp.lifecycle or "STABLE").upper()
            )
            life_status = (
                exp.lifecycle_status.value
                if hasattr(exp.lifecycle_status, "value")
                else str(exp.lifecycle_status or "ACTIVE").upper()
            )

            records.append(
                {
                    "id": exp.id,
                    "content": exp.content,
                    "type": type_str.upper(),
                    "domain": exp.domain,
                    "lifecycle": lifecycle_str,
                    "lifecycle_status": life_status,
                    "temporal_context": exp.temporal_context,
                    "emotion": emotion_val,
                    "created_at": created,
                }
            )

        for ctx_item in raw_context_items:
            if ctx_item.experience_id in seen_ids:
                continue
            seen_ids.add(ctx_item.experience_id)

            emotion_val = None
            if ctx_item.emotional_context and isinstance(ctx_item.emotional_context, dict):
                emotion_val = ctx_item.emotional_context.get("emotion")

            created = ctx_item.created_at if isinstance(ctx_item.created_at, datetime) else None
            if created and created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)

            records.append(
                {
                    "id": ctx_item.experience_id,
                    "content": ctx_item.content,
                    "type": (ctx_item.type or "").upper(),
                    "domain": ctx_item.domain,
                    "lifecycle_status": (ctx_item.lifecycle_status or "ACTIVE").upper(),
                    "temporal_context": ctx_item.temporal_context,
                    "emotion": emotion_val,
                    "created_at": created,
                }
            )

        return records

    def _extract_patterns(
        self,
        user_id: uuid.UUID,
        context: Optional[Union[PersonalContext, List[Any]]] = None,
        personal_context: Optional[PersonalContext] = None,
        patterns: Optional[List[Union[PersonalPattern, PersonalPatternContextItem]]] = None,
    ) -> List[Dict[str, Any]]:
        """Normalize, quality-filter, and isolate active PersonalPattern hypotheses using MemoryQualityService."""
        raw_patterns: List[Any] = []

        # 1. From context if PersonalContext
        if isinstance(context, PersonalContext):
            personal_context = context

        # 2. From personal_context.patterns
        if personal_context and getattr(personal_context, "patterns", None):
            if _normalize_user_id(personal_context.user_id) == user_id:
                raw_patterns.extend(personal_context.patterns)
            else:
                logger.warning(
                    "User isolation violation prevented: personal_context patterns user_id mismatch [expected=%s, got=%s]",
                    user_id,
                    personal_context.user_id,
                )

        # 3. From explicit patterns list
        if patterns:
            raw_patterns.extend(patterns)

        # 4. Delegate pattern validation, lifecycle filtering, and deduplication to MemoryQualityService
        filtered_pats = self._quality_service.filter_and_deduplicate_patterns(
            user_id=user_id,
            patterns=raw_patterns,
        )

        pattern_records: List[Dict[str, Any]] = []
        for pat in filtered_pats:
            if isinstance(pat, PersonalPattern):
                status_val = pat.status.value if hasattr(pat.status, "value") else str(pat.status).upper()
                domain_val = pat.domain.value if hasattr(pat.domain, "value") else str(pat.domain)
                pattern_records.append(
                    {
                        "id": pat.id,
                        "description": pat.description,
                        "domain": domain_val,
                        "confidence": pat.confidence,
                        "status": status_val,
                        "evidence_count": len(pat.evidence_ids) if pat.evidence_ids else 0,
                    }
                )
            elif isinstance(pat, PersonalPatternContextItem):
                pattern_records.append(
                    {
                        "id": pat.pattern_id,
                        "description": pat.description,
                        "domain": pat.domain,
                        "confidence": pat.confidence,
                        "status": (pat.status or "HYPOTHESIS").upper(),
                        "evidence_count": pat.evidence_count,
                    }
                )
            elif isinstance(pat, dict):
                pattern_records.append(
                    {
                        "id": pat.get("id") or pat.get("pattern_id"),
                        "description": pat.get("description", ""),
                        "domain": pat.get("domain", "GENERAL"),
                        "confidence": float(pat.get("confidence", 0.70)),
                        "status": str(pat.get("status", "HYPOTHESIS")).upper(),
                        "evidence_count": int(pat.get("evidence_count", 1)),
                    }
                )

        return pattern_records

    def _extract_subject_words(self, text: str) -> Set[str]:
        """Extract meaningful, non-generic subject keywords (length >= 2) from text."""
        if not text:
            return set()
        words = re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text.lower())
        return {w for w in words if w not in self._GENERIC_STOP_WORDS}

    def _is_pattern_contradicted_by_message(self, pattern_desc: str, current_message: Optional[str]) -> bool:
        """Check deterministically if the current user message explicitly contradicts a historical pattern.

        Rules:
        1. Generic words (e.g. 'switched', 'changed', 'stopped') with unrelated objects (e.g. 'switched companies')
           do NOT constitute a contradiction of a routine pattern.
        2. Genuine contradiction requires:
           (a) Explicit negation of the pattern's specific routine/slot (e.g. 'no longer work at night', 'stopped running').
           (b) An explicit temporal shift to a conflicting time slot for the same subject/activity
               (e.g. pattern has 'night', message states 'switched to ... in the morning now').
        """
        if not current_message or not pattern_desc:
            return False

        msg_lower = current_message.lower()
        pat_lower = pattern_desc.lower()

        # Slot definition pairs for routine contradiction
        slot_pairs = [
            ({"night", "evening", "late night"}, {"morning", "early morning", "afternoon", "day"}),
            ({"morning", "early morning"}, {"night", "evening", "afternoon", "late night"}),
            ({"weekend", "weekends"}, {"weekday", "weekdays"}),
            ({"weekday", "weekdays"}, {"weekend", "weekends"}),
        ]

        # Check for slot shift
        for source_slots, conflicting_slots in slot_pairs:
            has_source = any(re.search(r"\b" + re.escape(s) + r"\b", pat_lower) for s in source_slots)
            if not has_source:
                continue

            # Direct negation of source slot in message
            for s in source_slots:
                if re.search(r"\b(no longer|not|stopped|don'?t)\b.*\b" + re.escape(s) + r"\b", msg_lower):
                    return True

            # Explicit shift to a conflicting slot
            for c in conflicting_slots:
                if re.search(r"\b" + re.escape(c) + r"\b", msg_lower):
                    shift_patterns = [
                        r"\b(switched|changed|moved|shifted)\b.*\b(to|into)\b.*\b" + re.escape(c) + r"\b",
                        r"\b(in the|at)\b\s+" + re.escape(c) + r"\s+(now|instead)\b",
                        r"\bnow\b.*\b(in the|at)\b\s+" + re.escape(c) + r"\b",
                    ]
                    if any(re.search(sp, msg_lower) for sp in shift_patterns):
                        return True

        # Check for explicit discontinuation of specific action/activity mentioned in pattern
        action_keywords = ["run", "running", "code", "coding", "study", "studying", "write", "writing", "exercise", "workout", "gym"]
        for act in action_keywords:
            if re.search(r"\b" + re.escape(act) + r"\b", pat_lower):
                if re.search(r"\b(stopped|quit|no longer|gave up on)\b\s+" + re.escape(act) + r"\b", msg_lower):
                    return True

        return False

    def _find_supporting_pattern(
        self,
        signal: ProactiveSignal,
        patterns: List[Dict[str, Any]],
        records_by_id: Dict[uuid.UUID, Dict[str, Any]],
        current_message: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Find conservative, subject-compatible active pattern supporting the proactive signal.

        Rules:
        1. A pattern supports a signal ONLY when it is meaningfully related to the SAME specific subject.
        2. Generic words (work, project, morning, night, routine, exercise) can NEVER independently make a pattern relevant.
        3. Current user statements strictly supersede contradictory historical patterns on the same subject.
        """
        if not patterns:
            return None

        # Extract subject keywords from signal's related experiences
        signal_subject_words: Set[str] = set()
        for eid in signal.related_experience_ids:
            rec = records_by_id.get(eid)
            if rec:
                signal_subject_words.update(self._extract_subject_words(rec.get("content", "")))

        # Also extract state subject for REPEATED_STATE
        if signal.type == ProactiveSignalType.REPEATED_STATE:
            for cluster_name in self._STATE_KEYWORDS.keys():
                if cluster_name in signal.reason.lower():
                    signal_subject_words.add(cluster_name)
                    signal_subject_words.update(self._STATE_KEYWORDS[cluster_name])

        for pat in patterns:
            pat_desc = pat.get("description", "")
            pat_words = self._extract_subject_words(pat_desc)

            # Check conservative subject compatibility (specific keyword overlap required)
            common_subjects = signal_subject_words.intersection(pat_words)
            if not common_subjects:
                continue

            # Check if current user message contradicts this pattern
            if self._is_pattern_contradicted_by_message(pat_desc, current_message):
                logger.debug("Suppressed historical pattern '%s' due to explicit user contradiction.", pat_desc)
                continue

            return pat

        return None

    def _has_meaningful_goal_activity(
        self,
        goal_content: str,
        goal_id: uuid.UUID,
        records: List[Dict[str, Any]],
        current_message: Optional[str],
        now: datetime,
    ) -> bool:
        """Check whether there is strong, non-generic evidence of recent progress or activity toward a goal."""
        words = list(self._extract_subject_words(goal_content))
        if not words:
            return False

        # 1. Check current user message
        if current_message:
            msg_lower = current_message.lower()
            has_inactivity = any(phrase in msg_lower for phrase in self._EXPLICIT_INACTIVITY_PHRASES)
            if not has_inactivity:
                matched_words = [w for w in words if w in msg_lower]
                if len(matched_words) >= 2 or (len(words) == 1 and words[0] in msg_lower and len(words[0]) >= 4):
                    return True

        # 2. Check recent experience records
        for other in records:
            if other["id"] == goal_id:
                continue

            other_type = other.get("type", "")
            if other_type in ("EVENT", "PROJECT", "HABIT", "DECISION", "STATE"):
                other_content = other["content"].lower()
                if any(phrase in other_content for phrase in self._EXPLICIT_INACTIVITY_PHRASES):
                    continue
                matched_words = [w for w in words if w in other_content]
                if len(matched_words) >= 2 or (len(words) == 1 and words[0] in other_content and len(words[0]) >= 4):
                    other_created = other.get("created_at")
                    if other_created is None or (now - other_created) < timedelta(days=14):
                        return True

        return False

    def _detect_goal_inactivity_signals(
        self,
        records: List[Dict[str, Any]],
        current_message: Optional[str],
        now: datetime,
    ) -> List[ProactiveSignal]:
        """Detect potential goal inactivity signals based strictly on explicit inactivity evidence."""
        signals: List[ProactiveSignal] = []
        msg_lower = (current_message or "").lower()

        for rec in records:
            if rec["type"] != "GOAL" and rec.get("domain") != "goal":
                continue

            if rec["lifecycle_status"] in ("EXPIRED", "SUPERSEDED"):
                continue

            goal_content = rec["content"]
            goal_id = rec["id"]
            temporal_ctx = (rec.get("temporal_context") or "").lower()
            content_lower = goal_content.lower()

            goal_words = list(self._extract_subject_words(goal_content))

            # 1. Detect explicit inactivity evidence relevant to the goal FIRST
            has_explicit_inactivity = False
            confidence = 0.72

            combined_stored_text = f"{content_lower} {temporal_ctx}"
            for phrase in self._EXPLICIT_INACTIVITY_PHRASES:
                if phrase in combined_stored_text:
                    has_explicit_inactivity = True
                    confidence = 0.78
                    break

            # Also check if current message indicates explicit inactivity for this goal
            if not has_explicit_inactivity and msg_lower:
                for phrase in self._EXPLICIT_INACTIVITY_PHRASES:
                    if phrase in msg_lower:
                        matched = [w for w in goal_words if w in msg_lower]
                        if len(matched) >= 2 or (len(goal_words) == 1 and goal_words[0] in msg_lower):
                            has_explicit_inactivity = True
                            confidence = 0.82
                            break

            # 2. If explicit inactivity is present, generate signal directly
            if has_explicit_inactivity:
                signals.append(
                    ProactiveSignal(
                        type=ProactiveSignalType.GOAL_INACTIVITY,
                        reason="The goal appears inactive based on available context.",
                        confidence=confidence,
                        related_experience_ids=[goal_id],
                    )
                )
                continue

            # 3. If not explicit inactivity, check if active
            if self._has_meaningful_goal_activity(goal_content, goal_id, records, current_message, now):
                continue

            # 4. If neither: NO SIGNAL (goal age alone is never sufficient)

        return signals

    def _detect_commitment_missed_signals(
        self,
        records: List[Dict[str, Any]],
        current_message: Optional[str],
        now: datetime,
    ) -> List[ProactiveSignal]:
        """Detect missed commitment signals conservatively based on explicit commitment and temporal evidence."""
        signals: List[ProactiveSignal] = []
        msg_lower = (current_message or "").lower()

        for rec in records:
            content_lower = rec["content"].lower()
            temporal_ctx = (rec.get("temporal_context") or "").lower()
            rec_type = rec.get("type", "")

            # 1. Require strong commitment evidence
            is_explicit_type = (rec_type == "COMMITMENT")
            has_commitment_phrasing = any(re.search(pat, content_lower) for pat in self._EXPLICIT_COMMITMENT_INDICATORS)

            if not (is_explicit_type or has_commitment_phrasing):
                continue

            commitment_id = rec["id"]

            # 2. Require evidence that deadline has passed
            is_past_deadline = False
            past_indicators = ["yesterday", "last week", "last night", "passed deadline", "overdue", "missed date"]
            if any(p in temporal_ctx for p in past_indicators) or any(p in content_lower for p in past_indicators):
                is_past_deadline = True

            future_indicators = ["tomorrow", "next week", "next month", "tonight", "in the future", "upcoming"]
            if any(f in temporal_ctx for f in future_indicators):
                is_past_deadline = False

            # 3. Require evidence that action was missed
            has_missed_evidence = False
            if any(re.search(pat, content_lower) for pat in self._MISSED_INDICATORS) or any(
                re.search(pat, temporal_ctx) for pat in self._MISSED_INDICATORS
            ):
                has_missed_evidence = True
            elif msg_lower and any(re.search(pat, msg_lower) for pat in self._MISSED_INDICATORS):
                keywords = list(self._extract_subject_words(content_lower))
                if any(kw in msg_lower for kw in keywords):
                    has_missed_evidence = True

            if is_past_deadline and has_missed_evidence:
                signals.append(
                    ProactiveSignal(
                        type=ProactiveSignalType.COMMITMENT_MISSED,
                        reason="Evidence indicates a planned commitment may have been missed.",
                        confidence=0.78,
                        related_experience_ids=[commitment_id],
                    )
                )

        return signals

    def _detect_repeated_state_signals(
        self,
        records: List[Dict[str, Any]],
        now: datetime,
    ) -> List[ProactiveSignal]:
        """Detect repeated state signals across distinct experiences with strictly observational semantics."""
        signals: List[ProactiveSignal] = []
        state_clusters: Dict[str, List[uuid.UUID]] = {}

        for rec in records:
            content_lower = rec["content"].lower()
            emotion = (rec.get("emotion") or "").lower()
            rec_id = rec["id"]
            rec_created = rec.get("created_at")

            # Check if record is within recent 30-day window
            if rec_created and (now - rec_created) > timedelta(days=30):
                continue

            detected_cluster = None

            # 1. Check explicit emotion field
            if emotion:
                for cluster_name, keywords in self._STATE_KEYWORDS.items():
                    if emotion == cluster_name or any(kw in emotion for kw in keywords):
                        detected_cluster = cluster_name
                        break

            # 2. Check content
            if not detected_cluster and (rec["type"] in ("STATE", "EMOTION_STATE", "") or emotion):
                for cluster_name, keywords in self._STATE_KEYWORDS.items():
                    if any(re.search(r"\b" + re.escape(kw) + r"\b", content_lower) for kw in keywords):
                        detected_cluster = cluster_name
                        break

            if detected_cluster:
                if detected_cluster not in state_clusters:
                    state_clusters[detected_cluster] = []
                if rec_id not in state_clusters[detected_cluster]:
                    state_clusters[detected_cluster].append(rec_id)

        # Evaluate repetition (count >= 2 distinct observations required)
        for cluster_name, exp_ids in state_clusters.items():
            if len(exp_ids) >= 2:
                count = len(exp_ids)
                confidence = round(min(0.70 + (count - 2) * 0.08, 0.86), 2)
                signals.append(
                    ProactiveSignal(
                        type=ProactiveSignalType.REPEATED_STATE,
                        reason=f"A similar state ('{cluster_name}') was recorded more than once recently.",
                        confidence=confidence,
                        related_experience_ids=exp_ids,
                    )
                )

        return signals

    def detect_signals(
        self,
        records: List[Dict[str, Any]],
        current_message: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> List[ProactiveSignal]:
        """Detect observation signals using the intermediate ProactiveSignal layer."""
        ref_time = now or _utc_now()
        if ref_time.tzinfo is None:
            ref_time = ref_time.replace(tzinfo=timezone.utc)

        signals: List[ProactiveSignal] = []
        signals.extend(self._detect_goal_inactivity_signals(records, current_message, ref_time))
        signals.extend(self._detect_commitment_missed_signals(records, current_message, ref_time))
        signals.extend(self._detect_repeated_state_signals(records, ref_time))
        return signals

    def generate_candidates(
        self,
        signals: List[ProactiveSignal],
        patterns: Optional[List[Dict[str, Any]]] = None,
        records: Optional[List[Dict[str, Any]]] = None,
        current_message: Optional[str] = None,
    ) -> List[ProactiveCandidate]:
        """Convert detected signals into evidence-evaluated, non-authoritative candidate proposals."""
        candidates: List[ProactiveCandidate] = []
        patterns_list = patterns or []
        records_by_id = {r["id"]: r for r in (records or []) if "id" in r}

        base_actions = {
            ProactiveSignalType.GOAL_INACTIVITY: (
                ProactivePriority.LOW,
                "Ask whether the user wants to revisit or work on this goal.",
            ),
            ProactiveSignalType.COMMITMENT_MISSED: (
                ProactivePriority.MEDIUM,
                "Ask whether the user wants to reschedule, adjust, or check in on this commitment.",
            ),
            ProactiveSignalType.REPEATED_STATE: (
                ProactivePriority.MEDIUM,
                "Ask how the user is feeling and whether they would like help understanding recent observations.",
            ),
        }

        for signal in signals:
            ev_count = len(signal.related_experience_ids)
            if ev_count == 0:
                continue

            # For REPEATED_STATE, enforce count >= 2 distinct observations
            if signal.type == ProactiveSignalType.REPEATED_STATE and ev_count < 2:
                continue

            priority, action = base_actions.get(
                signal.type,
                (ProactivePriority.LOW, "Check in with the user regarding recent observations."),
            )

            # Conservative, subject-compatible pattern matching
            supporting_pattern = self._find_supporting_pattern(
                signal=signal,
                patterns=patterns_list,
                records_by_id=records_by_id,
                current_message=current_message,
            )
            supporting_pattern_ids: List[uuid.UUID] = []
            confidence = signal.confidence

            if supporting_pattern and supporting_pattern.get("id"):
                pat_id = supporting_pattern["id"]
                if isinstance(pat_id, uuid.UUID):
                    supporting_pattern_ids.append(pat_id)
                elif isinstance(pat_id, str):
                    try:
                        supporting_pattern_ids.append(uuid.UUID(pat_id))
                    except ValueError:
                        pass

                pat_desc = supporting_pattern.get("description", "").strip()
                pat_conf = supporting_pattern.get("confidence", 0.70)

                # Bounded deterministic confidence bump
                confidence = round(min(confidence + min(0.04 * pat_conf, 0.04), 0.86), 4)

                # Non-authoritative hypothesis phrasing for suggestions
                if signal.type == ProactiveSignalType.GOAL_INACTIVITY:
                    if "night" in pat_desc.lower():
                        action = "Your project appears inactive recently. Your past activity suggests you often work on it at night. Would you like to revisit it tonight?"
                    else:
                        action = f"Your goal appears inactive recently. Based on your past routine hypothesis ('{pat_desc}'), would you like to revisit or schedule time for this?"
                elif signal.type == ProactiveSignalType.COMMITMENT_MISSED:
                    action = f"Check in on this commitment, taking into account your typical workflow hypothesis ('{pat_desc}')."

            # Format evidence summary
            if supporting_pattern_ids:
                obs_str = "1 observation" if ev_count == 1 else f"{ev_count} observations"
                pat_str = "1 supporting pattern" if len(supporting_pattern_ids) == 1 else f"{len(supporting_pattern_ids)} supporting patterns"
                ev_summary = f"Based on {obs_str} and {pat_str}."
            else:
                obs_str = "1 observation" if ev_count == 1 else f"{ev_count} recent observations"
                ev_summary = f"Based on {obs_str}."

            candidates.append(
                ProactiveCandidate(
                    signal_type=signal.type,
                    reason=signal.reason,
                    confidence=confidence,
                    priority=priority,
                    suggested_action=action,
                    related_experience_ids=signal.related_experience_ids,
                    evidence_count=ev_count,
                    supporting_pattern_ids=supporting_pattern_ids,
                    evidence_summary=ev_summary,
                )
            )

        return candidates

    def _deduplicate_candidates(self, candidates: List[ProactiveCandidate]) -> List[ProactiveCandidate]:
        """Perform in-memory per-analysis deduplication of proactive candidates."""
        seen: Set[Tuple[ProactiveSignalType, str]] = set()
        deduped: List[ProactiveCandidate] = []

        for candidate in candidates:
            exp_key = ",".join(sorted(str(eid) for eid in candidate.related_experience_ids))
            dedup_key = (candidate.signal_type, exp_key or candidate.reason)

            if dedup_key not in seen:
                seen.add(dedup_key)
                deduped.append(candidate)

        return deduped

    async def analyze(
        self,
        user_id: uuid.UUID,
        context: Optional[Union[PersonalContext, List[Experience], List[PersonalContextItem]]] = None,
        *,
        personal_context: Optional[PersonalContext] = None,
        experiences: Optional[List[Experience]] = None,
        patterns: Optional[List[Union[PersonalPattern, PersonalPatternContextItem]]] = None,
        current_message: Optional[str] = None,
        reference_time: Optional[datetime] = None,
    ) -> List[ProactiveCandidate]:
        """Analyze observations and detect evidence-evaluated proactive intervention proposals."""
        norm_user = _normalize_user_id(user_id)
        if not norm_user:
            logger.warning("ProactiveIntelligenceService.analyze called without valid user_id: %s", user_id)
            return []

        now = reference_time or _utc_now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        # 1. Normalize, quality-filter, and deduplicate records via MemoryQualityService
        records = self._extract_items(
            user_id=norm_user,
            context=context,
            personal_context=personal_context,
            experiences=experiences,
        )

        if not records:
            return []

        # 2. Extract and isolate active patterns via MemoryQualityService
        pattern_records = self._extract_patterns(
            user_id=norm_user,
            context=context,
            personal_context=personal_context,
            patterns=patterns,
        )

        logger.info(
            "Analyzing proactive intelligence signals [user_id=%s, observations_count=%d, patterns_count=%d]",
            norm_user,
            len(records),
            len(pattern_records),
        )

        # 3. Detect signals
        signals = self.detect_signals(
            records=records,
            current_message=current_message,
            now=now,
        )

        # 4. Evidence evaluation and candidate generation
        candidates = self.generate_candidates(
            signals=signals,
            patterns=pattern_records,
            records=records,
            current_message=current_message,
        )

        # 5. In-memory per-analysis deduplication
        deduped = self._deduplicate_candidates(candidates)

        logger.info(
            "Proactive intelligence analysis complete [user_id=%s, signals_count=%d, candidates_count=%d]",
            norm_user,
            len(signals),
            len(deduped),
        )

        return deduped
