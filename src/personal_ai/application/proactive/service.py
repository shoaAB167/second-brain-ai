from datetime import datetime, timedelta, timezone
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

from personal_ai.application.memory.quality_service import MemoryQualityService
from personal_ai.core.logger import get_logger
from personal_ai.domain.experience import (
    Experience,
    ExperienceLifecycle,
    ExperienceLifecycleStatus,
    ExperienceType,
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


class ProactiveIntelligenceService:
    """Deterministic, model-agnostic proactive intelligence service for PR #22 & PR #27.

    Philosophy:
        Observation -> Signal -> Supporting Evidence -> Pattern/Context -> Proactive Candidate

    Invariants:
        1. ProactiveCandidate != Action: Produces passive proposal containers only.
           Does NOT execute tools, send notifications, modify memory, or make LLM calls.
        2. Preserves Uncertainty: Explicit emotion > extracted emotion > inferred possibility.
           Never diagnoses conditions, makes causal claims, or turns temporary states into personality traits.
        3. Strict User Isolation: Every analysis is scoped exclusively to authenticated user_id.
        4. Anti-Double-Counting & Deduplication: Deduplicates experiences so identical memories
           never artificially inflate evidence counts.
        5. Conservative Signal Generation: Prefers NO SIGNAL over FALSE POSITIVE when evidence is insufficient.
        6. Pattern Precedence: Patterns are supporting hypotheses, not facts. Current explicit user
           statements strictly supersede contradictory historical patterns.
    """

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

    # Explicit commitment indicator phrases (generic plans like "I plan to" or "I will" are NOT commitments)
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
        """Initialize ProactiveIntelligenceService with optional MemoryQualityService."""
        self._quality_service = memory_quality_service or quality_service or MemoryQualityService()

    def _normalize_user_id(self, user_id: Any) -> Optional[uuid.UUID]:
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

    def _normalize_text(self, text: str) -> str:
        """Normalize text for conservative deduplication comparison."""
        if not text:
            return ""
        cleaned = re.sub(r"[^\w\s]", "", text.lower())
        return re.sub(r"\s+", " ", cleaned).strip()

    def _extract_items(
        self,
        user_id: uuid.UUID,
        context: Optional[Union[PersonalContext, List[Experience], List[PersonalContextItem]]] = None,
        personal_context: Optional[PersonalContext] = None,
        experiences: Optional[List[Experience]] = None,
    ) -> List[Dict[str, Any]]:
        """Normalize and deduplicate observation input sources into user-isolated, anti-double-counted records."""
        raw_records: List[Dict[str, Any]] = []

        # 1. From context argument
        if isinstance(context, PersonalContext):
            personal_context = context
        elif isinstance(context, list):
            for item in context:
                if isinstance(item, Experience):
                    experiences = (experiences or []) + [item]
                elif isinstance(item, PersonalContextItem):
                    norm = self._normalize_context_item(item, user_id)
                    if norm:
                        raw_records.append(norm)

        # 2. From personal_context
        if personal_context and not personal_context.is_empty:
            # Enforce user isolation at the container level
            if self._normalize_user_id(personal_context.user_id) == user_id:
                for ctx_item in personal_context.items:
                    norm = self._normalize_context_item(ctx_item, user_id)
                    if norm:
                        raw_records.append(norm)
            else:
                logger.warning(
                    "User isolation violation prevented: personal_context user_id mismatch [expected=%s, got=%s]",
                    user_id,
                    personal_context.user_id,
                )

        # 3. From experiences list
        if experiences:
            for exp in experiences:
                # Enforce fail-closed user isolation
                exp_user = self._normalize_user_id(exp.user_id)
                if exp_user is None or exp_user != user_id:
                    logger.warning(
                        "User isolation violation prevented: experience user_id mismatch [expected=%s, got=%s]",
                        user_id,
                        exp.user_id,
                    )
                    continue

                # Filter out expired or superseded experiences from active proactive observation
                life_status = (
                    exp.lifecycle_status.value
                    if hasattr(exp.lifecycle_status, "value")
                    else str(exp.lifecycle_status or "ACTIVE").upper()
                )
                if life_status in ("EXPIRED", "SUPERSEDED"):
                    continue

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

                raw_records.append(
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

        # 4. Anti-Double-Counting Deduplication: Collapse duplicate IDs and identical normalized content
        deduped_records: List[Dict[str, Any]] = []
        seen_ids: Set[uuid.UUID] = set()
        seen_content_keys: Set[Tuple[str, str]] = set()

        for rec in raw_records:
            rec_id = rec.get("id")
            if rec_id:
                if rec_id in seen_ids:
                    continue
                seen_ids.add(rec_id)

            norm_content = self._normalize_text(rec.get("content", ""))
            temporal_key = self._normalize_text(rec.get("temporal_context") or "")
            content_key = (norm_content, temporal_key)

            if content_key in seen_content_keys:
                continue
            seen_content_keys.add(content_key)

            deduped_records.append(rec)

        return deduped_records

    def _normalize_context_item(self, item: PersonalContextItem, user_id: uuid.UUID) -> Dict[str, Any]:
        """Convert a PersonalContextItem into a standardized observation dict."""
        emotion_val = None
        if item.emotional_context and isinstance(item.emotional_context, dict):
            emotion_val = item.emotional_context.get("emotion")

        created = item.created_at if isinstance(item.created_at, datetime) else None
        if created and created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)

        return {
            "id": item.experience_id,
            "content": item.content,
            "type": (item.type or "").upper(),
            "domain": item.domain,
            "lifecycle_status": (item.lifecycle_status or "ACTIVE").upper(),
            "temporal_context": item.temporal_context,
            "emotion": emotion_val,
            "created_at": created,
        }

    def _has_meaningful_goal_activity(
        self,
        goal_content: str,
        goal_id: uuid.UUID,
        records: List[Dict[str, Any]],
        current_message: Optional[str],
        now: datetime,
    ) -> bool:
        """Check whether there is strong, non-generic evidence of recent progress or activity toward a goal.

        Conservative rule:
        - Weak or single generic word overlap (e.g. 'AI') does NOT count as activity.
        - Meaningful activity requires significant multi-word phrase matching or explicit activity records.
        - Messages or records containing explicit inactivity phrases do NOT count as positive activity.
        """
        # Extract meaningful multi-word key phrases (length >= 4 chars, excluding stopwords)
        stop_words = {
            "want", "goal", "need", "like", "will", "make", "this", "that", "with", "from",
            "have", "been", "about", "project", "work", "more", "some", "into", "their"
        }
        words = [w for w in re.findall(r"\b\w{4,}\b", goal_content.lower()) if w not in stop_words]
        if not words:
            return False

        # 1. Check current user message
        if current_message:
            msg_lower = current_message.lower()
            # If current message contains explicit inactivity phrases, it is NOT positive progress
            has_inactivity = any(phrase in msg_lower for phrase in self._EXPLICIT_INACTIVITY_PHRASES)
            if not has_inactivity:
                matched_words = [w for w in words if w in msg_lower]
                if len(matched_words) >= 2 or (len(words) == 1 and words[0] in msg_lower and len(words[0]) >= 6):
                    return True

        # 2. Check recent experience records
        for other in records:
            if other["id"] == goal_id:
                continue

            other_type = other.get("type", "")
            # Only action/progress experience types count toward activity
            if other_type in ("EVENT", "PROJECT", "HABIT", "DECISION", "STATE"):
                other_content = other["content"].lower()
                # Ensure the other record doesn't indicate inactivity
                if any(phrase in other_content for phrase in self._EXPLICIT_INACTIVITY_PHRASES):
                    continue
                matched_words = [w for w in words if w in other_content]
                # Require strong topical overlap (at least 2 meaningful words)
                if len(matched_words) >= 2:
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
        """Detect potential goal inactivity signals based strictly on explicit inactivity evidence.

        Conservative Invariant & Priority Order:
        1. Detect explicit inactivity evidence relevant to the goal FIRST.
        2. If explicit inactivity is present -> generate GOAL_INACTIVITY signal.
        3. Only when explicit inactivity is NOT present should meaningful recent activity be considered.
        4. Goal age alone is NEVER sufficient evidence of inactivity.
        5. If evidence is insufficient, prefer NO SIGNAL.
        """
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

            stop_words = {
                "want", "goal", "need", "like", "will", "make", "this", "that", "with", "from",
                "have", "been", "about", "project", "work", "more", "some", "into", "their"
            }
            goal_words = [w for w in re.findall(r"\b\w{4,}\b", goal_content.lower()) if w not in stop_words]

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
                        # Check if message is relevant to this goal
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

            # 3. Only when explicit inactivity is NOT present, check for meaningful recent activity
            if self._has_meaningful_goal_activity(goal_content, goal_id, records, current_message, now):
                # Active goal: do not generate inactivity signal
                continue

            # 4. If neither explicit inactivity nor meaningful activity: NO SIGNAL

        return signals

    def _detect_commitment_missed_signals(
        self,
        records: List[Dict[str, Any]],
        current_message: Optional[str],
        now: datetime,
    ) -> List[ProactiveSignal]:
        """Detect missed commitment signals conservatively based on explicit commitment and temporal evidence.

        Conservative Invariant (All 3 conditions required):
        1. Strong commitment evidence (ExperienceType.COMMITMENT or explicit commitment phrasing).
        2. Temporal evidence that the deadline/timeframe is in the past.
        3. Evidence that the expected action was missed.
        If any one of these 3 conditions is missing -> NO SIGNAL.
        """
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
                # Generic plan, intention, or goal without explicit commitment -> NO SIGNAL
                continue

            commitment_id = rec["id"]

            # 2. Require evidence that the deadline/timeframe has passed
            is_past_deadline = False
            past_indicators = ["yesterday", "last week", "last night", "passed deadline", "overdue", "missed date"]
            if any(p in temporal_ctx for p in past_indicators) or any(p in content_lower for p in past_indicators):
                is_past_deadline = True

            # If the temporal context indicates a future date/time without past indicators -> NOT missed
            future_indicators = ["tomorrow", "next week", "next month", "tonight", "in the future", "upcoming"]
            if any(f in temporal_ctx for f in future_indicators):
                is_past_deadline = False

            # 3. Require evidence that the expected action did not happen
            has_missed_evidence = False
            if any(re.search(pat, content_lower) for pat in self._MISSED_INDICATORS) or any(
                re.search(pat, temporal_ctx) for pat in self._MISSED_INDICATORS
            ):
                has_missed_evidence = True
            elif msg_lower and any(re.search(pat, msg_lower) for pat in self._MISSED_INDICATORS):
                # Ensure current message refers to this commitment topic
                keywords = [w for w in re.findall(r"\b\w{4,}\b", content_lower)]
                if any(kw in msg_lower for kw in keywords):
                    has_missed_evidence = True

            # All 3 conditions must be strictly met: commitment + past deadline + missed evidence
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
        """Detect repeated state signals across distinct experiences with strictly observational semantics.

        Conservative Invariant:
        - Single state occurrence -> NO SIGNAL.
        - Repeated occurrences (count >= 2) within recent observations -> REPEATED_STATE signal.
        - Reason is strictly observational: "A similar state was recorded more than once recently."
        - Strictly NO medical diagnosis, NO causal claims, NO personality trait inference.
        """
        signals: List[ProactiveSignal] = []
        state_clusters: Dict[str, List[uuid.UUID]] = {}

        for rec in records:
            content_lower = rec["content"].lower()
            emotion = (rec.get("emotion") or "").lower()
            rec_id = rec["id"]
            rec_created = rec.get("created_at")

            # Check if record is within recent window (e.g. 30 days) if created_at is present
            if rec_created and (now - rec_created) > timedelta(days=30):
                continue

            detected_cluster = None

            # 1. Check explicit emotion field
            if emotion:
                for cluster_name, keywords in self._STATE_KEYWORDS.items():
                    if emotion == cluster_name or any(kw in emotion for kw in keywords):
                        detected_cluster = cluster_name
                        break

            # 2. Check content for state patterns if type is STATE or EMOTION_STATE or emotion is present
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

        # Evaluate repetition (count >= 2 distinct experiences)
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

    def _extract_patterns(
        self,
        user_id: uuid.UUID,
        context: Optional[Union[PersonalContext, List[Any]]] = None,
        personal_context: Optional[PersonalContext] = None,
        patterns: Optional[List[Union[PersonalPattern, PersonalPatternContextItem]]] = None,
    ) -> List[Dict[str, Any]]:
        """Normalize and isolate active PersonalPattern hypotheses for the authenticated user."""
        raw_patterns: List[Dict[str, Any]] = []

        # 1. From context if PersonalContext
        if isinstance(context, PersonalContext):
            personal_context = context

        # 2. From personal_context.patterns
        if personal_context and getattr(personal_context, "patterns", None):
            if self._normalize_user_id(personal_context.user_id) == user_id:
                for pat in personal_context.patterns:
                    norm = self._normalize_pattern_item(pat, user_id)
                    if norm:
                        raw_patterns.append(norm)
            else:
                logger.warning(
                    "User isolation violation prevented: personal_context patterns user_id mismatch [expected=%s, got=%s]",
                    user_id,
                    personal_context.user_id,
                )

        # 3. From explicit patterns list
        if patterns:
            for pat in patterns:
                norm = self._normalize_pattern_item(pat, user_id)
                if norm:
                    raw_patterns.append(norm)

        # 4. Deduplicate active patterns
        deduped_patterns: List[Dict[str, Any]] = []
        seen_pat_ids: Set[uuid.UUID] = set()
        seen_pat_descs: Set[str] = set()

        for pat_dict in raw_patterns:
            pid = pat_dict.get("id")
            if pid:
                if pid in seen_pat_ids:
                    continue
                seen_pat_ids.add(pid)

            norm_desc = self._normalize_text(pat_dict.get("description", ""))
            if norm_desc in seen_pat_descs:
                continue
            seen_pat_descs.add(norm_desc)

            deduped_patterns.append(pat_dict)

        return deduped_patterns

    def _normalize_pattern_item(
        self,
        pat: Union[PersonalPattern, PersonalPatternContextItem, Dict[str, Any]],
        user_id: uuid.UUID,
    ) -> Optional[Dict[str, Any]]:
        """Validate and normalize a PersonalPattern into a user-isolated dictionary."""
        if isinstance(pat, PersonalPattern):
            # Strict fail-closed user isolation
            pat_user = self._normalize_user_id(pat.user_id)
            if pat_user is None or pat_user != user_id:
                logger.warning("Dropped pattern due to user isolation mismatch or missing user_id.")
                return None

            status_val = pat.status.value if hasattr(pat.status, "value") else str(pat.status).upper()
            if status_val not in (PatternStatus.HYPOTHESIS.value, PatternStatus.CONFIRMED.value):
                return None

            return {
                "id": pat.id,
                "description": pat.description,
                "domain": pat.domain.value if hasattr(pat.domain, "value") else str(pat.domain),
                "confidence": pat.confidence,
                "status": status_val,
                "evidence_count": len(pat.evidence_ids) if pat.evidence_ids else 0,
            }

        elif isinstance(pat, PersonalPatternContextItem):
            status_val = (pat.status or "HYPOTHESIS").upper()
            if status_val not in (PatternStatus.HYPOTHESIS.value, PatternStatus.CONFIRMED.value):
                return None

            return {
                "id": pat.pattern_id,
                "description": pat.description,
                "domain": pat.domain,
                "confidence": pat.confidence,
                "status": status_val,
                "evidence_count": pat.evidence_count,
            }

        elif isinstance(pat, dict):
            status_val = str(pat.get("status", "HYPOTHESIS")).upper()
            if status_val not in (PatternStatus.HYPOTHESIS.value, PatternStatus.CONFIRMED.value):
                return None

            return {
                "id": pat.get("id") or pat.get("pattern_id"),
                "description": pat.get("description", ""),
                "domain": pat.get("domain", "GENERAL"),
                "confidence": float(pat.get("confidence", 0.70)),
                "status": status_val,
                "evidence_count": int(pat.get("evidence_count", 1)),
            }

        return None

    def _find_supporting_pattern(
        self,
        signal: ProactiveSignal,
        patterns: List[Dict[str, Any]],
        current_message: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Find relevant active supporting pattern and check for explicit statement contradiction.

        Precedence Invariant:
        - If current_message contradicts the pattern (e.g. pattern says 'studies at night', but user says
          'switched to morning now'), the pattern is suppressed and discarded from proactive suggestions.
        """
        if not patterns:
            return None

        msg_lower = (current_message or "").lower()

        for pat in patterns:
            pat_desc = pat.get("description", "")
            pat_desc_lower = pat_desc.lower()
            pat_domain = (pat.get("domain") or "").lower()

            # Check relevance to signal
            is_relevant = False
            if signal.type == ProactiveSignalType.GOAL_INACTIVITY:
                if pat_domain in ("career", "projects", "learning", "work", "fitness") or any(
                    kw in pat_desc_lower for kw in ["work", "code", "study", "project", "night", "morning", "habit", "routine", "exercise", "run"]
                ):
                    is_relevant = True
            elif signal.type == ProactiveSignalType.COMMITMENT_MISSED:
                if pat_domain in ("career", "projects", "learning", "work", "general") or any(
                    kw in pat_desc_lower for kw in ["schedule", "deadline", "time", "focus", "work", "plan"]
                ):
                    is_relevant = True
            elif signal.type == ProactiveSignalType.REPEATED_STATE:
                if pat_domain in ("health", "fitness", "relationships", "social", "general") or any(
                    kw in pat_desc_lower for kw in ["tired", "energy", "stress", "sleep", "rest", "walk"]
                ):
                    is_relevant = True

            if not is_relevant:
                continue

            # Check Current User Statement Precedence / Contradiction:
            # If current message states a switch or contradiction (e.g., 'morning' vs 'night', 'stopped', 'no longer')
            if msg_lower:
                if "night" in pat_desc_lower and any(
                    m in msg_lower for m in ["morning now", "switched to morning", "study in the morning", "work in the morning", "now in the morning"]
                ):
                    logger.debug("Suppressed historical pattern '%s' due to explicit user contradiction.", pat_desc)
                    continue
                if "morning" in pat_desc_lower and any(
                    m in msg_lower for m in ["night now", "switched to night", "study at night", "work at night", "now at night"]
                ):
                    logger.debug("Suppressed historical pattern '%s' due to explicit user contradiction.", pat_desc)
                    continue
                if any(neg in msg_lower for neg in ["no longer", "stopped", "changed my routine", "switched"]):
                    pat_keywords = [w for w in re.findall(r"\b\w{4,}\b", pat_desc_lower)]
                    if any(kw in msg_lower for kw in pat_keywords):
                        logger.debug("Suppressed historical pattern '%s' due to explicit user routine change.", pat_desc)
                        continue

            return pat

        return None

    def detect_signals(
        self,
        records: List[Dict[str, Any]],
        current_message: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> List[ProactiveSignal]:
        """Detect observation signals using the intermediate ProactiveSignal layer.

        Args:
            records: List of normalized, user-isolated observation records.
            current_message: Optional current user message.
            now: Current reference UTC datetime.

        Returns:
            List[ProactiveSignal]: Detected observation signals.
        """
        ref_time = now or _utc_now()
        if ref_time.tzinfo is None:
            ref_time = ref_time.replace(tzinfo=timezone.utc)

        signals: List[ProactiveSignal] = []

        # 1. Goal Inactivity Signals
        signals.extend(self._detect_goal_inactivity_signals(records, current_message, ref_time))

        # 2. Missed Commitment Signals
        signals.extend(self._detect_commitment_missed_signals(records, current_message, ref_time))

        # 3. Repeated State Signals
        signals.extend(self._detect_repeated_state_signals(records, ref_time))

        return signals

    def generate_candidates(
        self,
        signals: List[ProactiveSignal],
        patterns: Optional[List[Dict[str, Any]]] = None,
        current_message: Optional[str] = None,
    ) -> List[ProactiveCandidate]:
        """Convert detected ProactiveSignal objects into evidence-evaluated ProactiveCandidate proposals.

        Args:
            signals: List of detected ProactiveSignal objects.
            patterns: Optional list of active, user-isolated supporting patterns.
            current_message: Optional current user message.

        Returns:
            List[ProactiveCandidate]: Formatted candidate proposals with evidence metadata.
        """
        candidates: List[ProactiveCandidate] = []
        patterns_list = patterns or []

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
                "Ask how the user is feeling and whether they want help understanding the pattern.",
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

            # Evidence evaluation and pattern support enrichment
            supporting_pattern = self._find_supporting_pattern(signal, patterns_list, current_message)
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

                # Bounded deterministic confidence bump (max 0.86)
                confidence = round(min(confidence + min(0.06 * pat_conf, 0.06), 0.86), 4)

                # Enrich suggested action if applicable
                if signal.type == ProactiveSignalType.GOAL_INACTIVITY:
                    if "night" in pat_desc.lower():
                        action = "Since you usually work on AI projects at night, ask whether you'd like to work on it tonight."
                    else:
                        action = f"Based on your observed routine ('{pat_desc}'), ask whether you want to schedule time for this goal."
                elif signal.type == ProactiveSignalType.COMMITMENT_MISSED:
                    action = f"Check in on this commitment, considering your typical workflow hypothesis ('{pat_desc}')."

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
            # Key based on signal type and sorted related experience IDs (or reason)
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
        """Analyze user-isolated observations and detect meaningful proactive intervention candidates.

        Pipeline:
            Observations + Patterns -> detect_signals() -> generate_candidates() -> deduplicate -> List[ProactiveCandidate]

        Args:
            user_id: Authenticated user UUID for strict isolation.
            context: Context container (PersonalContext or list of Experiences).
            personal_context: Optional explicit PersonalContext container.
            experiences: Optional explicit list of Experience entities.
            patterns: Optional explicit list of PersonalPattern or PersonalPatternContextItem entities.
            current_message: Optional current user message / query.
            reference_time: Optional reference UTC timestamp (defaults to current UTC time).

        Returns:
            List[ProactiveCandidate]: Bounded list of proposed intervention candidates (empty if nothing meaningful).
        """
        if not user_id:
            logger.warning("ProactiveIntelligenceService.analyze called without authenticated user_id.")
            return []

        now = reference_time or _utc_now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        # 1. Normalize, isolate, and deduplicate records for authenticated user
        records = self._extract_items(
            user_id=user_id,
            context=context,
            personal_context=personal_context,
            experiences=experiences,
        )

        if not records:
            return []

        # 2. Extract and isolate active patterns for authenticated user
        pattern_records = self._extract_patterns(
            user_id=user_id,
            context=context,
            personal_context=personal_context,
            patterns=patterns,
        )

        logger.info(
            "Analyzing proactive intelligence signals [user_id=%s, observations_count=%d, patterns_count=%d]",
            user_id,
            len(records),
            len(pattern_records),
        )

        # 3. Intermediate layer: detect signals
        signals = self.detect_signals(
            records=records,
            current_message=current_message,
            now=now,
        )

        # 4. Evidence evaluation and candidate generation
        candidates = self.generate_candidates(
            signals=signals,
            patterns=pattern_records,
            current_message=current_message,
        )

        # 5. In-memory per-analysis deduplication
        deduped = self._deduplicate_candidates(candidates)

        logger.info(
            "Proactive intelligence analysis complete [user_id=%s, signals_count=%d, candidates_count=%d]",
            user_id,
            len(signals),
            len(deduped),
        )

        return deduped

