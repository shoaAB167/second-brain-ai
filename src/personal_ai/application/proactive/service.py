from datetime import datetime, timedelta, timezone
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

from personal_ai.core.logger import get_logger
from personal_ai.domain.experience import (
    Experience,
    ExperienceLifecycleStatus,
    ExperienceType,
    PersonalContext,
    PersonalContextItem,
)
from personal_ai.domain.proactive.enums import ProactivePriority, ProactiveSignalType
from personal_ai.domain.proactive.models import ProactiveCandidate, ProactiveSignal

logger = get_logger(__name__)


def _utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


class ProactiveIntelligenceService:
    """Deterministic, model-agnostic proactive intelligence service for PR #22.

    Philosophy:
        Observation -> Signal -> Hypothesis -> Possible Intervention

    Invariants:
        1. ProactiveCandidate != Action: Produces passive proposal containers only.
           Does NOT execute tools, send notifications, modify memory, or make LLM calls.
        2. Preserves Uncertainty: Explicit emotion > extracted emotion > inferred possibility.
           Never diagnoses conditions or turns temporary states into personality traits.
        3. Strict User Isolation: Every analysis is scoped exclusively to authenticated user_id.
        4. In-Memory Deduplication: Avoids duplicate candidates for identical evidence in a single pass.
    """

    # Emotional/Mental state keywords for repeated state detection
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

    # Commitment indicator keywords
    _COMMITMENT_INDICATORS = [
        r"\b(committed to|promised to|agreed to|scheduled to)\b",
        r"\b(will work on|plan to complete|deadline is|due on|due by|finish by)\b",
        r"\b(every day|daily goal|every morning|every evening)\b",
    ]

    # Missed commitment evidence keywords
    _MISSED_INDICATORS = [
        r"\b(didn'?t|did not|couldn'?t|could not|failed to|missed|forgot to|haven'?t|have not)\b",
        r"\b(fell behind|delayed|unable to finish|skipped|postponed)\b",
    ]

    def __init__(self) -> None:
        """Initialize ProactiveIntelligenceService with zero external side effects."""
        pass

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

    def _extract_items(
        self,
        user_id: uuid.UUID,
        context: Optional[Union[PersonalContext, List[Experience], List[PersonalContextItem]]] = None,
        personal_context: Optional[PersonalContext] = None,
        experiences: Optional[List[Experience]] = None,
    ) -> List[Dict[str, Any]]:
        """Normalize various observation input sources into standardized, user-isolated records."""
        records: List[Dict[str, Any]] = []

        # 1. From context argument
        if isinstance(context, PersonalContext):
            personal_context = context
        elif isinstance(context, list):
            for item in context:
                if isinstance(item, Experience):
                    experiences = (experiences or []) + [item]
                elif isinstance(item, PersonalContextItem):
                    records.append(self._normalize_context_item(item, user_id))

        # 2. From personal_context
        if personal_context and not personal_context.is_empty:
            # Enforce user isolation at the container level
            if personal_context.user_id == user_id:
                for ctx_item in personal_context.items:
                    records.append(self._normalize_context_item(ctx_item, user_id))
            else:
                logger.warning(
                    "User isolation violation prevented: personal_context user_id mismatch [expected=%s, got=%s]",
                    user_id,
                    personal_context.user_id,
                )

        # 3. From experiences list
        if experiences:
            for exp in experiences:
                # Enforce user isolation
                exp_user = self._normalize_user_id(exp.user_id)
                if exp_user is not None and exp_user != user_id:
                    logger.warning(
                        "User isolation violation prevented: experience user_id mismatch [expected=%s, got=%s]",
                        user_id,
                        exp.user_id,
                    )
                    continue

                emotion_val = None
                if exp.emotional_context:
                    emotion_val = exp.emotional_context.emotion

                created = exp.created_at if isinstance(exp.created_at, datetime) else None
                if created and created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)

                type_str = exp.type.value if hasattr(exp.type, "value") else str(exp.type or "")
                lifecycle_status_str = (
                    exp.lifecycle_status.value
                    if hasattr(exp.lifecycle_status, "value")
                    else str(exp.lifecycle_status or "ACTIVE")
                )

                records.append(
                    {
                        "id": exp.id,
                        "content": exp.content,
                        "type": type_str.upper(),
                        "domain": exp.domain,
                        "lifecycle_status": lifecycle_status_str.upper(),
                        "temporal_context": exp.temporal_context,
                        "emotion": emotion_val,
                        "created_at": created,
                    }
                )

        return records

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

    def _detect_goal_inactivity(
        self,
        records: List[Dict[str, Any]],
        current_message: Optional[str],
        now: datetime,
    ) -> List[ProactiveCandidate]:
        """Detect potential goal inactivity while preserving uncertainty.

        Rules:
        - Must have an ACTIVE GOAL experience.
        - Must have evidence of inactivity (e.g. aged > 14 days without activity, or explicit temporal inactivity notes).
        - Must NOT produce a signal if the goal is active, newly created, or has recent related activity.
        - Must NEVER infer lack of motivation, laziness, or emotional state.
        """
        candidates: List[ProactiveCandidate] = []
        msg_lower = (current_message or "").lower()

        for rec in records:
            if rec["type"] != "GOAL" and rec.get("domain") != "goal":
                continue

            if rec["lifecycle_status"] in ("EXPIRED", "SUPERSEDED"):
                continue

            goal_content = rec["content"]
            goal_id = rec["id"]
            created_at = rec.get("created_at")
            temporal_ctx = (rec.get("temporal_context") or "").lower()

            # Check if there is active evidence in current message or recent records
            goal_keywords = [
                w for w in re.findall(r"\w+", goal_content.lower())
                if len(w) > 3 and w not in ("want", "goal", "need", "like", "will", "make", "this", "that")
            ]

            has_recent_activity = False
            # 1. Check current message
            if goal_keywords and any(kw in msg_lower for kw in goal_keywords):
                has_recent_activity = True

            # 2. Check other recent records for progress or activity
            if not has_recent_activity:
                for other in records:
                    if other["id"] == goal_id:
                        continue
                    other_content = other["content"].lower()
                    if goal_keywords and any(kw in other_content for kw in goal_keywords):
                        other_created = other.get("created_at")
                        if other_created and (now - other_created) < timedelta(days=7):
                            has_recent_activity = True
                            break

            if has_recent_activity:
                # Active goal: do not generate inactivity signal
                continue

            # Check for evidence of inactivity
            has_inactivity_evidence = False
            confidence = 0.72

            # Temporal context hints
            inactivity_phrases = [
                "inactive", "haven't touched", "no progress", "on hold",
                "last month", "weeks ago", "paused", "not worked on"
            ]
            if any(phrase in temporal_ctx for phrase in inactivity_phrases):
                has_inactivity_evidence = True
                confidence = 0.78
            elif created_at is not None:
                age = now - created_at
                if age >= timedelta(days=14):
                    has_inactivity_evidence = True
                    confidence = 0.72 if age < timedelta(days=30) else 0.82

            if has_inactivity_evidence:
                candidates.append(
                    ProactiveCandidate(
                        signal_type=ProactiveSignalType.GOAL_INACTIVITY,
                        reason="The goal appears inactive based on recent available context.",
                        confidence=confidence,
                        priority=ProactivePriority.LOW,
                        suggested_action="Ask whether the user wants to revisit or work on this goal.",
                        related_experience_ids=[goal_id],
                    )
                )

        return candidates

    def _detect_commitment_missed(
        self,
        records: List[Dict[str, Any]],
        current_message: Optional[str],
        now: datetime,
    ) -> List[ProactiveCandidate]:
        """Detect missed commitments conservatively based on temporal evidence.

        Rules:
        - Must have clear evidence of a commitment (not just a vague intention).
        - Must have temporal evidence that the commitment window passed without expected action.
        - If insufficient evidence, return NO signal.
        """
        candidates: List[ProactiveCandidate] = []
        msg_lower = (current_message or "").lower()

        for rec in records:
            content_lower = rec["content"].lower()
            temporal_ctx = (rec.get("temporal_context") or "").lower()

            # Check if this item represents a commitment
            is_commitment = any(re.search(pat, content_lower) for pat in self._COMMITMENT_INDICATORS)
            if not is_commitment:
                continue

            commitment_id = rec["id"]

            # Check for evidence that the commitment was missed
            has_missed_evidence = False
            confidence = 0.75

            # 1. Missed indicators in content or temporal context
            if any(re.search(pat, content_lower) for pat in self._MISSED_INDICATORS) or any(
                re.search(pat, temporal_ctx) for pat in self._MISSED_INDICATORS
            ):
                has_missed_evidence = True
                confidence = 0.80

            # 2. Missed indicators in current user message regarding this commitment
            elif any(re.search(pat, msg_lower) for pat in self._MISSED_INDICATORS):
                # Ensure current message refers to this commitment topic
                keywords = [w for w in re.findall(r"\w+", content_lower) if len(w) > 3]
                if any(kw in msg_lower for kw in keywords):
                    has_missed_evidence = True
                    confidence = 0.82

            # 3. Explicit past deadline in temporal context
            elif any(p in temporal_ctx for p in ["yesterday", "last week", "passed deadline", "overdue", "missed date"]):
                has_missed_evidence = True
                confidence = 0.75

            if has_missed_evidence:
                candidates.append(
                    ProactiveCandidate(
                        signal_type=ProactiveSignalType.COMMITMENT_MISSED,
                        reason="Evidence indicates a planned commitment may have been missed based on available temporal context.",
                        confidence=confidence,
                        priority=ProactivePriority.MEDIUM,
                        suggested_action="Ask whether the user wants to reschedule, adjust, or check in on this commitment.",
                        related_experience_ids=[commitment_id],
                    )
                )

        return candidates

    def _detect_repeated_state(
        self,
        records: List[Dict[str, Any]],
    ) -> List[ProactiveCandidate]:
        """Detect repeated states across distinct experiences while preserving uncertainty.

        Rules:
        - Single temporary state -> NO signal.
        - Repeated evidence (count >= 2) -> REPEATED_STATE signal.
        - NEVER diagnose medical/psychological conditions.
        - NEVER turn temporary emotions into permanent personality traits.
        """
        candidates: List[ProactiveCandidate] = []
        state_clusters: Dict[str, List[uuid.UUID]] = {}

        for rec in records:
            content_lower = rec["content"].lower()
            emotion = (rec.get("emotion") or "").lower()
            rec_id = rec["id"]

            detected_cluster = None

            # 1. Check explicit emotion field
            if emotion:
                for cluster_name, keywords in self._STATE_KEYWORDS.items():
                    if emotion == cluster_name or any(kw in emotion for kw in keywords):
                        detected_cluster = cluster_name
                        break

            # 2. Check content for state patterns if type is STATE or EMOTION_STATE or not matched yet
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

        # Evaluate repetition (count >= 2)
        for cluster_name, exp_ids in state_clusters.items():
            if len(exp_ids) >= 2:
                count = len(exp_ids)
                confidence = round(min(0.70 + (count - 2) * 0.08, 0.90), 2)
                candidates.append(
                    ProactiveCandidate(
                        signal_type=ProactiveSignalType.REPEATED_STATE,
                        reason=f"A similar state ('{cluster_name}') has appeared repeatedly in recent experiences.",
                        confidence=confidence,
                        priority=ProactivePriority.MEDIUM,
                        suggested_action="Ask how the user is feeling and whether they want help understanding the pattern.",
                        related_experience_ids=exp_ids,
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
        current_message: Optional[str] = None,
        reference_time: Optional[datetime] = None,
    ) -> List[ProactiveCandidate]:
        """Analyze user-isolated observations and detect meaningful proactive intervention candidates.

        Args:
            user_id: Authenticated user UUID for strict isolation.
            context: Context container (PersonalContext or list of Experiences).
            personal_context: Optional explicit PersonalContext container.
            experiences: Optional explicit list of Experience entities.
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

        # 1. Normalize and isolate records for authenticated user
        records = self._extract_items(
            user_id=user_id,
            context=context,
            personal_context=personal_context,
            experiences=experiences,
        )

        if not records:
            return []

        logger.info(
            "Analyzing proactive intelligence signals [user_id=%s, observations_count=%d]",
            user_id,
            len(records),
        )

        candidates: List[ProactiveCandidate] = []

        # 2. Detect Goal Inactivity
        goal_candidates = self._detect_goal_inactivity(
            records=records,
            current_message=current_message,
            now=now,
        )
        candidates.extend(goal_candidates)

        # 3. Detect Missed Commitments
        commitment_candidates = self._detect_commitment_missed(
            records=records,
            current_message=current_message,
            now=now,
        )
        candidates.extend(commitment_candidates)

        # 4. Detect Repeated State
        state_candidates = self._detect_repeated_state(records=records)
        candidates.extend(state_candidates)

        # 5. In-memory per-analysis deduplication
        deduped = self._deduplicate_candidates(candidates)

        logger.info(
            "Proactive intelligence analysis complete [user_id=%s, candidates_count=%d]",
            user_id,
            len(deduped),
        )

        return deduped
