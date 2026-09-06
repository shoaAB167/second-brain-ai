from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

from personal_ai.application.memory.dimension_analyzer import QueryDimensionAnalyzer
from personal_ai.config.settings import get_settings
from personal_ai.core.logger import get_logger
from personal_ai.domain.experience import (
    Experience,
    PersonalContextItem,
    PersonalPatternContextItem,
    RetrievalDimension,
)
from personal_ai.domain.experience.enums import (
    ExperienceEvidenceLevel,
    ExperienceImportance,
    ExperienceLifecycle,
    ExperienceLifecycleStatus,
)
from personal_ai.domain.pattern.enums import PatternStatus
from personal_ai.llm.models import LLMMessage

logger = get_logger(__name__)


def _normalize_user_id(uid: Any) -> Optional[str]:
    """Normalize user_id to a standardized string representation.

    Fails closed (returns None) on missing, empty, or malformed user IDs.

    Args:
        uid: Candidate user ID value (UUID, str, or None).

    Returns:
        Optional[str]: Normalized UUID string or None if invalid.
    """
    if uid is None:
        return None
    if isinstance(uid, uuid.UUID):
        return str(uid)
    if isinstance(uid, str):
        cleaned = uid.strip()
        if not cleaned:
            return None
        try:
            return str(uuid.UUID(cleaned))
        except (ValueError, AttributeError):
            return None
    return None


class MemoryQualityService:
    """Deterministic, auditable quality and selection layer for retrieved memories and patterns.

    Sits between candidate retrieval and final PersonalContext assembly:
    - Candidate validation and fail-closed user isolation
    - Conservative duplicate reduction (exact ID and content near-duplicates with temporal preservation)
    - Lifecycle-aware filtering and contradiction handling
    - Durability-aware freshness evaluation (permanent facts never decay; temporary states decay)
    - Multi-signal deterministic ranking with relevance dominance
    - Bounded context budget enforcement
    """

    def __init__(
        self,
        dimension_analyzer: Optional[QueryDimensionAnalyzer] = None,
    ) -> None:
        """Initialize MemoryQualityService.

        Args:
            dimension_analyzer: QueryDimensionAnalyzer for dimension and temporal extraction.
        """
        self._dimension_analyzer = dimension_analyzer or QueryDimensionAnalyzer()

    def filter_and_deduplicate_experiences(
        self,
        user_id: uuid.UUID,
        experiences: List[Experience],
        is_historical: bool = False,
    ) -> List[Experience]:
        """Validate, filter lifecycle, and conservatively deduplicate experiences for an authenticated user.

        Reuses canonical PR26 behavior:
        - Candidate validation & fail-closed user isolation
        - Lifecycle filtering (excludes EXPIRED/SUPERSEDED unless is_historical=True)
        - Conservative deduplication with temporal variant preservation (and exact ID collapsing)
        """
        target_user_str = _normalize_user_id(user_id)
        if not target_user_str:
            logger.warning("MemoryQualityService.filter_and_deduplicate_experiences called with invalid user_id: %s", user_id)
            return []

        candidates = [(exp, 1.0) for exp in experiences if isinstance(exp, Experience)]
        valid = self._validate_experience_candidates(candidates, target_user_str)
        lifecycle_filtered = self._filter_experience_lifecycle(valid, is_historical=is_historical)
        deduped = self._deduplicate_experiences(lifecycle_filtered)
        return [exp for exp, _ in deduped]

    def filter_and_deduplicate_patterns(
        self,
        user_id: uuid.UUID,
        patterns: List[Any],
    ) -> List[Any]:
        """Validate, filter lifecycle (HYPOTHESIS/CONFIRMED), and deduplicate patterns for an authenticated user.

        Reuses canonical PR26 behavior:
        - Pattern validation & fail-closed user isolation
        - Pattern lifecycle filtering (HYPOTHESIS and CONFIRMED only)
        - Pattern deduplication by ID and normalized description
        """
        target_user_str = _normalize_user_id(user_id)
        if not target_user_str:
            logger.warning("MemoryQualityService.filter_and_deduplicate_patterns called with invalid user_id: %s", user_id)
            return []

        valid = self._validate_pattern_candidates(patterns, target_user_str)
        lifecycle_filtered = self._filter_pattern_lifecycle(valid)
        return self._deduplicate_patterns(lifecycle_filtered)

    def process_candidates(
        self,
        user_id: uuid.UUID,
        query: str,
        experience_candidates: List[Tuple[Experience, float]],
        pattern_candidates: Optional[List[Any]] = None,
        detected_dimensions: Optional[List[RetrievalDimension]] = None,
        conversation_context: Optional[List[LLMMessage]] = None,
        is_historical: bool = False,
        final_limit: Optional[int] = None,
        pattern_limit: Optional[int] = None,
        min_pattern_query_relevance: Optional[float] = None,
    ) -> Tuple[List[PersonalContextItem], List[PersonalPatternContextItem]]:
        """Process, validate, deduplicate, filter, score, and select context items deterministically.

        Args:
            user_id: Authenticated user UUID for strict isolation.
            query: User's search query or prompt.
            experience_candidates: List of (Experience, similarity_score) tuples from retrieval.
            pattern_candidates: Optional list of active PersonalPattern candidate entities.
            detected_dimensions: List of detected query RetrievalDimension values.
            conversation_context: Optional recent conversation context messages.
            is_historical: Whether historical/superseded memories should be permitted.
            final_limit: Bounded limit for selected episodic memory items.
            pattern_limit: Bounded limit for selected pattern hypotheses.
            min_pattern_query_relevance: Minimum query relevance gate for patterns.

        Returns:
            Tuple[List[PersonalContextItem], List[PersonalPatternContextItem]]:
                Selected bounded memories and selected bounded patterns.
        """
        target_user_str = _normalize_user_id(user_id)
        if not target_user_str:
            logger.warning("MemoryQualityService invoked with invalid or missing user_id: %s", user_id)
            return [], []

        settings = get_settings()
        fin_limit = final_limit or settings.personal_context_final_limit
        pat_limit = pattern_limit or settings.personal_context_pattern_limit
        min_pat_rel = (
            min_pattern_query_relevance
            if min_pattern_query_relevance is not None
            else settings.personal_context_min_pattern_query_relevance
        )

        detected_dims = detected_dimensions or []
        now = datetime.now(timezone.utc)

        try:
            # -------------------------------------------------------------
            # Part 1: Experience Memories Pipeline
            # -------------------------------------------------------------
            # 1. Candidate Validation & Fail-Closed User Isolation
            valid_exp_candidates = self._validate_experience_candidates(
                candidates=experience_candidates,
                target_user_id=target_user_str,
            )

            # 2. Lifecycle & Contradiction Filtering
            lifecycle_filtered_exps = self._filter_experience_lifecycle(
                candidates=valid_exp_candidates,
                is_historical=is_historical,
            )

            # 3. Conservative Duplicate Reduction with Temporal Preservation
            deduplicated_exps = self._deduplicate_experiences(
                candidates=lifecycle_filtered_exps,
            )

            # 4. Multi-Signal Quality Scoring & Ranking
            scored_items = self._score_and_rank_experiences(
                candidates=deduplicated_exps,
                detected_dimensions=detected_dims,
                now=now,
            )

            # 5. Context Budget Bounding
            final_items = scored_items[:fin_limit]

            # -------------------------------------------------------------
            # Part 2: Personal Patterns Pipeline
            # -------------------------------------------------------------
            final_patterns: List[PersonalPatternContextItem] = []
            if pattern_candidates:
                # 1. Pattern Validation & Fail-Closed User Isolation
                valid_pat_candidates = self._validate_pattern_candidates(
                    patterns=pattern_candidates,
                    target_user_id=target_user_str,
                )

                # 2. Pattern Lifecycle Filtering (HYPOTHESIS and CONFIRMED only)
                lifecycle_filtered_pats = self._filter_pattern_lifecycle(
                    patterns=valid_pat_candidates,
                )

                # 3. Pattern Deduplication
                deduplicated_pats = self._deduplicate_patterns(
                    patterns=lifecycle_filtered_pats,
                )

                # 4. Pattern Scoring, Relevance Gating & Ranking
                scored_patterns = self._score_and_rank_patterns(
                    patterns=deduplicated_pats,
                    query=query,
                    detected_dimensions=detected_dims,
                    conversation_context=conversation_context,
                    min_query_relevance=min_pat_rel,
                )

                # 5. Context Budget Bounding
                final_patterns = scored_patterns[:pat_limit]

            logger.info(
                "MemoryQualityService processing complete [user_id=%s, exp_in=%d, exp_selected=%d, pat_in=%d, pat_selected=%d]",
                target_user_str,
                len(experience_candidates),
                len(final_items),
                len(pattern_candidates or []),
                len(final_patterns),
            )

            return final_items, final_patterns

        except Exception as exc:
            logger.error(
                "MemoryQualityService unexpected error during candidate processing for user %s: %s",
                target_user_str,
                exc,
                exc_info=True,
            )
            # Fail-safe: return empty or safe fallback without violating user isolation
            return [], []

    # =========================================================================
    # Experience Memories Processing Helpers
    # =========================================================================

    def _validate_experience_candidates(
        self,
        candidates: List[Tuple[Experience, float]],
        target_user_id: str,
    ) -> List[Tuple[Experience, float]]:
        """Validate candidate memories and enforce fail-closed user isolation.

        Rejects:
        - None / non-Experience items
        - Candidates with missing, invalid, or mismatched user_id
        - Candidates with missing ID or empty content
        """
        valid: List[Tuple[Experience, float]] = []
        for exp, similarity in candidates:
            if not isinstance(exp, Experience):
                logger.warning("Rejected non-Experience candidate object.")
                continue

            if not exp.id or not isinstance(exp.id, uuid.UUID):
                logger.warning("Rejected Experience candidate with missing or invalid ID.")
                continue

            if not exp.content or not exp.content.strip():
                logger.warning("Rejected Experience candidate %s with empty content.", exp.id)
                continue

            # Strict Fail-Closed User Isolation
            candidate_user = _normalize_user_id(exp.user_id)
            if candidate_user is None or candidate_user != target_user_id:
                logger.warning(
                    "SECURITY: Dropped candidate memory due to user isolation violation [candidate_id=%s, candidate_user=%s, target_user=%s]",
                    exp.id,
                    candidate_user,
                    target_user_id,
                )
                continue

            sim_val = max(0.0, min(float(similarity), 1.0))
            valid.append((exp, sim_val))

        return valid

    def _filter_experience_lifecycle(
        self,
        candidates: List[Tuple[Experience, float]],
        is_historical: bool,
    ) -> List[Tuple[Experience, float]]:
        """Filter experience candidates based on lifecycle status and query historical intent.

        Rules:
        - ACTIVE: Always eligible.
        - SUPERSEDED: Excluded in normal queries; included when query is explicitly historical.
        - EXPIRED: Excluded from active context unless query is explicitly historical.
        """
        filtered: List[Tuple[Experience, float]] = []
        for exp, sim in candidates:
            status_val = (
                exp.lifecycle_status.value
                if hasattr(exp.lifecycle_status, "value")
                else str(exp.lifecycle_status or "ACTIVE").upper()
            )

            if status_val == ExperienceLifecycleStatus.EXPIRED.value:
                if not is_historical:
                    logger.debug("Filtered EXPIRED experience %s from active context.", exp.id)
                    continue

            elif status_val == ExperienceLifecycleStatus.SUPERSEDED.value:
                if not is_historical:
                    logger.debug("Filtered SUPERSEDED experience %s from active context.", exp.id)
                    continue

            filtered.append((exp, sim))

        return filtered

    def _deduplicate_experiences(
        self,
        candidates: List[Tuple[Experience, float]],
    ) -> List[Tuple[Experience, float]]:
        """Conservatively deduplicate experiences while preserving meaningful temporal variants.

        Rules:
        1. Exact duplicate Experience IDs are collapsed into one (keeping highest similarity).
        2. Content near-duplicates with identical/empty temporal markers are collapsed into
           the single highest-quality representative candidate.
        3. Candidates with identical semantic core but distinct temporal markers/years
           (e.g., "in 2024" vs "in 2026") are preserved.
        """
        # 1. Deduplicate exact IDs first
        id_map: Dict[uuid.UUID, Tuple[Experience, float]] = {}
        for exp, sim in candidates:
            if exp.id not in id_map or sim > id_map[exp.id][1]:
                id_map[exp.id] = (exp, sim)

        unique_id_candidates = list(id_map.values())

        # 2. Content & Temporal-Aware Deduplication
        deduped: List[Tuple[Experience, float]] = []
        # Group candidates by composite key: (normalized_content, temporal_signature)
        grouped_candidates: Dict[Tuple[str, str], List[Tuple[Experience, float]]] = {}

        for exp, sim in unique_id_candidates:
            norm_content = self._normalize_content(exp.content)
            temporal_sig = self._get_temporal_signature(exp)
            key = (norm_content, temporal_sig)

            if key not in grouped_candidates:
                grouped_candidates[key] = []
            grouped_candidates[key].append((exp, sim))

        # For each group, select the single best representative
        for (norm_content, temporal_sig), group in grouped_candidates.items():
            best_candidate = self._select_best_experience_representative(group)
            deduped.append(best_candidate)

        return deduped

    def _normalize_content(self, text: str) -> str:
        """Normalize text by lowercasing, stripping punctuation, and collapsing whitespace."""
        if not text:
            return ""
        cleaned = re.sub(r"[^\w\s]", "", text.lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _get_temporal_signature(self, exp: Experience) -> str:
        """Extract a deterministic temporal signature for a candidate experience.

        Considers:
        - Explicit 4-digit years (1900-2099) in content
        - Relative temporal keywords (e.g., 'previously', 'currently', 'last year')
        - Explicit temporal_context field
        """
        markers: Set[str] = set()

        if exp.temporal_context and exp.temporal_context.strip():
            markers.add(self._normalize_content(exp.temporal_context))

        content_lower = exp.content.lower()

        # Extract 4-digit years
        years = re.findall(r"\b(19\d\d|20\d\d)\b", content_lower)
        for yr in years:
            markers.add(yr)

        # Extract temporal phrases
        temporal_phrases = [
            "last year",
            "this year",
            "next year",
            "in the past",
            "previously",
            "currently",
            "now",
            "today",
            "yesterday",
        ]
        for phrase in temporal_phrases:
            if phrase in content_lower:
                markers.add(phrase)

        if not markers:
            return "__NO_TEMPORAL_MARKER__"

        return "|".join(sorted(markers))

    def _select_best_experience_representative(
        self,
        group: List[Tuple[Experience, float]],
    ) -> Tuple[Experience, float]:
        """Select the highest quality representative candidate among near-duplicates.

        Prioritizes:
        1. Importance level (HIGH > MEDIUM > LOW)
        2. Evidence level (EXPLICIT_USER > EXTRACTED > INFERRED)
        3. Recency (created_at timestamp)
        4. Vector similarity
        """
        importance_weights = {
            ExperienceImportance.HIGH.value: 3,
            ExperienceImportance.MEDIUM.value: 2,
            ExperienceImportance.LOW.value: 1,
        }
        evidence_weights = {
            ExperienceEvidenceLevel.EXPLICIT_USER.value: 3,
            ExperienceEvidenceLevel.EXTRACTED.value: 2,
            ExperienceEvidenceLevel.INFERRED.value: 1,
        }

        def rank_key(item: Tuple[Experience, float]) -> Tuple[int, int, float, float]:
            exp, sim = item
            imp_str = exp.importance.value if hasattr(exp.importance, "value") else str(exp.importance or "MEDIUM").upper()
            ev_str = exp.evidence_level.value if hasattr(exp.evidence_level, "value") else str(exp.evidence_level or "EXTRACTED").upper()

            imp_score = importance_weights.get(imp_str, 2)
            ev_score = evidence_weights.get(ev_str, 2)
            created_ts = exp.created_at.timestamp() if exp.created_at else 0.0

            return (imp_score, ev_score, created_ts, sim)

        return max(group, key=rank_key)

    def _calculate_freshness_factor(
        self,
        exp: Experience,
        now: datetime,
    ) -> float:
        """Calculate durability-aware freshness factor for an experience.

        Rules:
        - STABLE (permanent facts like name, identity, core preferences):
          Freshness is 1.0 regardless of age. Age alone does NOT degrade permanent facts.
        - TEMPORARY / TIME_BOUND (transient states):
          Decays with age:
            age <= 7 days: 1.0
            age <= 30 days: 0.7
            age <= 90 days: 0.4
            age > 90 days: 0.2
        - RECURRING (habits, recurring routines):
          Mild decay:
            age <= 30 days: 1.0
            age <= 90 days: 0.8
            age > 90 days: 0.6
        """
        lifecycle_val = (
            exp.lifecycle.value
            if hasattr(exp.lifecycle, "value")
            else str(exp.lifecycle or "STABLE").upper()
        )

        if lifecycle_val == ExperienceLifecycle.STABLE.value:
            return 1.0

        if not exp.created_at:
            return 1.0

        exp_dt = exp.created_at if exp.created_at.tzinfo else exp.created_at.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (now - exp_dt).total_seconds() / 86400.0)

        if lifecycle_val in (ExperienceLifecycle.TEMPORARY.value, ExperienceLifecycle.TIME_BOUND.value):
            if age_days <= 7.0:
                return 1.0
            elif age_days <= 30.0:
                return 0.7
            elif age_days <= 90.0:
                return 0.4
            else:
                return 0.2

        if lifecycle_val == ExperienceLifecycle.RECURRING.value:
            if age_days <= 30.0:
                return 1.0
            elif age_days <= 90.0:
                return 0.8
            else:
                return 0.6

        return 1.0

    def _score_and_rank_experiences(
        self,
        candidates: List[Tuple[Experience, float]],
        detected_dimensions: List[RetrievalDimension],
        now: datetime,
    ) -> List[PersonalContextItem]:
        """Compute multi-signal quality score and rank experience candidates descending.

        Maintains relevance dominance while factoring dimension match, importance,
        and durability-aware freshness.
        """
        settings = get_settings()
        w_sim = settings.personal_context_weight_similarity
        w_dim = settings.personal_context_weight_dimension
        w_imp = settings.personal_context_weight_importance
        w_rec = settings.personal_context_weight_recency

        scored_items: List[PersonalContextItem] = []

        for exp, similarity in candidates:
            matched_dims = self._dimension_analyzer.match_experience_dimensions(exp)

            # 1. Similarity score
            sim_val = max(0.0, min(similarity, 1.0))

            # 2. Dimension alignment boost (1.0 if matching detected dimension, else 0.0)
            if detected_dimensions and set(detected_dimensions).intersection(set(matched_dims)):
                dim_boost = 1.0
            else:
                dim_boost = 0.0

            # 3. Importance boost
            imp_val = (
                exp.importance.value
                if hasattr(exp.importance, "value")
                else str(exp.importance or "MEDIUM").upper()
            )
            if imp_val == ExperienceImportance.HIGH.value:
                imp_boost = 1.0
            elif imp_val == ExperienceImportance.MEDIUM.value:
                imp_boost = 0.5
            else:
                imp_boost = 0.0  # LOW

            # 4. Durability-aware freshness / recency boost
            freshness_factor = self._calculate_freshness_factor(exp=exp, now=now)

            if exp.created_at:
                exp_dt = exp.created_at if exp.created_at.tzinfo else exp.created_at.replace(tzinfo=timezone.utc)
                age_days = max(0.0, (now - exp_dt).total_seconds() / 86400.0)
            else:
                age_days = 0.0

            # Compute recency component scaled by durability freshness
            lifecycle_val = (
                exp.lifecycle.value
                if hasattr(exp.lifecycle, "value")
                else str(exp.lifecycle or "STABLE").upper()
            )

            if lifecycle_val == ExperienceLifecycle.STABLE.value:
                rec_boost = 1.0
            elif age_days <= 7.0:
                rec_boost = 1.0 * freshness_factor
            elif age_days <= 30.0:
                rec_boost = 0.5 * freshness_factor
            else:
                rec_boost = 0.0 * freshness_factor

            # Composite weighted score
            composite_score = (
                (w_sim * sim_val)
                + (w_dim * dim_boost)
                + (w_imp * imp_boost)
                + (w_rec * rec_boost)
            )

            exp_emo_dict = (
                exp.emotional_context.to_dict()
                if hasattr(exp.emotional_context, "to_dict")
                else (exp.emotional_context if isinstance(exp.emotional_context, dict) else None)
            )
            exp_people_list = (
                [p.to_dict() if hasattr(p, "to_dict") else p for p in exp.people_involved]
                if exp.people_involved
                else None
            )

            life_status_val = (
                exp.lifecycle_status.value
                if hasattr(exp.lifecycle_status, "value")
                else (str(exp.lifecycle_status) if exp.lifecycle_status else "ACTIVE")
            )

            scored_items.append(
                PersonalContextItem(
                    experience_id=exp.id,
                    content=exp.content,
                    type=exp.type.value if exp.type and hasattr(exp.type, "value") else (str(exp.type) if exp.type else None),
                    domain=exp.domain,
                    importance=imp_val or "MEDIUM",
                    lifecycle=lifecycle_val or "STABLE",
                    lifecycle_status=life_status_val or "ACTIVE",
                    matched_dimensions=matched_dims,
                    score=round(composite_score, 4),
                    similarity=round(similarity, 4),
                    emotional_context=exp_emo_dict,
                    people_involved=exp_people_list,
                    temporal_context=exp.temporal_context,
                    evidence_level=exp.evidence_level.value if hasattr(exp.evidence_level, "value") else str(exp.evidence_level or "EXTRACTED"),
                    created_at=exp.created_at,
                )
            )

        # Deterministic descending sort: score descending, then similarity descending, then recency descending
        scored_items.sort(
            key=lambda x: (
                x.score,
                x.similarity,
                x.created_at.timestamp() if x.created_at else 0.0,
            ),
            reverse=True,
        )

        return scored_items

    # =========================================================================
    # Personal Patterns Processing Helpers
    # =========================================================================

    def _validate_pattern_candidates(
        self,
        patterns: List[Any],
        target_user_id: str,
    ) -> List[Any]:
        """Validate candidate patterns and enforce fail-closed user isolation.

        Rejects:
        - Patterns missing ID or description
        - Patterns with mismatched or invalid user_id
        """
        valid: List[Any] = []
        for pat in patterns:
            if not getattr(pat, "id", None) or not isinstance(pat.id, uuid.UUID):
                logger.warning("Rejected pattern with missing or invalid ID.")
                continue

            desc = getattr(pat, "description", None)
            if not desc or not str(desc).strip():
                logger.warning("Rejected pattern %s with empty description.", pat.id)
                continue

            # Fail-closed user isolation if user_id is present on pattern entity
            if hasattr(pat, "user_id") and pat.user_id is not None:
                pat_user = _normalize_user_id(pat.user_id)
                if pat_user is None or pat_user != target_user_id:
                    logger.warning(
                        "SECURITY: Dropped pattern due to user isolation violation [pattern_id=%s, pat_user=%s, target_user=%s]",
                        pat.id,
                        pat_user,
                        target_user_id,
                    )
                    continue

            valid.append(pat)

        return valid

    def _filter_pattern_lifecycle(
        self,
        patterns: List[Any],
    ) -> List[Any]:
        """Filter patterns to allow only active states (HYPOTHESIS and CONFIRMED).

        Excludes:
        - WEAKENED
        - SUPERSEDED
        """
        active: List[Any] = []
        for pat in patterns:
            status_val = pat.status.value if hasattr(pat.status, "value") else str(pat.status).upper()
            if status_val in (PatternStatus.HYPOTHESIS.value, PatternStatus.CONFIRMED.value):
                active.append(pat)
            else:
                logger.debug("Filtered inactive pattern %s with status %s.", pat.id, status_val)

        return active

    def _deduplicate_patterns(
        self,
        patterns: List[Any],
    ) -> List[Any]:
        """Deduplicate patterns by ID and normalized description."""
        seen_ids: Set[uuid.UUID] = set()
        seen_descriptions: Dict[str, Any] = {}

        for pat in patterns:
            if pat.id in seen_ids:
                continue
            seen_ids.add(pat.id)

            norm_desc = self._normalize_content(pat.description)
            if norm_desc not in seen_descriptions:
                seen_descriptions[norm_desc] = pat
            else:
                # Keep pattern with higher confidence or evidence count
                existing = seen_descriptions[norm_desc]
                existing_ev = len(existing.evidence_ids) if getattr(existing, "evidence_ids", None) else 0
                current_ev = len(pat.evidence_ids) if getattr(pat, "evidence_ids", None) else 0

                if (pat.confidence, current_ev) > (existing.confidence, existing_ev):
                    seen_descriptions[norm_desc] = pat

        return list(seen_descriptions.values())

    def _score_and_rank_patterns(
        self,
        patterns: List[Any],
        query: str,
        detected_dimensions: List[RetrievalDimension],
        conversation_context: Optional[List[LLMMessage]],
        min_query_relevance: float,
    ) -> List[PersonalPatternContextItem]:
        """Score, gate, and rank pattern candidates deterministically."""
        scored_patterns: List[PersonalPatternContextItem] = []

        for pat in patterns:
            pat_status_val = pat.status.value if hasattr(pat.status, "value") else str(pat.status)
            pat_dims = self._dimension_analyzer.match_pattern_dimensions(pat)

            # Compute query relevance
            q_rel = self._dimension_analyzer.calculate_pattern_query_relevance(
                query=query,
                pattern=pat,
                conversation_context=conversation_context,
            )

            # Retrieval Gate: Query relevance is the primary gate for candidate inclusion.
            if q_rel < min_query_relevance:
                continue

            # Check dimension alignment (ranking boost signal)
            dim_matched = bool(detected_dimensions and set(detected_dimensions).intersection(set(pat_dims)))
            dim_score = 1.0 if dim_matched else 0.0

            # Calculate evidence strength
            ev_count = len(pat.evidence_ids) if getattr(pat, "evidence_ids", None) else 0
            ev_strength = min(ev_count / 5.0, 1.0)

            # Deterministic composite pattern ranking score:
            # Priority: query relevance (40%) + dimension alignment (30%) + confidence (20%) + evidence count (10%)
            pat_score = (
                (0.40 * q_rel)
                + (0.30 * dim_score)
                + (0.20 * pat.confidence)
                + (0.10 * ev_strength)
            )

            scored_patterns.append(
                PersonalPatternContextItem(
                    pattern_id=pat.id,
                    description=pat.description,
                    domain=pat.domain,
                    confidence=pat.confidence,
                    status=pat_status_val,
                    evidence_count=ev_count,
                    score=round(pat_score, 4),
                    matched_dimensions=pat_dims,
                    first_observed_at=pat.first_observed_at,
                    last_observed_at=pat.last_observed_at,
                )
            )

        # Deterministic sort: score descending, then confidence descending, then evidence count descending
        scored_patterns.sort(key=lambda p: (p.score, p.confidence, p.evidence_count), reverse=True)

        return scored_patterns
