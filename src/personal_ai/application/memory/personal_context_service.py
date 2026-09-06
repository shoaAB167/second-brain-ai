from datetime import datetime, timezone
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

from personal_ai.application.memory.dimension_analyzer import QueryDimensionAnalyzer
from personal_ai.config.settings import get_settings
from personal_ai.core.exceptions import AppException
from personal_ai.core.logger import get_logger
from personal_ai.domain.experience import (
    Experience,
    ExperienceRepository,
    PersonalContext,
    PersonalContextItem,
    PersonalPatternContextItem,
    RetrievalDimension,
)
from personal_ai.domain.pattern.enums import PatternStatus
from personal_ai.domain.pattern.repository import PersonalPatternRepository
from personal_ai.infrastructure.embedding.provider import EmbeddingProvider
from personal_ai.llm.models import LLMMessage

logger = get_logger(__name__)


class PersonalContextRetrievalService:
    """Application service for performing context-aware, dimension-guided personal memory and pattern retrieval."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        experience_repo: ExperienceRepository,
        dimension_analyzer: Optional[QueryDimensionAnalyzer] = None,
        pattern_repo: Optional[PersonalPatternRepository] = None,
    ) -> None:
        """Initialize PersonalContextRetrievalService with abstract ports.

        Args:
            embedding_provider: EmbeddingProvider for query vector generation.
            experience_repo: ExperienceRepository interface for user-scoped vector persistence.
            dimension_analyzer: Optional QueryDimensionAnalyzer instance.
            pattern_repo: Optional PersonalPatternRepository interface for active pattern retrieval.
        """
        self._provider = embedding_provider
        self._experience_repo = experience_repo
        self._dimension_analyzer = dimension_analyzer or QueryDimensionAnalyzer()
        self._pattern_repo = pattern_repo

    async def retrieve_context(
        self,
        user_id: uuid.UUID,
        query: str,
        conversation_context: Optional[List[LLMMessage]] = None,
        candidate_limit: Optional[int] = None,
        final_limit: Optional[int] = None,
        pattern_limit: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        include_historical: Optional[bool] = None,
    ) -> PersonalContext:
        """Retrieve bounded, dimension-aware PersonalContext for the authenticated user and query.

        Combines:
        - Query & context dimension understanding
        - User-scoped semantic vector candidate retrieval
        - Multi-signal re-ranking (vector similarity + dimension match + importance + recency + lifecycle)
        - User-scoped active Personal Pattern hypothesis retrieval and deterministic ranking
        - Bounded top-K selection

        Args:
            user_id: Authenticated user UUID (strict isolation).
            query: Current user query message text.
            conversation_context: Optional recent conversation context messages.
            candidate_limit: Optional override for candidate vector search limit.
            final_limit: Optional override for final context items limit.
            pattern_limit: Optional override for personal context pattern limit.
            similarity_threshold: Optional minimum cosine similarity threshold.
            include_historical: Optional flag to explicitly search historical/superseded memories.

        Returns:
            PersonalContext: Bounded personal context object containing ranked items and patterns.

        Raises:
            AppException: On invalid user_id, empty query, or unrecoverable provider failure.
        """
        if not user_id:
            raise AppException(message="user_id is required for personal context retrieval.", status_code=400)
        if not query or not query.strip():
            raise AppException(message="Search query cannot be empty.", status_code=400)

        settings = get_settings()
        cand_limit = candidate_limit or settings.personal_context_candidate_limit
        fin_limit = final_limit or settings.personal_context_final_limit
        pat_lim = pattern_limit or settings.personal_context_pattern_limit
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.personal_context_similarity_threshold
        )

        w_sim = settings.personal_context_weight_similarity
        w_dim = settings.personal_context_weight_dimension
        w_imp = settings.personal_context_weight_importance
        w_rec = settings.personal_context_weight_recency

        start_time = time.perf_counter()

        # Step 1: Detect relevant retrieval dimensions from query & short-term context
        detected_dimensions = self._dimension_analyzer.analyze_query(
            query=query,
            conversation_context=conversation_context,
        )

        # Step 2: Conservative historical search policy
        # Only search historical/superseded memories if explicitly requested or high-confidence historical query
        is_historical = (
            include_historical
            if include_historical is not None
            else self._dimension_analyzer.is_historical_query(query)
        )
        lifecycle_filter = None if is_historical else "ACTIVE"

        # Step 3: Generate query embedding vector
        try:
            query_vector = await self._provider.embed(query.strip())
        except Exception as exc:
            logger.error("Personal context query embedding generation failed: %s", exc)
            raise AppException(
                message=f"Failed to generate query embedding: {str(exc)}",
                status_code=502,
            )

        # Validate embedding consistency
        if len(query_vector) != self._provider.dimensions or len(query_vector) != settings.embedding_dimensions:
            logger.error(
                "Query vector dimension mismatch [expected=%d, got=%d]",
                settings.embedding_dimensions,
                len(query_vector),
            )
            raise AppException(
                message="Query vector dimension mismatch with configured embedding model.",
                status_code=500,
            )

        # Step 4: Retrieve candidate experiences scoped strictly to user_id
        scored_candidates: List[Tuple[Experience, float]] = await self._experience_repo.search_by_vector(
            user_id=user_id,
            query_vector=query_vector,
            limit=cand_limit,
            threshold=threshold,
            lifecycle_status=lifecycle_filter,
        )

        total_candidates = len(scored_candidates)

        # Step 5: Multi-Signal Composite Scoring & Re-Ranking for Memories
        scored_items: List[PersonalContextItem] = []
        now = datetime.now(timezone.utc)

        for exp, similarity in scored_candidates:
            matched_dims = self._dimension_analyzer.match_experience_dimensions(exp)

            # 1. Similarity score component
            sim_val = max(0.0, min(similarity, 1.0))

            # 2. Dimension alignment boost (1.0 if matching detected dimension, else 0.0)
            if detected_dimensions and set(detected_dimensions).intersection(set(matched_dims)):
                dim_boost = 1.0
            else:
                dim_boost = 0.0

            # 3. Importance boost (HIGH: 1.0, MEDIUM: 0.5, LOW: 0.0)
            imp_val = (
                exp.importance.value if hasattr(exp.importance, "value") else str(exp.importance or "")
            ).upper()
            if imp_val == "HIGH":
                imp_boost = 1.0
            elif imp_val == "MEDIUM":
                imp_boost = 0.5
            else:
                imp_boost = 0.0  # LOW

            # 4. Recency boost (decay over time)
            if exp.created_at:
                exp_dt = exp.created_at if exp.created_at.tzinfo else exp.created_at.replace(tzinfo=timezone.utc)
                age_days = (now - exp_dt).total_seconds() / 86400.0
            else:
                age_days = 0.0

            if age_days <= 7.0:
                rec_boost = 1.0
            elif age_days <= 30.0:
                rec_boost = 0.5
            else:
                rec_boost = 0.0

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
                    lifecycle=exp.lifecycle.value if hasattr(exp.lifecycle, "value") else (str(exp.lifecycle) if exp.lifecycle else "STABLE"),
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

        # Step 6: Rank descending by composite score and slice to final_limit
        scored_items.sort(key=lambda x: x.score, reverse=True)
        final_items = scored_items[:fin_limit]

        # Step 7: Retrieve and rank active Personal Patterns (fail-safe)
        retrieved_patterns: List[PersonalPatternContextItem] = []
        total_pattern_candidates = 0

        if self._pattern_repo is not None:
            try:
                # Retrieve active (HYPOTHESIS or CONFIRMED) patterns strictly scoped to user_id
                active_patterns = await self._pattern_repo.get_active_patterns(user_id=user_id)
                total_pattern_candidates = len(active_patterns)

                scored_patterns: List[PersonalPatternContextItem] = []
                for pat in active_patterns:
                    # Enforce active lifecycle states (exclude WEAKENED and SUPERSEDED)
                    pat_status_val = pat.status.value if hasattr(pat.status, "value") else str(pat.status)
                    if pat_status_val not in (PatternStatus.HYPOTHESIS.value, PatternStatus.CONFIRMED.value):
                        continue

                    pat_dims = self._dimension_analyzer.match_pattern_dimensions(pat)

                    # Compute query relevance
                    q_rel = self._dimension_analyzer.calculate_pattern_query_relevance(
                        query=query,
                        pattern=pat,
                        conversation_context=conversation_context,
                    )

                    # Check dimension alignment
                    dim_matched = bool(detected_dimensions and set(detected_dimensions).intersection(set(pat_dims)))
                    dim_score = 1.0 if dim_matched else 0.0

                    # Exclude pattern if completely irrelevant to both query and detected dimensions
                    if q_rel <= 0.0 and not dim_matched:
                        continue

                    # Calculate evidence strength
                    ev_count = len(pat.evidence_ids) if pat.evidence_ids else 0
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
                retrieved_patterns = scored_patterns[:pat_lim]

            except Exception as exc:
                logger.warning(
                    "Personal pattern retrieval failed safely for user %s: %s",
                    user_id,
                    exc,
                )
                retrieved_patterns = []

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "Personal context retrieved [user_id=%s, candidates=%d, selected=%d, patterns_selected=%d, dimensions=%s, duration_ms=%.1f]",
            user_id,
            total_candidates,
            len(final_items),
            len(retrieved_patterns),
            [d.value for d in detected_dimensions],
            duration_ms,
        )

        return PersonalContext(
            user_id=user_id,
            query=query,
            detected_dimensions=detected_dimensions,
            items=final_items,
            patterns=retrieved_patterns,
            total_candidates=total_candidates,
            total_pattern_candidates=total_pattern_candidates,
        )
