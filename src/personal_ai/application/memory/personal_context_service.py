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
    RetrievalDimension,
)
from personal_ai.infrastructure.embedding.provider import EmbeddingProvider
from personal_ai.llm.models import LLMMessage

logger = get_logger(__name__)


class PersonalContextRetrievalService:
    """Application service for performing context-aware, dimension-guided personal memory retrieval."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        experience_repo: ExperienceRepository,
        dimension_analyzer: Optional[QueryDimensionAnalyzer] = None,
    ) -> None:
        """Initialize PersonalContextRetrievalService with abstract ports.

        Args:
            embedding_provider: EmbeddingProvider for query vector generation.
            experience_repo: ExperienceRepository interface for user-scoped vector persistence.
            dimension_analyzer: Optional QueryDimensionAnalyzer instance.
        """
        self._provider = embedding_provider
        self._experience_repo = experience_repo
        self._dimension_analyzer = dimension_analyzer or QueryDimensionAnalyzer()

    async def retrieve_context(
        self,
        user_id: uuid.UUID,
        query: str,
        conversation_context: Optional[List[LLMMessage]] = None,
        candidate_limit: Optional[int] = None,
        final_limit: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        include_historical: Optional[bool] = None,
    ) -> PersonalContext:
        """Retrieve bounded, dimension-aware PersonalContext for the authenticated user and query.

        Combines:
        - Query & context dimension understanding
        - User-scoped semantic vector candidate retrieval
        - Multi-signal re-ranking (vector similarity + dimension match + importance + recency + lifecycle)
        - Bounded top-K selection

        Args:
            user_id: Authenticated user UUID (strict isolation).
            query: Current user query message text.
            conversation_context: Optional recent conversation context messages.
            candidate_limit: Optional override for candidate vector search limit.
            final_limit: Optional override for final context items limit.
            similarity_threshold: Optional minimum cosine similarity threshold.
            include_historical: Optional flag to explicitly search historical/superseded memories.

        Returns:
            PersonalContext: Bounded personal context object containing ranked items.

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
        threshold = similarity_threshold

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

        # Step 5: Multi-Signal Composite Scoring & Re-Ranking
        # Dominant signal: semantic similarity (e.g. 70%), with small bounded boosts for dimension (15%), importance (10%), recency (5%)
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

            # Simplified composite weighted score
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

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "Personal context retrieved [user_id=%s, candidates=%d, selected=%d, dimensions=%s, duration_ms=%.1f]",
            user_id,
            total_candidates,
            len(final_items),
            [d.value for d in detected_dimensions],
            duration_ms,
        )

        return PersonalContext(
            user_id=user_id,
            query=query,
            detected_dimensions=detected_dimensions,
            items=final_items,
            total_candidates=total_candidates,
        )
