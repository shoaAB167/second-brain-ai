from datetime import datetime, timezone
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

from personal_ai.application.memory.dimension_analyzer import QueryDimensionAnalyzer
from personal_ai.application.memory.quality_service import MemoryQualityService
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
        quality_service: Optional[MemoryQualityService] = None,
    ) -> None:
        """Initialize PersonalContextRetrievalService with abstract ports and quality service.

        Args:
            embedding_provider: EmbeddingProvider for query vector generation.
            experience_repo: ExperienceRepository interface for user-scoped vector persistence.
            dimension_analyzer: Optional QueryDimensionAnalyzer instance.
            pattern_repo: Optional PersonalPatternRepository interface for active pattern retrieval.
            quality_service: Optional MemoryQualityService instance for deterministic quality processing.
        """
        self._provider = embedding_provider
        self._experience_repo = experience_repo
        self._dimension_analyzer = dimension_analyzer or QueryDimensionAnalyzer()
        self._pattern_repo = pattern_repo
        self._quality_service = quality_service or MemoryQualityService(
            dimension_analyzer=self._dimension_analyzer
        )

    async def retrieve_context(
        self,
        user_id: uuid.UUID,
        query: str,
        conversation_context: Optional[List[LLMMessage]] = None,
        candidate_limit: Optional[int] = None,
        final_limit: Optional[int] = None,
        pattern_limit: Optional[int] = None,
        min_pattern_query_relevance: Optional[float] = None,
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
            min_pattern_query_relevance: Optional override for minimum query relevance threshold.
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
        min_pat_relevance = (
            min_pattern_query_relevance
            if min_pattern_query_relevance is not None
            else settings.personal_context_min_pattern_query_relevance
        )
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

        # Step 5: Retrieve candidate Personal Patterns strictly scoped to user_id (fail-safe)
        candidate_patterns: List[Any] = []
        total_pattern_candidates = 0

        if self._pattern_repo is not None:
            try:
                active_patterns = await self._pattern_repo.get_active_patterns(user_id=user_id)
                candidate_patterns = active_patterns or []
                total_pattern_candidates = len(candidate_patterns)
            except Exception as exc:
                logger.warning(
                    "Personal pattern candidate retrieval failed safely for user %s: %s",
                    user_id,
                    exc,
                )
                candidate_patterns = []

        # Step 6: Deterministic Memory & Pattern Quality Processing & Bounded Selection (PR #25)
        final_items, retrieved_patterns = self._quality_service.process_candidates(
            user_id=user_id,
            query=query,
            experience_candidates=scored_candidates,
            pattern_candidates=candidate_patterns,
            detected_dimensions=detected_dimensions,
            conversation_context=conversation_context,
            is_historical=is_historical,
            final_limit=fin_limit,
            pattern_limit=pat_lim,
            min_pattern_query_relevance=min_pat_relevance,
        )


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
