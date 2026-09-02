from typing import List, Optional
import uuid

from personal_ai.application.experience.evolution_classifier import (
    ExperienceEvolutionClassificationResult,
    ExperienceEvolutionClassifier,
)
from personal_ai.config.settings import get_settings
from personal_ai.core.logger import get_logger
from personal_ai.domain.experience.entity import Experience
from personal_ai.domain.experience.enums import (
    ExperienceLifecycleStatus,
    ExperienceRelationshipType,
)
from personal_ai.domain.experience.relationship import ExperienceRelationship
from personal_ai.domain.experience.relationship_repository import (
    ExperienceRelationshipRepository,
)
from personal_ai.domain.experience.repository import ExperienceRepository

logger = get_logger(__name__)


class ExperienceEvolutionService:
    """Application service coordinating memory evolution, relationship discovery, and lifecycle transitions."""

    def __init__(
        self,
        experience_repo: ExperienceRepository,
        relationship_repo: ExperienceRelationshipRepository,
        classifier: ExperienceEvolutionClassifier,
        candidate_limit: Optional[int] = None,
        candidate_similarity_threshold: Optional[float] = None,
        supersede_confidence_threshold: Optional[float] = None,
        min_relationship_confidence: Optional[float] = None,
    ) -> None:
        """Initialize ExperienceEvolutionService.

        Args:
            experience_repo: Repository for experience persistence.
            relationship_repo: Repository for relationship persistence.
            classifier: LLM-backed batch evolution classifier.
            candidate_limit: Maximum candidate memories to retrieve (default from settings or 3).
            candidate_similarity_threshold: Semantic similarity threshold to fetch candidates (default from settings or 0.4).
            supersede_confidence_threshold: Minimum classifier confidence to trigger a SUPERSEDED transition (default from settings or 0.75).
            min_relationship_confidence: Minimum classifier confidence to persist a relationship (default from settings or 0.60).
        """
        settings = get_settings()
        self._experience_repo = experience_repo
        self._relationship_repo = relationship_repo
        self._classifier = classifier
        self._candidate_limit = (
            candidate_limit
            if candidate_limit is not None
            else getattr(settings, "memory_evolution_candidate_limit", 3)
        )
        self._candidate_similarity_threshold = (
            candidate_similarity_threshold
            if candidate_similarity_threshold is not None
            else getattr(settings, "memory_evolution_similarity_threshold", 0.4)
        )
        self._supersede_confidence_threshold = (
            supersede_confidence_threshold
            if supersede_confidence_threshold is not None
            else getattr(settings, "memory_evolution_supersede_confidence_threshold", 0.75)
        )
        self._min_relationship_confidence = (
            min_relationship_confidence
            if min_relationship_confidence is not None
            else getattr(settings, "memory_evolution_min_relationship_confidence", 0.60)
        )

    async def evolve_experience(
        self,
        experience: Experience,
        user_id: uuid.UUID,
    ) -> List[ExperienceRelationship]:
        """Analyze a newly recorded experience against existing memories and apply evolution rules.

        Strictly user-scoped at the repository layer.
        Performs 0 LLM calls if 0 candidates are found, and exactly 1 batch LLM call for 1–N candidates.
        Does not mutate the incoming experience's lifecycle status (remains ACTIVE).
        Conserves old memories (never deletes).

        Args:
            experience: The newly created Experience entity (with vector embedding).
            user_id: The authenticated user UUID.

        Returns:
            List[ExperienceRelationship]: List of newly discovered and persisted relationships.
        """
        if not experience.embedding:
            logger.debug(
                "Skipping evolution analysis for experience without embedding [experience_id=%s]",
                experience.id,
            )
            return []

        logger.info(
            "Experience evolution started [experience_id=%s, user_id=%s, candidate_limit=%d, threshold=%.2f]",
            experience.id,
            user_id,
            self._candidate_limit,
            self._candidate_similarity_threshold,
        )

        # 1. Fetch relevant existing candidate experiences for this user (strictly user-scoped)
        candidates_with_score = await self._experience_repo.search_by_vector(
            user_id=user_id,
            query_vector=experience.embedding,
            limit=self._candidate_limit,
            threshold=self._candidate_similarity_threshold,
            lifecycle_status=None,  # Search all candidates to link historical/active memories
        )

        # Exclude self-comparison
        candidates = [c for c, _ in candidates_with_score if c.id != experience.id]

        # Requirement 4: 0 candidates -> 0 LLM calls
        if not candidates:
            logger.info(
                "No candidate memories found for evolution [experience_id=%s]",
                experience.id,
            )
            return []

        logger.info(
            "Found %d candidate memories for batch evolution classification [experience_id=%s]",
            len(candidates),
            experience.id,
        )

        # Requirement 4 & 5: Exactly ONE batch LLM call for all candidates
        classifications = await self._classifier.classify_relationships(
            new_experience=experience,
            candidate_experiences=candidates,
        )

        created_relationships: List[ExperienceRelationship] = []

        # Process each candidate's classification
        for candidate in candidates:
            classification: Optional[ExperienceEvolutionClassificationResult] = classifications.get(candidate.id)
            if not classification:
                continue

            if (
                classification.relationship == ExperienceRelationshipType.UNRELATED
                or classification.confidence < self._min_relationship_confidence
            ):
                continue

            # Prevent duplicate relationships
            try:
                already_exists = await self._relationship_repo.exists(
                    source_id=experience.id,
                    target_id=candidate.id,
                    relationship_type=classification.relationship,
                )

                if not already_exists:
                    rel = ExperienceRelationship(
                        source_experience_id=experience.id,
                        target_experience_id=candidate.id,
                        relationship_type=classification.relationship,
                        confidence=classification.confidence,
                        reason=classification.reason,
                    )
                    rel = await self._relationship_repo.create(rel)
                    created_relationships.append(rel)

                    logger.info(
                        "Experience relationship classified and persisted [source_id=%s, target_id=%s, relationship=%s, confidence=%.2f]",
                        experience.id,
                        candidate.id,
                        classification.relationship.value,
                        classification.confidence,
                    )

                # Apply conservative lifecycle transitions
                if classification.relationship == ExperienceRelationshipType.UPDATES:
                    if classification.confidence >= self._supersede_confidence_threshold:
                        if candidate.lifecycle_status == ExperienceLifecycleStatus.ACTIVE:
                            old_status = candidate.lifecycle_status
                            candidate.lifecycle_status = ExperienceLifecycleStatus.SUPERSEDED
                            await self._experience_repo.update(candidate)

                            logger.info(
                                "Experience lifecycle transitioned [experience_id=%s, old_status=%s, new_status=%s, trigger=%s, source_id=%s]",
                                candidate.id,
                                old_status.value,
                                candidate.lifecycle_status.value,
                                classification.relationship.value,
                                experience.id,
                            )

                elif classification.relationship == ExperienceRelationshipType.CONTRADICTS:
                    logger.info(
                        "Experience contradiction recorded without lifecycle transition [source_id=%s, target_id=%s]",
                        experience.id,
                        candidate.id,
                    )

                elif classification.relationship in (
                    ExperienceRelationshipType.REINFORCES,
                    ExperienceRelationshipType.RELATED,
                ):
                    logger.info(
                        "Experience reinforcement/relation recorded without lifecycle transition [source_id=%s, target_id=%s]",
                        experience.id,
                        candidate.id,
                    )

            except Exception as candidate_exc:
                logger.warning(
                    "Error during candidate evolution persistence [new_id=%s, candidate_id=%s]: %s",
                    experience.id,
                    candidate.id,
                    candidate_exc,
                )

        logger.info(
            "Experience evolution completed [experience_id=%s, relationships_created=%d]",
            experience.id,
            len(created_relationships),
        )

        return created_relationships
