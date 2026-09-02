from typing import List, Optional
import uuid

from personal_ai.application.experience.evolution_classifier import (
    ExperienceEvolutionClassificationResult,
    ExperienceEvolutionClassifier,
)
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
        candidate_similarity_threshold: float = 0.4,
        supersede_confidence_threshold: float = 0.75,
    ) -> None:
        """Initialize ExperienceEvolutionService.

        Args:
            experience_repo: Repository for experience persistence.
            relationship_repo: Repository for relationship persistence.
            classifier: LLM-backed evolution classifier.
            candidate_similarity_threshold: Semantic similarity threshold to fetch candidate memories (default: 0.4).
            supersede_confidence_threshold: Minimum classifier confidence to trigger a SUPERSEDED transition (default: 0.75).
        """
        self._experience_repo = experience_repo
        self._relationship_repo = relationship_repo
        self._classifier = classifier
        self._candidate_similarity_threshold = candidate_similarity_threshold
        self._supersede_confidence_threshold = supersede_confidence_threshold

    async def evolve_experience(
        self,
        experience: Experience,
        user_id: uuid.UUID,
    ) -> List[ExperienceRelationship]:
        """Analyze a newly recorded experience against existing memories and apply evolution rules.

        Strictly user-scoped at the repository layer.
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
            "Experience evolution started [experience_id=%s, user_id=%s]",
            experience.id,
            user_id,
        )

        # 1. Fetch relevant existing candidate experiences for this user
        candidates_with_score = await self._experience_repo.search_by_vector(
            user_id=user_id,
            query_vector=experience.embedding,
            limit=5,
            threshold=self._candidate_similarity_threshold,
            lifecycle_status=None,  # Search all candidates to link historical/active memories
        )

        # Exclude self-comparison
        candidates = [c for c, _ in candidates_with_score if c.id != experience.id]

        logger.info(
            "Found %d evolution candidate experiences [experience_id=%s]",
            len(candidates),
            experience.id,
        )

        created_relationships: List[ExperienceRelationship] = []

        # 2. Compare against each candidate
        for candidate in candidates:
            try:
                classification: ExperienceEvolutionClassificationResult = (
                    await self._classifier.classify_relationship(
                        new_experience=experience,
                        existing_experience=candidate,
                    )
                )

                if (
                    classification.relationship == ExperienceRelationshipType.UNRELATED
                    or classification.confidence < 0.6
                ):
                    continue

                # 3. Prevent duplicate relationships
                already_exists = await self._relationship_repo.exists(
                    source_id=experience.id,
                    target_id=candidate.id,
                    relationship_type=classification.relationship,
                )

                rel: Optional[ExperienceRelationship] = None
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

                # 4. Apply conservative lifecycle transitions
                if classification.relationship in (
                    ExperienceRelationshipType.UPDATES,
                    ExperienceRelationshipType.SUPERSEDES,
                ):
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
                    "Error during candidate evolution analysis [new_id=%s, candidate_id=%s]: %s",
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
