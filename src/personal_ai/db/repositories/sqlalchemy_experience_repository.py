import asyncio
import math
from typing import List, Optional, Tuple, Union
import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from personal_ai.db.models import ExperienceModel
from personal_ai.domain.experience import (
    Experience,
    ExperienceRepository,
    ExperienceSource,
    ExperienceStatus,
    ExperienceType,
)


class SQLAlchemyExperienceRepository(ExperienceRepository):
    """Concrete SQLAlchemy implementation of ExperienceRepository interface."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with SQLAlchemy AsyncSession.

        Args:
            session: Active database session.
        """
        self._session = session

    async def create(self, experience: Experience) -> Experience:
        """Persist a new Experience entity to PostgreSQL.

        Enforces source_message_id unique constraint race condition safety.

        Args:
            experience: Domain Experience entity to persist.

        Returns:
            Experience: Persisted Experience domain entity (or existing on duplicate race).
        """
        user_id_val = uuid.UUID(str(experience.user_id)) if experience.user_id else None
        source_msg_val = uuid.UUID(str(experience.source_message_id)) if experience.source_message_id else None

        exp_type_val = (
            experience.type.value if hasattr(experience.type, "value") else str(experience.type)
        ) if experience.type else None

        model = ExperienceModel(
            id=experience.id,
            user_id=user_id_val,
            source_message_id=source_msg_val,
            content=experience.content,
            type=exp_type_val,
            domain=experience.domain,
            extraction_confidence=experience.extraction_confidence,
            embedding=experience.embedding,
            embedding_model=experience.embedding_model,
            embedding_status=experience.embedding_status or "PENDING",
            embedded_at=experience.embedded_at,
            source=experience.source.value if hasattr(experience.source, "value") else str(experience.source),
            status=experience.status.value if hasattr(experience.status, "value") else str(experience.status),
            created_at=experience.created_at,
        )

        try:
            self._session.add(model)
            await self._session.commit()
            return self._model_to_domain(model)
        except IntegrityError:
            await self._session.rollback()
            self._session.expire_all()
            if experience.source_message_id:
                for _ in range(10):
                    existing = await self.get_by_source_message_id(experience.source_message_id)
                    if existing:
                        return existing
                    await asyncio.sleep(0.05)
            raise

    async def update(self, experience: Experience) -> Experience:
        """Update an existing Experience entity in the database.

        Args:
            experience: Domain Experience entity to update.

        Returns:
            Experience: Updated domain Experience entity.
        """
        stmt = select(ExperienceModel).where(ExperienceModel.id == experience.id)
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()

        if not model:
            raise ValueError(f"Cannot update non-existent Experience with ID {experience.id}")

        model.content = experience.content
        model.type = (
            experience.type.value if hasattr(experience.type, "value") else str(experience.type)
        ) if experience.type else None
        model.domain = experience.domain
        model.extraction_confidence = experience.extraction_confidence
        model.embedding = experience.embedding
        model.embedding_model = experience.embedding_model
        model.embedding_status = experience.embedding_status or "PENDING"
        model.embedded_at = experience.embedded_at
        model.status = experience.status.value if hasattr(experience.status, "value") else str(experience.status)

        await self._session.commit()
        return self._model_to_domain(model)

    async def get_by_id(self, experience_id: uuid.UUID) -> Optional[Experience]:
        """Retrieve an Experience entity by UUID.

        Args:
            experience_id: UUID of the target experience.

        Returns:
            Optional[Experience]: Found domain entity or None.
        """
        stmt = select(ExperienceModel).where(ExperienceModel.id == experience_id)
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()

        if not model:
            return None

        return self._model_to_domain(model)

    async def get_by_source_message_id(self, source_message_id: Union[uuid.UUID, str]) -> Optional[Experience]:
        """Retrieve an Experience domain entity by source message ID provenance.

        Args:
            source_message_id: UUID of the originating message.

        Returns:
            Optional[Experience]: Found domain entity or None.
        """
        if not source_message_id:
            return None

        source_uuid = uuid.UUID(str(source_message_id)) if isinstance(source_message_id, (uuid.UUID, str)) else source_message_id

        stmt = select(ExperienceModel).where(ExperienceModel.source_message_id == source_uuid)
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()

        if not model:
            return None

        return self._model_to_domain(model)

    async def search_by_vector(
        self,
        user_id: uuid.UUID,
        query_vector: List[float],
        limit: int = 5,
        threshold: Optional[float] = None,
    ) -> List[Tuple[Experience, float]]:
        """Search experiences for a specific user ordered by semantic cosine similarity to query_vector.

        Enforces strict user isolation at the database level.
        In PostgreSQL: uses pgvector cosine distance operator (<=>).
        In SQLite (test environment): evaluates cosine distance for user experiences.

        Args:
            user_id: Target user UUID.
            query_vector: Embedding vector for semantic comparison.
            limit: Maximum number of ranked results to return.
            threshold: Optional minimum cosine similarity threshold in [-1.0, 1.0].

        Returns:
            List[Tuple[Experience, float]]: List of (Experience, similarity_score) tuples ordered by descending similarity.
        """
        user_uuid = uuid.UUID(str(user_id)) if isinstance(user_id, str) else user_id

        # Determine dialect from connection
        bind = self._session.bind
        dialect_name = bind.dialect.name if bind else "postgresql"

        if dialect_name == "sqlite":
            # SQLite test environment fallback
            stmt = select(ExperienceModel).where(
                ExperienceModel.user_id == user_uuid,
                ExperienceModel.embedding.is_not(None),
                ExperienceModel.embedding_status == "COMPLETED",
            )
            res = await self._session.execute(stmt)
            models = list(res.scalars().all())

            scored: List[Tuple[ExperienceModel, float, float]] = []
            for m in models:
                if not m.embedding:
                    continue
                vec = [float(x) for x in m.embedding]
                dot = sum(a * b for a, b in zip(vec, query_vector))
                norm_a = math.sqrt(sum(a * a for a in vec)) or 1.0
                norm_b = math.sqrt(sum(b * b for b in query_vector)) or 1.0
                sim = dot / (norm_a * norm_b)
                dist = 1.0 - sim
                scored.append((m, dist, sim))

            # Order by distance ascending (most similar first)
            scored.sort(key=lambda item: item[1])

            results: List[Tuple[Experience, float]] = []
            for m, dist, sim in scored:
                if threshold is not None and sim < threshold:
                    continue
                results.append((self._model_to_domain(m), float(sim)))
                if len(results) >= limit:
                    break
            return results

        else:
            # PostgreSQL pgvector similarity query
            distance_expr = ExperienceModel.embedding.cosine_distance(query_vector).label("distance")
            stmt = (
                select(ExperienceModel, distance_expr)
                .where(
                    ExperienceModel.user_id == user_uuid,
                    ExperienceModel.embedding.is_not(None),
                    ExperienceModel.embedding_status == "COMPLETED",
                )
                .order_by(distance_expr.asc())
                .limit(limit)
            )

            res = await self._session.execute(stmt)
            rows = res.all()

            results = []
            for row in rows:
                m = row[0]
                dist = float(row[1]) if row[1] is not None else 1.0
                sim = 1.0 - dist
                if threshold is not None and sim < threshold:
                    continue
                results.append((self._model_to_domain(m), float(sim)))
            return results

    @staticmethod
    def _model_to_domain(model: ExperienceModel) -> Experience:
        """Convert SQLAlchemy ExperienceModel to domain Experience entity."""
        exp_type = None
        if model.type:
            try:
                exp_type = ExperienceType(model.type)
            except ValueError:
                exp_type = None

        return Experience(
            id=model.id,
            user_id=str(model.user_id) if model.user_id else None,
            source_message_id=model.source_message_id,
            content=model.content,
            type=exp_type,
            domain=model.domain,
            extraction_confidence=model.extraction_confidence,
            embedding=[float(x) for x in model.embedding] if model.embedding is not None else None,
            embedding_model=model.embedding_model,
            embedding_status=model.embedding_status or "PENDING",
            embedded_at=model.embedded_at,
            source=ExperienceSource(model.source),
            status=ExperienceStatus(model.status),
            created_at=model.created_at,
        )
