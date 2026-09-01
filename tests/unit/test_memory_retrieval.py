from datetime import datetime, timezone
from typing import List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock
import uuid
import pytest

from personal_ai.application.memory import MemoryRetrievalService, MemorySearchResult
from personal_ai.core.exceptions import AppException
from personal_ai.domain.experience import (
    Experience,
    ExperienceRepository,
    ExperienceSource,
    ExperienceStatus,
    ExperienceType,
)
from personal_ai.infrastructure.embedding import MockEmbeddingProvider
from personal_ai.llm.exceptions import LLMConnectionException


class InMemoryExperienceRepository(ExperienceRepository):
    """In-memory Experience repository implementing search_by_vector for unit testing."""

    def __init__(self) -> None:
        self.experiences: List[Experience] = []

    async def create(self, experience: Experience) -> Experience:
        self.experiences.append(experience)
        return experience

    async def update(self, experience: Experience) -> Experience:
        for i, exp in enumerate(self.experiences):
            if exp.id == experience.id:
                self.experiences[i] = experience
                return experience
        self.experiences.append(experience)
        return experience

    async def get_by_id(self, experience_id: uuid.UUID) -> Optional[Experience]:
        for exp in self.experiences:
            if exp.id == experience_id:
                return exp
        return None

    async def get_by_source_message_id(self, source_message_id: uuid.UUID) -> Optional[Experience]:
        for exp in self.experiences:
            if exp.source_message_id == source_message_id:
                return exp
        return None

    async def search_by_vector(
        self,
        user_id: uuid.UUID,
        query_vector: List[float],
        limit: int = 5,
        threshold: Optional[float] = None,
    ) -> List[Tuple[Experience, float]]:
        import math

        user_str = str(user_id)
        candidates = [
            e
            for e in self.experiences
            if e.user_id == user_str
            and e.embedding is not None
            and e.embedding_status == "COMPLETED"
        ]

        scored: List[Tuple[Experience, float, float]] = []
        for exp in candidates:
            vec = exp.embedding
            assert vec is not None
            dot = sum(a * b for a, b in zip(vec, query_vector))
            norm_a = math.sqrt(sum(a * a for a in vec)) or 1.0
            norm_b = math.sqrt(sum(b * b for b in query_vector)) or 1.0
            sim = dot / (norm_a * norm_b)
            dist = 1.0 - sim

            # Filter threshold BEFORE limit
            if threshold is not None and sim < threshold:
                continue

            scored.append((exp, dist, sim))

        # Sort by distance ascending (similarity descending)
        scored.sort(key=lambda item: item[1])

        results: List[Tuple[Experience, float]] = []
        for exp, dist, sim in scored[:limit]:
            results.append((exp, sim))
        return results


@pytest.mark.asyncio
async def test_a_basic_semantic_retrieval() -> None:
    """Requirement 18A: Basic semantic retrieval returns matching Experience."""
    user_id = uuid.uuid4()
    provider = MockEmbeddingProvider(dimensions=1536)
    repo = InMemoryExperienceRepository()

    career_vec = await provider.embed("career goal 30 LPA")
    exp = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="Reach a salary of 30 LPA",
        type=ExperienceType.GOAL,
        domain="career",
        status=ExperienceStatus.RECEIVED,
        source=ExperienceSource.CHAT,
        embedding=career_vec,
        embedding_model="gemini-embedding-001",
        embedding_status="COMPLETED",
        embedded_at=datetime.now(timezone.utc),
    )
    await repo.create(exp)

    service = MemoryRetrievalService(embedding_provider=provider, experience_repo=repo)
    results = await service.search(user_id=user_id, query="career goal 30 LPA", limit=5)

    assert len(results) == 1
    assert results[0].experience_id == exp.id
    assert results[0].content == "Reach a salary of 30 LPA"
    assert results[0].type == "GOAL"
    assert results[0].similarity > 0.99


@pytest.mark.asyncio
async def test_b_ranking_most_relevant_ranks_first() -> None:
    """Requirement 18B: Career goal ranks above unrelated memories for career query."""
    user_id = uuid.uuid4()
    provider = MockEmbeddingProvider(dimensions=1536)
    repo = InMemoryExperienceRepository()

    career_vec = await provider.embed("Reach 30 LPA backend engineer")
    fitness_vec = await provider.embed("Run a 42km marathon")
    food_vec = await provider.embed("Prefers Italian pasta dishes")

    exp_career = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="Reach 30 LPA backend engineer",
        type=ExperienceType.GOAL,
        domain="career",
        source=ExperienceSource.CHAT,
        embedding=career_vec,
        embedding_model="gemini-embedding-001",
        embedding_status="COMPLETED",
    )
    exp_fitness = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="Run a 42km marathon",
        type=ExperienceType.GOAL,
        domain="fitness",
        source=ExperienceSource.CHAT,
        embedding=fitness_vec,
        embedding_model="gemini-embedding-001",
        embedding_status="COMPLETED",
    )
    exp_food = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="Prefers Italian pasta dishes",
        type=ExperienceType.PREFERENCE,
        domain="food",
        source=ExperienceSource.CHAT,
        embedding=food_vec,
        embedding_model="gemini-embedding-001",
        embedding_status="COMPLETED",
    )

    await repo.create(exp_fitness)
    await repo.create(exp_food)
    await repo.create(exp_career)

    service = MemoryRetrievalService(embedding_provider=provider, experience_repo=repo)
    results = await service.search(user_id=user_id, query="Reach 30 LPA backend engineer", limit=5)

    assert len(results) == 3
    # Career goal should be rank #1
    assert results[0].experience_id == exp_career.id
    assert results[0].similarity > results[1].similarity


@pytest.mark.asyncio
async def test_c_user_isolation_strictly_enforced() -> None:
    """Requirement 18C: User A cannot retrieve User B's memories."""
    user_a = uuid.uuid4()
    user_b = uuid.uuid4()
    provider = MockEmbeddingProvider(dimensions=1536)
    repo = InMemoryExperienceRepository()

    vec = await provider.embed("salary goal")

    exp_a = Experience(
        id=uuid.uuid4(),
        user_id=str(user_a),
        content="User A salary goal 30 LPA",
        type=ExperienceType.GOAL,
        source=ExperienceSource.CHAT,
        embedding=vec,
        embedding_model="gemini-embedding-001",
        embedding_status="COMPLETED",
    )
    exp_b = Experience(
        id=uuid.uuid4(),
        user_id=str(user_b),
        content="User B salary goal 50 LPA",
        type=ExperienceType.GOAL,
        source=ExperienceSource.CHAT,
        embedding=vec,
        embedding_model="gemini-embedding-001",
        embedding_status="COMPLETED",
    )
    await repo.create(exp_a)
    await repo.create(exp_b)

    service = MemoryRetrievalService(embedding_provider=provider, experience_repo=repo)

    results_a = await service.search(user_id=user_a, query="salary goal", limit=10)
    assert len(results_a) == 1
    assert results_a[0].experience_id == exp_a.id
    assert results_a[0].content == "User A salary goal 30 LPA"

    results_b = await service.search(user_id=user_b, query="salary goal", limit=10)
    assert len(results_b) == 1
    assert results_b[0].experience_id == exp_b.id
    assert results_b[0].content == "User B salary goal 50 LPA"


@pytest.mark.asyncio
async def test_d_null_embedding_excluded() -> None:
    """Requirement 18D: Experience with NULL embedding is excluded from search."""
    user_id = uuid.uuid4()
    provider = MockEmbeddingProvider(dimensions=1536)
    repo = InMemoryExperienceRepository()

    exp = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="No embedding experience",
        type=ExperienceType.FACT,
        source=ExperienceSource.CHAT,
        embedding=None,
        embedding_status="PENDING",
    )
    await repo.create(exp)

    service = MemoryRetrievalService(embedding_provider=provider, experience_repo=repo)
    results = await service.search(user_id=user_id, query="No embedding experience", limit=5)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_e_failed_embedding_excluded() -> None:
    """Requirement 18E: Experience with embedding_status=FAILED is excluded from search."""
    user_id = uuid.uuid4()
    provider = MockEmbeddingProvider(dimensions=1536)
    repo = InMemoryExperienceRepository()

    exp = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="Failed embedding experience",
        type=ExperienceType.FACT,
        source=ExperienceSource.CHAT,
        embedding=None,
        embedding_status="FAILED",
    )
    await repo.create(exp)

    service = MemoryRetrievalService(embedding_provider=provider, experience_repo=repo)
    results = await service.search(user_id=user_id, query="Failed embedding experience", limit=5)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_f_empty_results_when_no_memories() -> None:
    """Requirement 18F: Returns empty list [] when user has no matching memories."""
    user_id = uuid.uuid4()
    provider = MockEmbeddingProvider(dimensions=1536)
    repo = InMemoryExperienceRepository()

    service = MemoryRetrievalService(embedding_provider=provider, experience_repo=repo)
    results = await service.search(user_id=user_id, query="Anything", limit=5)
    assert results == []


@pytest.mark.asyncio
async def test_g_limit_enforced() -> None:
    """Requirement 18G: Limit parameter bounds number of results returned."""
    user_id = uuid.uuid4()
    provider = MockEmbeddingProvider(dimensions=1536)
    repo = InMemoryExperienceRepository()

    for i in range(10):
        vec = await provider.embed(f"Memory item {i}")
        await repo.create(
            Experience(
                id=uuid.uuid4(),
                user_id=str(user_id),
                content=f"Memory item {i}",
                type=ExperienceType.FACT,
                source=ExperienceSource.CHAT,
                embedding=vec,
                embedding_model="gemini-embedding-001",
                embedding_status="COMPLETED",
            )
        )

    service = MemoryRetrievalService(embedding_provider=provider, experience_repo=repo)
    results = await service.search(user_id=user_id, query="Memory item", limit=3)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_h_invalid_limits_raise_app_exception() -> None:
    """Requirement 3: Explicit limit validation (limits < 1 or > 20 raise 400 AppException)."""
    user_id = uuid.uuid4()
    provider = MockEmbeddingProvider(dimensions=1536)
    repo = InMemoryExperienceRepository()
    service = MemoryRetrievalService(embedding_provider=provider, experience_repo=repo)

    with pytest.raises(AppException) as exc_0:
        await service.search(user_id=user_id, query="Item", limit=0)
    assert exc_0.value.status_code == 400
    assert "between 1 and 20" in exc_0.value.message

    with pytest.raises(AppException) as exc_neg:
        await service.search(user_id=user_id, query="Item", limit=-5)
    assert exc_neg.value.status_code == 400

    with pytest.raises(AppException) as exc_25:
        await service.search(user_id=user_id, query="Item", limit=25)
    assert exc_25.value.status_code == 400


@pytest.mark.asyncio
async def test_regression_threshold_plus_limit_interaction() -> None:
    """Requirement 2: Regression test proving threshold filtering happens before applying LIMIT.

    Candidate similarities: 0.95, 0.91, 0.89, 0.87.
    With threshold=0.90, the 0.89 result must never consume a result slot.
    """
    user_id = uuid.uuid4()
    provider = MockEmbeddingProvider(dimensions=1536)

    # Mock repository returning deterministic similarities based on custom mock
    repo = MagicMock(spec=ExperienceRepository)

    exp_95 = Experience(id=uuid.uuid4(), user_id=str(user_id), content="Match 0.95", source=ExperienceSource.CHAT)
    exp_91 = Experience(id=uuid.uuid4(), user_id=str(user_id), content="Match 0.91", source=ExperienceSource.CHAT)
    exp_89 = Experience(id=uuid.uuid4(), user_id=str(user_id), content="Match 0.89", source=ExperienceSource.CHAT)
    exp_87 = Experience(id=uuid.uuid4(), user_id=str(user_id), content="Match 0.87", source=ExperienceSource.CHAT)

    candidates = [
        (exp_95, 0.95),
        (exp_91, 0.91),
        (exp_89, 0.89),
        (exp_87, 0.87),
    ]

    async def mock_search_by_vector(user_id, query_vector, limit=5, threshold=None):
        filtered = [(e, sim) for e, sim in candidates if threshold is None or sim >= threshold]
        return filtered[:limit]

    repo.search_by_vector = AsyncMock(side_effect=mock_search_by_vector)

    service = MemoryRetrievalService(embedding_provider=provider, experience_repo=repo)

    # Case 1: limit = 2, threshold = 0.90 -> expected [0.95, 0.91]
    res_limit_2 = await service.search(user_id=user_id, query="test", limit=2, threshold=0.90)
    assert len(res_limit_2) == 2
    assert [r.similarity for r in res_limit_2] == [0.95, 0.91]

    # Case 2: limit = 3, threshold = 0.90 -> expected [0.95, 0.91] (0.89 is not included)
    res_limit_3 = await service.search(user_id=user_id, query="test", limit=3, threshold=0.90)
    assert len(res_limit_3) == 2
    assert [r.similarity for r in res_limit_3] == [0.95, 0.91]


@pytest.mark.asyncio
async def test_invalid_threshold_raises_app_exception() -> None:
    """Verify threshold outside [-1.0, 1.0] raises 400 AppException."""
    user_id = uuid.uuid4()
    provider = MockEmbeddingProvider()
    repo = InMemoryExperienceRepository()
    service = MemoryRetrievalService(embedding_provider=provider, experience_repo=repo)

    with pytest.raises(AppException) as exc_low:
        await service.search(user_id=user_id, query="test", threshold=-1.5)
    assert exc_low.value.status_code == 400

    with pytest.raises(AppException) as exc_high:
        await service.search(user_id=user_id, query="test", threshold=1.5)
    assert exc_high.value.status_code == 400


@pytest.mark.asyncio
async def test_incompatible_embedding_model_raises_app_exception() -> None:
    """Requirement 4: Provider using incompatible model name raises 500 AppException."""
    user_id = uuid.uuid4()
    incompatible_provider = MockEmbeddingProvider(model="text-embedding-ada-002")
    repo = InMemoryExperienceRepository()
    service = MemoryRetrievalService(embedding_provider=incompatible_provider, experience_repo=repo)

    with pytest.raises(AppException) as exc_info:
        await service.search(user_id=user_id, query="test")

    assert exc_info.value.status_code == 500
    assert "incompatible with configured model" in exc_info.value.message


@pytest.mark.asyncio
async def test_j_query_embedding_failure_raises_app_exception() -> None:
    """Requirement 18J: Provider failure raises clean 502 AppException without fake results."""
    user_id = uuid.uuid4()
    failing_provider = MockEmbeddingProvider(should_fail=True)
    repo = InMemoryExperienceRepository()

    service = MemoryRetrievalService(embedding_provider=failing_provider, experience_repo=repo)
    with pytest.raises(AppException) as exc_info:
        await service.search(user_id=user_id, query="query", limit=5)

    assert exc_info.value.status_code == 502
    assert "Failed to generate query embedding" in exc_info.value.message


@pytest.mark.asyncio
async def test_k_dimension_mismatch_raises_app_exception() -> None:
    """Requirement 18K: Provider returning incorrect vector dimension raises clean 500 AppException."""
    user_id = uuid.uuid4()
    provider = MagicMock(spec=MockEmbeddingProvider)
    provider.model_name = "gemini-embedding-001"
    provider.dimensions = 1536
    # Provider returns wrong vector length 512
    provider.embed = AsyncMock(return_value=[0.1] * 512)
    repo = InMemoryExperienceRepository()

    service = MemoryRetrievalService(embedding_provider=provider, experience_repo=repo)
    with pytest.raises(AppException) as exc_info:
        await service.search(user_id=user_id, query="query", limit=5)

    assert exc_info.value.status_code == 500
    assert "dimension mismatch" in exc_info.value.message


@pytest.mark.asyncio
async def test_l_empty_query_raises_bad_request() -> None:
    """Requirement 18L: Empty query string raises 400 AppException."""
    user_id = uuid.uuid4()
    provider = MockEmbeddingProvider()
    repo = InMemoryExperienceRepository()

    service = MemoryRetrievalService(embedding_provider=provider, experience_repo=repo)
    with pytest.raises(AppException) as exc_info:
        await service.search(user_id=user_id, query="   ", limit=5)

    assert exc_info.value.status_code == 400
    assert "cannot be empty" in exc_info.value.message


@pytest.mark.asyncio
async def test_retrieval_returns_importance_and_lifecycle() -> None:
    """Verify MemorySearchResult preserves importance and lifecycle of retrieved experience."""
    from personal_ai.domain.experience import ExperienceImportance, ExperienceLifecycle

    user_id = uuid.uuid4()
    provider = MockEmbeddingProvider(dimensions=1536)
    repo = InMemoryExperienceRepository()

    vec = await provider.embed("gym routine 6 PM")
    exp = Experience(
        content="Usually goes to the gym around 6 PM",
        source=ExperienceSource.CHAT,
        user_id=str(user_id),
        type=ExperienceType.HABIT,
        importance=ExperienceImportance.HIGH,
        lifecycle=ExperienceLifecycle.RECURRING,
        embedding=vec,
        embedding_status="COMPLETED",
    )
    await repo.create(exp)

    service = MemoryRetrievalService(embedding_provider=provider, experience_repo=repo)
    results = await service.search(user_id=user_id, query="gym routine 6 PM", limit=5)

    assert len(results) == 1
    assert results[0].importance == "HIGH"
    assert results[0].lifecycle == "RECURRING"
    assert results[0].type == "HABIT"
    assert "usually" in results[0].content.lower() or "around" in results[0].content.lower()

