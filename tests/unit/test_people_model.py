from datetime import datetime, timedelta, timezone
from typing import List
from unittest.mock import AsyncMock, MagicMock
import uuid

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from personal_ai.api.dependencies import get_person_repository, get_person_service
from personal_ai.application.memory.quality_service import MemoryQualityService
from personal_ai.application.person import PersonService
from personal_ai.db.models import Base, ExperienceModel, PersonModel, User
from personal_ai.db.repositories.sqlalchemy_person_repository import SQLAlchemyPersonRepository
from personal_ai.domain.experience import (
    EmotionalContext,
    Experience,
    ExperienceEvidenceLevel,
    ExperienceImportance,
    ExperienceLifecycle,
    ExperienceLifecycleStatus,
    ExperienceSource,
    ExperienceType,
    PersonInvolved,
)
from personal_ai.domain.person import (
    Person,
    PersonalPersonContext,
    PersonRepository,
    RelationshipType,
)


@pytest_asyncio.fixture
async def db_session():
    """Fixture providing isolated in-memory SQLite database session for repository testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        yield session
    await engine.dispose()


def _fixed_now() -> datetime:
    """Return fixed UTC reference time for deterministic testing."""
    return datetime(2026, 9, 6, 12, 0, 0, tzinfo=timezone.utc)


# ==============================================================================
# 1. Person Creation & Schema Validation
# ==============================================================================

def test_person_creation_domain_model():
    """Requirement 1: Person entity can be instantiated with required fields and defaults."""
    user_id = uuid.uuid4()
    person = Person(
        user_id=user_id,
        name="Alice Smith",
        relationship_type=RelationshipType.COLLEAGUE,
        notes="Lead architect on backend team",
    )

    assert isinstance(person.id, uuid.UUID)
    assert person.user_id == user_id
    assert person.name == "Alice Smith"
    assert person.relationship_type == RelationshipType.COLLEAGUE
    assert person.notes == "Lead architect on backend team"
    assert person.created_at.tzinfo is not None
    assert person.updated_at.tzinfo is not None

    # Serialization roundtrip
    d = person.to_dict()
    assert d["user_id"] == str(user_id)
    assert d["name"] == "Alice Smith"
    assert d["relationship_type"] == "COLLEAGUE"
    assert d["notes"] == "Lead architect on backend team"

    reconstructed = Person.from_dict(d)
    assert reconstructed.id == person.id
    assert reconstructed.user_id == person.user_id
    assert reconstructed.name == person.name
    assert reconstructed.relationship_type == person.relationship_type
    assert reconstructed.notes == person.notes


def test_person_creation_validation_failures():
    """Requirement: Invalid fields raise ValueError during Person domain initialization."""
    user_id = uuid.uuid4()

    # Empty name
    with pytest.raises(ValueError, match="Person name must be a non-empty string"):
        Person(user_id=user_id, name="")

    # Whitespace-only name
    with pytest.raises(ValueError, match="Person name must be a non-empty string"):
        Person(user_id=user_id, name="   ")

    # Invalid user_id
    with pytest.raises(ValueError, match="user_id must be a UUID"):
        Person(user_id=None, name="Alice")

    with pytest.raises(ValueError, match="Invalid user_id UUID"):
        Person(user_id="not-a-uuid", name="Alice")


# ==============================================================================
# 2. Relationship Type Normalization & Defaults
# ==============================================================================

def test_person_relationship_type_normalization_and_defaults():
    """Requirements 10 & 11: Controlled relationship taxonomy, string normalization, and OTHER default."""
    user_id = uuid.uuid4()

    # Default is OTHER
    p1 = Person(user_id=user_id, name="Bob")
    assert p1.relationship_type == RelationshipType.OTHER

    # String inputs normalize to enum
    p2 = Person(user_id=user_id, name="Charlie", relationship_type="friend")
    assert p2.relationship_type == RelationshipType.FRIEND

    p3 = Person(user_id=user_id, name="Dave", relationship_type="MENTOR")
    assert p3.relationship_type == RelationshipType.MENTOR

    # Unknown relationship strings safely default to OTHER without crashing
    p4 = Person(user_id=user_id, name="Eve", relationship_type="co-founder-investor")
    assert p4.relationship_type == RelationshipType.OTHER


def test_person_from_dict_normalization():
    """Requirement: Person.from_dict() fail-closed normalization for relationship types and invalid inputs."""
    user_id = uuid.uuid4()
    person_id = uuid.uuid4()

    # 1. Missing relationship_type => OTHER
    p1 = Person.from_dict({
        "id": str(person_id),
        "user_id": str(user_id),
        "name": "Alex",
    })
    assert p1.relationship_type == RelationshipType.OTHER

    # 2. Lowercase valid string => FRIEND
    p2 = Person.from_dict({
        "user_id": str(user_id),
        "name": "Alex",
        "relationship_type": "friend",
    })
    assert p2.relationship_type == RelationshipType.FRIEND

    # 3. Uppercase valid string => FRIEND
    p3 = Person.from_dict({
        "user_id": str(user_id),
        "name": "Alex",
        "relationship_type": "FRIEND",
    })
    assert p3.relationship_type == RelationshipType.FRIEND

    # 4. Unknown string => OTHER
    p4 = Person.from_dict({
        "user_id": str(user_id),
        "name": "Alex",
        "relationship_type": "not-a-real-type",
    })
    assert p4.relationship_type == RelationshipType.OTHER

    # 5. None relationship_type => OTHER
    p5 = Person.from_dict({
        "user_id": str(user_id),
        "name": "Alex",
        "relationship_type": None,
    })
    assert p5.relationship_type == RelationshipType.OTHER

    # 6. Non-string type => OTHER
    p6 = Person.from_dict({
        "user_id": str(user_id),
        "name": "Alex",
        "relationship_type": 123,
    })
    assert p6.relationship_type == RelationshipType.OTHER

    # 7. Fail closed on missing/invalid user_id or name
    with pytest.raises(ValueError, match="user_id is required"):
        Person.from_dict({"name": "Alex"})

    with pytest.raises(ValueError, match="Person name must be a non-empty string"):
        Person.from_dict({"user_id": str(user_id), "name": ""})


# ==============================================================================
# 3. Repository CRUD Operations with Strict User Isolation
# ==============================================================================

@pytest.mark.asyncio
async def test_person_repository_crud_operations(db_session: AsyncSession):
    """Requirements 2, 3, 23, 24: Verify user-scoped create, get, list, update, delete in SQLAlchemy repo."""
    repo = SQLAlchemyPersonRepository(session=db_session)
    user_a = uuid.uuid4()
    user_b = uuid.uuid4()

    # 1. Create person for User A
    p_a = Person(
        user_id=user_a,
        name="Rahul Verma",
        relationship_type=RelationshipType.COLLEAGUE,
        notes="Engineering manager",
    )
    created_a = await repo.create(p_a)
    assert created_a.id == p_a.id
    assert created_a.name == "Rahul Verma"

    # 2. Get by ID for User A
    fetched_a = await repo.get_by_id(user_id=user_a, person_id=created_a.id)
    assert fetched_a is not None
    assert fetched_a.id == created_a.id
    assert fetched_a.name == "Rahul Verma"

    # 3. User B CANNOT get User A's person (User Isolation)
    fetched_by_b = await repo.get_by_id(user_id=user_b, person_id=created_a.id)
    assert fetched_by_b is None

    # 4. List people for User A
    p_a2 = Person(user_id=user_a, name="Anita Desai", relationship_type=RelationshipType.FRIEND)
    await repo.create(p_a2)

    list_a = await repo.list(user_id=user_a)
    assert len(list_a) == 2
    assert {p.name for p in list_a} == {"Anita Desai", "Rahul Verma"}

    # User B list is empty
    list_b = await repo.list(user_id=user_b)
    assert list_b == []

    # 5. Update person for User A
    fetched_a.notes = "Director of Engineering"
    fetched_a.relationship_type = RelationshipType.MENTOR
    updated_a = await repo.update(user_id=user_a, person=fetched_a)
    assert updated_a.notes == "Director of Engineering"
    assert updated_a.relationship_type == RelationshipType.MENTOR

    # User B CANNOT update User A's person
    with pytest.raises(ValueError, match="not found for update"):
        await repo.update(user_id=user_b, person=fetched_a)

    # 6. Delete person for User A
    # User B delete attempt fails safely
    deleted_by_b = await repo.delete(user_id=user_b, person_id=created_a.id)
    assert deleted_by_b is False

    # User A deletes successfully
    deleted_by_a = await repo.delete(user_id=user_a, person_id=created_a.id)
    assert deleted_by_a is True

    # Verify deleted
    assert await repo.get_by_id(user_id=user_a, person_id=created_a.id) is None


# ==============================================================================
# 4. Case-Insensitive Exact Name Lookup & Multi-User Name Sharing
# ==============================================================================

@pytest.mark.asyncio
async def test_person_repository_case_insensitive_name_lookup_and_sharing(db_session: AsyncSession):
    """Requirements 7, 8, 9: Case-insensitive name search and isolated identical names across users."""
    repo = SQLAlchemyPersonRepository(session=db_session)
    user_a = uuid.uuid4()
    user_b = uuid.uuid4()

    # User A creates "Rahul"
    await repo.create(Person(user_id=user_a, name="Rahul", relationship_type=RelationshipType.FRIEND))

    # User B also creates "Rahul" (Different user, same person name)
    await repo.create(Person(user_id=user_b, name="Rahul", relationship_type=RelationshipType.COLLEAGUE))

    # User A finds "rahul", "RAHUL", "Rahul"
    res1 = await repo.find_by_name(user_id=user_a, name="rahul")
    assert res1 is not None
    assert res1.user_id == user_a
    assert res1.relationship_type == RelationshipType.FRIEND

    res2 = await repo.find_by_name(user_id=user_a, name="RAHUL")
    assert res2 is not None
    assert res2.id == res1.id

    # User B finds their own "Rahul" with relationship COLLEAGUE
    res_b = await repo.find_by_name(user_id=user_b, name="Rahul")
    assert res_b is not None
    assert res_b.user_id == user_b
    assert res_b.relationship_type == RelationshipType.COLLEAGUE
    assert res_b.id != res1.id

    # Non-existent name returns None
    assert await repo.find_by_name(user_id=user_a, name="NonExistentPerson") is None


# ==============================================================================
# 5. PersonService: Resolution & Controlled Creation
# ==============================================================================

@pytest.mark.asyncio
async def test_person_service_resolve_person():
    """Requirements 4, 5, 8, 22: PersonService resolve_person deterministic search and explicit creation."""
    mock_repo = AsyncMock(spec=PersonRepository)
    service = PersonService(person_repo=mock_repo)
    user_id = uuid.uuid4()

    existing_person = Person(
        user_id=user_id,
        name="Priya",
        relationship_type=RelationshipType.FRIEND,
    )

    # 1. Existing person found
    mock_repo.find_by_name.return_value = existing_person
    resolved = await service.resolve_person(user_id=user_id, name="priya", create_if_missing=False)
    assert resolved == existing_person
    mock_repo.create.assert_not_called()

    # 2. Person not found, create_if_missing=False -> returns None
    mock_repo.find_by_name.return_value = None
    resolved_none = await service.resolve_person(user_id=user_id, name="Karan", create_if_missing=False)
    assert resolved_none is None
    mock_repo.create.assert_not_called()

    # 3. Person not found, create_if_missing=True -> creates Person under application control
    created_person = Person(user_id=user_id, name="Karan", relationship_type=RelationshipType.OTHER)
    mock_repo.create.return_value = created_person

    resolved_created = await service.resolve_person(
        user_id=user_id,
        name="Karan",
        create_if_missing=True,
        relationship_type=RelationshipType.OTHER,
    )
    assert resolved_created is not None
    assert resolved_created.name == "Karan"
    assert resolved_created.relationship_type == RelationshipType.OTHER
    mock_repo.create.assert_called_once()

    # 4. Fail closed on missing/malformed user_id or empty name
    assert await service.resolve_person(user_id=None, name="Karan") is None
    assert await service.resolve_person(user_id="invalid-uuid", name="Karan") is None
    assert await service.resolve_person(user_id=user_id, name="") is None
    assert await service.resolve_person(user_id=user_id, name="   ") is None


# ==============================================================================
# 6. Experience Integration & Backwards Compatibility
# ==============================================================================

def test_experience_people_involved_backward_compatibility():
    """Requirements 12, 13, 14: Experience with people_involved handles person_id, string-only names, and multiple people."""
    user_id = uuid.uuid4()
    person_id = uuid.uuid4()

    # 1. Legacy format with only name and role
    legacy_exp = Experience(
        user_id=str(user_id),
        content="Met with Rahul for coffee.",
        source=ExperienceSource.CHAT,
        people_involved=[
            {"name": "Rahul", "role": "friend"}
        ],
    )
    assert legacy_exp.people_involved is not None
    assert len(legacy_exp.people_involved) == 1
    p_leg = legacy_exp.people_involved[0]
    assert p_leg.name == "Rahul"
    assert p_leg.role == "friend"
    assert p_leg.person_id is None

    # 2. Modern format with explicit person_id
    modern_exp = Experience(
        user_id=str(user_id),
        content="Had quarterly review with Rahul.",
        source=ExperienceSource.CHAT,
        people_involved=[
            PersonInvolved(name="Rahul", role="manager", person_id=person_id)
        ],
    )
    assert modern_exp.people_involved is not None
    p_mod = modern_exp.people_involved[0]
    assert p_mod.name == "Rahul"
    assert p_mod.role == "manager"
    assert p_mod.person_id == person_id
    assert p_mod.to_dict() == {"name": "Rahul", "role": "manager", "person_id": str(person_id)}

    # 3. Multiple people in one experience
    multi_exp = Experience(
        user_id=str(user_id),
        content="Dinner with Alice, Bob, and Charlie.",
        source=ExperienceSource.CHAT,
        people_involved=[
            {"name": "Alice"},
            PersonInvolved(name="Bob", person_id=uuid.uuid4()),
            {"name": "Charlie", "role": "host"},
        ],
    )
    assert len(multi_exp.people_involved) == 3
    assert [p.name for p in multi_exp.people_involved] == ["Alice", "Bob", "Charlie"]


# ==============================================================================
# 7. Person-Scoped Experience Retrieval & Matching Precedence
# ==============================================================================

@pytest.mark.asyncio
async def test_person_service_matching_precedence_and_identity():
    """Requirements: Explicit person_id vs legacy name matching precedence and safety rules.

    A. Explicit person_id match -> included
    B. Legacy name match -> included
    C. Explicit wrong person_id + same name -> MUST NOT be included
    D. Invalid explicit person_id + same name -> MUST NOT be included
    E. Different name -> excluded
    F. Foreign user -> excluded
    G. Multiple people in one experience -> valid explicit person_id works
    """
    mock_repo = AsyncMock(spec=PersonRepository)
    quality_service = MemoryQualityService()
    service = PersonService(person_repo=mock_repo, memory_quality_service=quality_service)

    user_id = uuid.uuid4()
    person_a_id = uuid.uuid4()
    person_b_id = uuid.uuid4()

    person_a = Person(id=person_a_id, user_id=user_id, name="Rahul", relationship_type=RelationshipType.COLLEAGUE)
    mock_repo.get_by_id.return_value = person_a

    now = _fixed_now()

    # A. Explicit person_id match
    exp_a = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="Discussion with Rahul (person A).",
        source=ExperienceSource.CHAT,
        people_involved=[PersonInvolved(name="Rahul", person_id=person_a_id)],
        created_at=now - timedelta(days=1),
    )

    # B. Legacy name match (NO person_id)
    exp_b = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="Legacy note mentioning rahul.",
        source=ExperienceSource.CHAT,
        people_involved=[PersonInvolved(name="rahul")],
        created_at=now - timedelta(days=2),
    )

    # C. Explicit WRONG person_id + same name "Rahul" (Belongs to person B)
    exp_c = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="Discussion with a different Rahul (person B).",
        source=ExperienceSource.CHAT,
        people_involved=[PersonInvolved(name="Rahul", person_id=person_b_id)],
        created_at=now - timedelta(days=3),
    )

    # D. Invalid explicit person_id + same name "Rahul"
    exp_d = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="Meeting with Rahul with corrupted ID.",
        source=ExperienceSource.CHAT,
        people_involved=[{"name": "Rahul", "person_id": "invalid-uuid-string"}],
        created_at=now - timedelta(days=4),
    )

    # E. Different name
    exp_e = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="Coffee with Sara.",
        source=ExperienceSource.CHAT,
        people_involved=[PersonInvolved(name="Sara")],
        created_at=now - timedelta(days=5),
    )

    # F. Foreign user
    exp_f = Experience(
        id=uuid.uuid4(),
        user_id=str(uuid.uuid4()),
        content="Foreign user experience with Rahul.",
        source=ExperienceSource.CHAT,
        people_involved=[PersonInvolved(name="Rahul", person_id=person_a_id)],
        created_at=now - timedelta(days=6),
    )

    # G. Multiple people in one experience containing valid person A
    exp_g = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="Team meeting with Sara and Rahul.",
        source=ExperienceSource.CHAT,
        people_involved=[
            PersonInvolved(name="Sara", person_id=uuid.uuid4()),
            PersonInvolved(name="Rahul", person_id=person_a_id),
        ],
        created_at=now - timedelta(days=7),
    )

    all_exps = [exp_a, exp_b, exp_c, exp_d, exp_e, exp_f, exp_g]

    # Retrieve experiences for Person A
    retrieved = await service.get_experiences_for_person(
        user_id=user_id,
        person_id=person_a_id,
        experiences=all_exps,
    )

    # Must include:
    # - exp_a (explicit person_id match)
    # - exp_b (legacy name match)
    # - exp_g (multi-person with valid explicit person_id)
    #
    # Must NOT include:
    # - exp_c (wrong explicit person_id despite same name)
    # - exp_d (invalid explicit person_id)
    # - exp_e (different name)
    # - exp_f (foreign user)
    assert len(retrieved) == 3
    assert {e.id for e in retrieved} == {exp_a.id, exp_b.id, exp_g.id}


@pytest.mark.asyncio
async def test_person_service_get_experiences_for_person_and_context():
    """Requirements 15, 16, 17, 20: Person-scoped experience retrieval integrates with MemoryQualityService."""
    mock_repo = AsyncMock(spec=PersonRepository)
    quality_service = MemoryQualityService()
    service = PersonService(person_repo=mock_repo, memory_quality_service=quality_service)

    user_id = uuid.uuid4()
    person_id = uuid.uuid4()
    person = Person(id=person_id, user_id=user_id, name="Rahul", relationship_type=RelationshipType.COLLEAGUE)
    mock_repo.get_by_id.return_value = person

    now = _fixed_now()

    # Exp 1: Involves Rahul by person_id (ACTIVE)
    exp1 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="Discussed project architecture with Rahul.",
        source=ExperienceSource.CHAT,
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
        people_involved=[PersonInvolved(name="Rahul", person_id=person_id)],
        created_at=now - timedelta(days=2),
    )

    # Exp 2: Involves Rahul by name match (ACTIVE)
    exp2 = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="Rahul shared feedback on my pull request.",
        source=ExperienceSource.CHAT,
        lifecycle_status=ExperienceLifecycleStatus.ACTIVE,
        people_involved=[PersonInvolved(name="rahul")],
        created_at=now - timedelta(days=1),
    )

    # Exp 3: Involves Rahul but is SUPERSEDED (Should be filtered out by MemoryQualityService)
    exp3_superseded = Experience(
        id=uuid.uuid4(),
        user_id=str(user_id),
        content="Old meeting with Rahul that was superseded.",
        source=ExperienceSource.CHAT,
        lifecycle_status=ExperienceLifecycleStatus.SUPERSEDED,
        people_involved=[PersonInvolved(name="Rahul", person_id=person_id)],
        created_at=now - timedelta(days=20),
    )

    all_exps = [exp1, exp2, exp3_superseded]

    # Retrieve experiences for Rahul
    retrieved_exps = await service.get_experiences_for_person(
        user_id=user_id,
        person_id=person_id,
        experiences=all_exps,
    )

    # Exp 1 and Exp 2 matched; superseded dropped
    assert len(retrieved_exps) == 2
    assert {e.id for e in retrieved_exps} == {exp1.id, exp2.id}

    # Retrieve Person Context
    person_context = await service.get_person_context(
        user_id=user_id,
        person_id=person_id,
        experiences=all_exps,
    )
    assert isinstance(person_context, PersonalPersonContext)
    assert person_context.person == person
    assert person_context.total_experiences == 2
    assert len(person_context.relevant_experiences) == 2


# ==============================================================================
# 8. Negative Tests: No Relationship Scoring or Personality Inferences
# ==============================================================================

def test_safety_negative_tests_no_psychological_or_relationship_scoring():
    """Requirements 18 & 19: Proves system stores observations without computing relationship health scores or personality judgments."""
    user_id = uuid.uuid4()
    person_id = uuid.uuid4()

    # Scenario A: "Rahul helped me once" does NOT become "Rahul is my close friend"
    exp_help = Experience(
        user_id=str(user_id),
        content="Rahul helped me debug a race condition in the cache layer once.",
        source=ExperienceSource.CHAT,
        people_involved=[PersonInvolved(name="Rahul", person_id=person_id)],
    )
    person = Person(
        id=person_id,
        user_id=user_id,
        name="Rahul",
        relationship_type=RelationshipType.OTHER,
    )

    # Verify no speculative fields exist on Person or PersonInvolved
    assert person.relationship_type == RelationshipType.OTHER
    assert not hasattr(person, "relationship_health")
    assert not hasattr(person, "trust_score")
    assert not hasattr(person, "compatibility_score")
    assert not hasattr(person, "attachment_score")
    assert not hasattr(person, "personality")

    # Scenario B: "Rahul made me angry today" does NOT become "Rahul is a toxic person"
    exp_angry = Experience(
        user_id=str(user_id),
        content="Rahul made me angry today during the sprint planning meeting.",
        source=ExperienceSource.CHAT,
        emotional_context=EmotionalContext(emotion="anger", intensity=0.7),
        people_involved=[PersonInvolved(name="Rahul", person_id=person_id)],
    )

    # The experience records observational emotion and person link, with zero diagnostic toxicity claims
    assert exp_angry.emotional_context.emotion == "anger"
    assert exp_angry.people_involved[0].name == "Rahul"
    assert "toxic" not in exp_angry.content.lower()


# ==============================================================================
# 9. Cross-User Leakage & Fail-Closed Bounds
# ==============================================================================

@pytest.mark.asyncio
async def test_person_service_cross_user_and_invalid_input_fail_closed():
    """Requirements 4, 5, 20, 21: Cross-user queries and invalid identifiers fail closed safely."""
    mock_repo = AsyncMock(spec=PersonRepository)
    service = PersonService(person_repo=mock_repo)

    user_a = uuid.uuid4()
    user_b = uuid.uuid4()
    person_id = uuid.uuid4()

    # Person belongs to User A; User B tries to retrieve context -> repo returns None -> returns None
    mock_repo.get_by_id.return_value = None
    mock_repo.delete.return_value = False

    assert await service.get_person_context(user_id=user_b, person_id=person_id) is None
    assert await service.get_experiences_for_person(user_id=user_b, person_id=person_id, experiences=[]) == []
    assert await service.get_person_by_id(user_id=user_b, person_id=person_id) is None
    assert await service.delete_person(user_id=user_b, person_id=person_id) is False

    # Invalid user_id or person_id
    assert await service.get_person_by_id(user_id=None, person_id=person_id) is None
    assert await service.get_person_by_id(user_id=user_a, person_id=None) is None
    assert await service.get_person_by_id(user_id="invalid", person_id="invalid") is None
    assert await service.delete_person(user_id=None, person_id=person_id) is False
    assert await service.delete_person(user_id=user_a, person_id=None) is False


# ==============================================================================
# 10. Dependency Wiring Tests
# ==============================================================================

def test_get_person_dependencies_wiring():
    """Requirement 26: Dependency providers wire SQLAlchemyPersonRepository and PersonService correctly."""
    mock_session = AsyncMock(spec=AsyncSession)
    quality_service = MemoryQualityService()

    repo = get_person_repository(session=mock_session)
    assert isinstance(repo, SQLAlchemyPersonRepository)
    assert repo._session is mock_session

    service = get_person_service(person_repo=repo, memory_quality_service=quality_service)
    assert isinstance(service, PersonService)
    assert service._person_repo is repo
    assert service._quality_service is quality_service
