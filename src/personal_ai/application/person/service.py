from datetime import datetime, timezone
from typing import Any, List, Optional
import uuid

from personal_ai.application.memory.quality_service import MemoryQualityService
from personal_ai.core.logger import get_logger
from personal_ai.domain.experience.entity import Experience
from personal_ai.domain.experience.emotional_context import PersonInvolved
from personal_ai.domain.person.entity import Person, PersonalPersonContext
from personal_ai.domain.person.enums import RelationshipType
from personal_ai.domain.person.repository import PersonRepository

logger = get_logger(__name__)


def _normalize_user_id(user_id: Any) -> Optional[uuid.UUID]:
    """Convert any user_id representation to UUID safely. Fails closed on missing or malformed input."""
    if user_id is None:
        return None
    if isinstance(user_id, uuid.UUID):
        return user_id
    if isinstance(user_id, str):
        cleaned = user_id.strip()
        if not cleaned:
            return None
        try:
            return uuid.UUID(cleaned)
        except (ValueError, AttributeError):
            return None
    return None


def _normalize_uuid(val: Any) -> Optional[uuid.UUID]:
    """Convert any UUID representation to UUID safely. Fails closed on missing or malformed input."""
    if val is None:
        return None
    if isinstance(val, uuid.UUID):
        return val
    if isinstance(val, str):
        cleaned = val.strip()
        if not cleaned:
            return None
        try:
            return uuid.UUID(cleaned)
        except (ValueError, AttributeError):
            return None
    return None


class PersonService:
    """Foundational application service for the People / Relationship model (PR #29).

    Core Architecture:
        Person (stable entity: name, relationship_type, notes, user ownership)
            +
        Experience (evidence: people_involved with person_id & name, emotional & temporal context)
            +
        MemoryQualityService (canonical validation, lifecycle filtering & deduplication)
            ↓
        Person-Scoped Retrieval (get_experiences_for_person, get_person_context)

    Safety Invariants:
        1. Purely Observational: Represents individuals and contextual experiences without
           inferring relationship health scores, attachment styles, or personality judgments.
        2. Strict User Isolation: All operations are strictly bounded to the authenticated user_id.
        3. Application-Controlled Creation: resolve_person searches deterministically and
           only creates new Person entities when explicitly instructed (no hallucinated auto-creation).
    """

    def __init__(
        self,
        person_repo: PersonRepository,
        memory_quality_service: Optional[MemoryQualityService] = None,
    ) -> None:
        """Initialize PersonService with repository and quality service."""
        self._person_repo = person_repo
        self._quality_service = memory_quality_service or MemoryQualityService()

    async def create_person(
        self,
        user_id: uuid.UUID,
        name: str,
        relationship_type: RelationshipType = RelationshipType.OTHER,
        notes: Optional[str] = None,
    ) -> Person:
        """Create and persist a new Person entity.

        Args:
            user_id: Authenticated user UUID.
            name: Non-empty name of the person.
            relationship_type: RelationshipType enum (defaults to OTHER).
            notes: Optional descriptive notes.

        Returns:
            Person: The persisted Person domain entity.

        Raises:
            ValueError: If user_id is invalid or name is empty.
        """
        norm_user = _normalize_user_id(user_id)
        if not norm_user:
            raise ValueError(f"Invalid user_id: {user_id}")

        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError("Person name must be a non-empty string.")

        person = Person(
            user_id=norm_user,
            name=name.strip(),
            relationship_type=relationship_type,
            notes=notes,
        )

        created = await self._person_repo.create(person)
        logger.info(
            "Created Person entity [person_id=%s, user_id=%s, name='%s', relationship_type=%s]",
            created.id,
            norm_user,
            created.name,
            created.relationship_type.value,
        )
        return created

    async def get_person_by_id(
        self,
        user_id: uuid.UUID,
        person_id: uuid.UUID,
    ) -> Optional[Person]:
        """Retrieve a Person by unique ID scoped to authenticated user.

        Args:
            user_id: Authenticated user UUID.
            person_id: Person UUID.

        Returns:
            Optional[Person]: Found Person entity or None.
        """
        norm_user = _normalize_user_id(user_id)
        norm_pid = _normalize_uuid(person_id)
        if not norm_user or not norm_pid:
            return None

        return await self._person_repo.get_by_id(user_id=norm_user, person_id=norm_pid)

    async def list_people(
        self,
        user_id: uuid.UUID,
    ) -> List[Person]:
        """List all Person records belonging to an authenticated user.

        Args:
            user_id: Authenticated user UUID.

        Returns:
            List[Person]: List of Person entities.
        """
        norm_user = _normalize_user_id(user_id)
        if not norm_user:
            return []

        return await self._person_repo.list(user_id=norm_user)

    async def find_by_name(
        self,
        user_id: uuid.UUID,
        name: str,
    ) -> Optional[Person]:
        """Find a Person by case-insensitive exact name match for an authenticated user.

        Args:
            user_id: Authenticated user UUID.
            name: Exact name to find.

        Returns:
            Optional[Person]: Matching Person entity or None.
        """
        norm_user = _normalize_user_id(user_id)
        if not norm_user or not name or not isinstance(name, str) or not name.strip():
            return None

        return await self._person_repo.find_by_name(user_id=norm_user, name=name.strip())

    async def resolve_person(
        self,
        user_id: uuid.UUID,
        name: str,
        create_if_missing: bool = False,
        relationship_type: RelationshipType = RelationshipType.OTHER,
    ) -> Optional[Person]:
        """Resolve a person reference by name, optionally creating one with explicit control.

        Steps:
            1. Validate user identity.
            2. Normalize name conservatively.
            3. Search exact case-insensitive match.
            4. Return existing Person if found.
            5. If not found and create_if_missing=True, create Person.

        Args:
            user_id: Authenticated user UUID.
            name: Name string.
            create_if_missing: True to create a Person entity if not found.
            relationship_type: Initial relationship type if created (defaults to OTHER).

        Returns:
            Optional[Person]: Resolved or newly created Person entity, or None.
        """
        norm_user = _normalize_user_id(user_id)
        if not norm_user or not name or not isinstance(name, str) or not name.strip():
            return None

        clean_name = name.strip()
        existing = await self.find_by_name(user_id=norm_user, name=clean_name)
        if existing:
            return existing

        if create_if_missing:
            return await self.create_person(
                user_id=norm_user,
                name=clean_name,
                relationship_type=relationship_type,
            )

        return None

    async def update_person(
        self,
        user_id: uuid.UUID,
        person_id: uuid.UUID,
        name: Optional[str] = None,
        relationship_type: Optional[RelationshipType] = None,
        notes: Optional[str] = None,
    ) -> Optional[Person]:
        """Update an existing Person entity with strict user isolation.

        Args:
            user_id: Authenticated user UUID.
            person_id: Person UUID.
            name: Optional new name.
            relationship_type: Optional new RelationshipType.
            notes: Optional new notes.

        Returns:
            Optional[Person]: Updated Person entity or None if not found.
        """
        norm_user = _normalize_user_id(user_id)
        norm_pid = _normalize_uuid(person_id)
        if not norm_user or not norm_pid:
            return None

        existing = await self.get_person_by_id(user_id=norm_user, person_id=norm_pid)
        if not existing:
            return None

        if name is not None:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Person name must be a non-empty string.")
            existing.name = name.strip()

        if relationship_type is not None:
            if isinstance(relationship_type, str):
                try:
                    existing.relationship_type = RelationshipType(relationship_type.upper().strip())
                except ValueError:
                    existing.relationship_type = RelationshipType.OTHER
            elif isinstance(relationship_type, RelationshipType):
                existing.relationship_type = relationship_type

        if notes is not None:
            existing.notes = str(notes).strip() or None

        existing.updated_at = datetime.now(timezone.utc)
        return await self._person_repo.update(user_id=norm_user, person=existing)

    async def delete_person(
        self,
        user_id: uuid.UUID,
        person_id: uuid.UUID,
    ) -> bool:
        """Delete a Person entity scoped to authenticated user.

        Args:
            user_id: Authenticated user UUID.
            person_id: Person UUID.

        Returns:
            bool: True if deleted, False if not found or cross-user.
        """
        norm_user = _normalize_user_id(user_id)
        norm_pid = _normalize_uuid(person_id)
        if not norm_user or not norm_pid:
            return False

        return await self._person_repo.delete(user_id=norm_user, person_id=norm_pid)

    async def get_experiences_for_person(
        self,
        user_id: uuid.UUID,
        person_id: uuid.UUID,
        experiences: Optional[List[Experience]] = None,
    ) -> List[Experience]:
        """Retrieve quality-filtered experiences associated with a specific Person.

        Matches experiences by either:
        - Explicit person_id link on PersonInvolved
        - Case-insensitive exact name match on PersonInvolved

        Args:
            user_id: Authenticated user UUID.
            person_id: Person UUID.
            experiences: Candidate experiences list.

        Returns:
            List[Experience]: Quality-filtered experiences involving the person.
        """
        norm_user = _normalize_user_id(user_id)
        norm_pid = _normalize_uuid(person_id)
        if not norm_user or not norm_pid:
            return []

        # Verify person exists and belongs to the authenticated user
        person = await self.get_person_by_id(user_id=norm_user, person_id=norm_pid)
        if not person:
            logger.warning(
                "get_experiences_for_person failed: person_id=%s not found for user_id=%s",
                norm_pid,
                norm_user,
            )
            return []

        if not experiences:
            return []

        # 1. Apply canonical MemoryQualityService validation, lifecycle filtering, and deduplication
        filtered = self._quality_service.filter_and_deduplicate_experiences(
            user_id=norm_user,
            experiences=experiences,
            is_historical=False,
        )

        if not filtered:
            return []

        # 2. Filter experiences involving this specific person
        person_name_lower = person.name.strip().lower()
        matched: List[Experience] = []

        for exp in filtered:
            # Enforce user isolation on experience
            exp_user = _normalize_user_id(exp.user_id)
            if exp_user is None or exp_user != norm_user:
                continue

            if not exp.people_involved:
                continue

            for p in exp.people_involved:
                # Handle PersonInvolved object or dictionary
                p_id: Optional[uuid.UUID] = None
                p_name: Optional[str] = None

                if isinstance(p, PersonInvolved):
                    p_id = p.person_id
                    p_name = p.name
                elif isinstance(p, dict):
                    raw_id = p.get("person_id")
                    if raw_id:
                        try:
                            p_id = uuid.UUID(str(raw_id).strip())
                        except (ValueError, AttributeError):
                            p_id = None
                    p_name = p.get("name")

                # Match by explicit ID or case-insensitive exact name
                if p_id and p_id == norm_pid:
                    matched.append(exp)
                    break
                elif p_name and p_name.strip().lower() == person_name_lower:
                    matched.append(exp)
                    break

        return matched

    async def get_person_context(
        self,
        user_id: uuid.UUID,
        person_id: uuid.UUID,
        experiences: Optional[List[Experience]] = None,
    ) -> Optional[PersonalPersonContext]:
        """Assemble structured, person-scoped context for an authenticated user.

        Args:
            user_id: Authenticated user UUID.
            person_id: Person UUID.
            experiences: Optional candidate experiences list.

        Returns:
            Optional[PersonalPersonContext]: Context container or None if person not found.
        """
        norm_user = _normalize_user_id(user_id)
        norm_pid = _normalize_uuid(person_id)
        if not norm_user or not norm_pid:
            return None

        person = await self.get_person_by_id(user_id=norm_user, person_id=norm_pid)
        if not person:
            return None

        relevant_exps = await self.get_experiences_for_person(
            user_id=norm_user,
            person_id=norm_pid,
            experiences=experiences,
        )

        return PersonalPersonContext(
            person=person,
            relevant_experiences=relevant_exps,
            total_experiences=len(relevant_exps),
        )
