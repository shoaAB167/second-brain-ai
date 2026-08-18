from abc import ABC, abstractmethod
from typing import List, Optional, Union
import uuid

from personal_ai.db.models import Conversation, Message, MessageRole, User


class UserRepository(ABC):
    """Abstract interface for user identity persistence repositories."""

    @abstractmethod
    async def create_user(self, email: str, password_hash: str) -> User:
        """Create and persist a new User.

        Args:
            email: Unique email address.
            password_hash: Bcrypt password hash.

        Returns:
            User: Created ORM user model instance.
        """
        pass

    @abstractmethod
    async def get_user_by_id(self, user_id: uuid.UUID) -> Optional[User]:
        """Fetch user by unique UUID.

        Args:
            user_id: User UUID.

        Returns:
            Optional[User]: Found user or None.
        """
        pass

    @abstractmethod
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Fetch user by email address.

        Args:
            email: User's email string.

        Returns:
            Optional[User]: Found user or None.
        """
        pass


class ConversationRepository(ABC):
    """Abstract interface for conversation persistence repositories.

    Enforces provider-independent persistence contract for Conversations and Messages.
    Includes user_id parameters to ensure data ownership and isolation.
    """

    @abstractmethod
    async def create_conversation(self, user_id: Optional[uuid.UUID] = None) -> Conversation:
        """Create and persist a new Conversation.

        Args:
            user_id: Optional authenticated user UUID associated with the conversation.

        Returns:
            Conversation: The created conversation entity.
        """
        pass

    @abstractmethod
    async def get_conversation(
        self,
        conversation_id: uuid.UUID,
        user_id: Optional[uuid.UUID] = None,
    ) -> Optional[Conversation]:
        """Fetch a conversation by ID and optional user_id for ownership enforcement.

        Args:
            conversation_id: UUID of the conversation.
            user_id: Optional authenticated user UUID for ownership verification.

        Returns:
            Optional[Conversation]: Found conversation entity matching user_id or None.
        """
        pass

    @abstractmethod
    async def get_conversation_messages(self, conversation_id: uuid.UUID) -> List[Message]:
        """Fetch messages for a conversation in chronological order (oldest to newest).

        Args:
            conversation_id: UUID of the conversation.

        Returns:
            List[Message]: List of messages sorted by created_at ascending.
        """
        pass

    @abstractmethod
    async def add_message(
        self,
        conversation_id: uuid.UUID,
        role: Union[MessageRole, str],
        content: str,
    ) -> Message:
        """Add and persist a new message to an existing conversation.

        Args:
            conversation_id: UUID of the target conversation.
            role: Message sender role (user, assistant, system).
            content: Text message content.

        Returns:
            Message: The newly created message entity.
        """
        pass
