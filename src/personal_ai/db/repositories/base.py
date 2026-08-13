from abc import ABC, abstractmethod
from typing import List, Optional, Union
import uuid

from personal_ai.db.models import Conversation, Message, MessageRole


class ConversationRepository(ABC):
    """Abstract interface for conversation persistence repositories.

    Enforces provider-independent persistence contract for Conversations and Messages.
    Includes optional user_id parameters to ensure future extensibility for
    multi-tenant authentication and ownership without breaking signatures.
    """

    @abstractmethod
    async def create_conversation(self, user_id: Optional[str] = None) -> Conversation:
        """Create and persist a new Conversation.

        Args:
            user_id: Optional user identifier for future authentication integration.

        Returns:
            Conversation: The created conversation entity.
        """
        pass

    @abstractmethod
    async def get_conversation(
        self,
        conversation_id: uuid.UUID,
        user_id: Optional[str] = None,
    ) -> Optional[Conversation]:
        """Fetch a conversation by ID and optional user_id.

        Args:
            conversation_id: UUID of the conversation.
            user_id: Optional user identifier for future ownership checks.

        Returns:
            Optional[Conversation]: Found conversation entity or None.
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
