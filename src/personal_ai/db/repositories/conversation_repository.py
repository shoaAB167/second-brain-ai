from typing import List, Optional, Union
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from personal_ai.db.models import Conversation, Message, MessageRole


class ConversationRepository:
    """Repository encapsulating persistence operations for Conversations and Messages."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize ConversationRepository with an AsyncSession.

        Args:
            session: The active async database session.
        """
        self._session = session

    async def create_conversation(self) -> Conversation:
        """Create and persist a new Conversation.

        Returns:
            Conversation: The newly created conversation entity.
        """
        conversation = Conversation()
        self._session.add(conversation)
        await self._session.commit()
        await self._session.refresh(conversation)
        return conversation

    async def get_conversation(self, conversation_id: uuid.UUID) -> Optional[Conversation]:
        """Fetch a conversation by ID.

        Args:
            conversation_id: UUID of the conversation.

        Returns:
            Optional[Conversation]: Found conversation entity or None.
        """
        result = await self._session.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        return result.scalar_one_or_none()

    async def get_conversation_messages(self, conversation_id: uuid.UUID) -> List[Message]:
        """Fetch messages for a conversation in chronological order (oldest to newest).

        Args:
            conversation_id: UUID of the conversation.

        Returns:
            List[Message]: List of messages sorted by created_at ascending.
        """
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

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
            Message: The newly created and persisted message entity.
        """
        safe_role = role if isinstance(role, MessageRole) else MessageRole(role.lower())
        message = Message(
            conversation_id=conversation_id,
            role=safe_role,
            content=content,
        )
        self._session.add(message)
        await self._session.commit()
        await self._session.refresh(message)
        return message
