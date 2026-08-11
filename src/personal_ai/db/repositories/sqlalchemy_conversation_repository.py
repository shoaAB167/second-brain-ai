from typing import List, Optional, Union
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from personal_ai.db.models import Conversation, Message, MessageRole, utc_now
from personal_ai.db.repositories.base import ConversationRepository


class SQLAlchemyConversationRepository(ConversationRepository):
    """SQLAlchemy 2.x async implementation of ConversationRepository interface.

    Encapsulates database access using SQLAlchemy AsyncSession.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with an active AsyncSession.

        Args:
            session: Active async database session.
        """
        self._session = session

    async def create_conversation(self, user_id: Optional[str] = None) -> Conversation:
        """Create and persist a new Conversation.

        Args:
            user_id: Optional user identifier (reserved for future auth support).

        Returns:
            Conversation: The newly created conversation entity.
        """
        conversation = Conversation()
        self._session.add(conversation)
        await self._session.commit()
        await self._session.refresh(conversation)
        return conversation

    async def get_conversation(
        self,
        conversation_id: uuid.UUID,
        user_id: Optional[str] = None,
    ) -> Optional[Conversation]:
        """Fetch a conversation by ID.

        Args:
            conversation_id: UUID of the conversation.
            user_id: Optional user identifier (reserved for future auth support).

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
        """Add a message to a conversation and update the conversation's updated_at timestamp.

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

        # Update the parent conversation's updated_at timestamp to reflect last activity
        conversation = await self.get_conversation(conversation_id)
        if conversation:
            conversation.updated_at = utc_now()

        await self._session.commit()
        await self._session.refresh(message)
        return message
