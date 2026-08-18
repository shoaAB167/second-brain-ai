from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import uuid

from sqlalchemy import Index, String, Text, DateTime, ForeignKey, Enum as SQLEnum, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base Declarative class for SQLAlchemy models."""
    pass


class MessageRole(str, Enum):
    """Constrained message roles allowed in the database."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


def utc_now() -> datetime:
    """Return current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class Conversation(Base):
    """ORM Model representing a chat conversation thread."""

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    # One-to-many relationship with Message
    messages: Mapped[list["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )


class Message(Base):
    """ORM Model representing an individual message in a conversation."""

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[MessageRole] = mapped_column(
        SQLEnum(MessageRole, name="messagerole_enum", native_enum=True, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False, index=True
    )

    conversation: Mapped[Conversation] = relationship(
        "Conversation", back_populates="messages"
    )

    __table_args__ = (
        Index("ix_messages_conversation_id_created_at", "conversation_id", "created_at"),
    )


class ExperienceModel(Base):
    """ORM Model representing a raw, uninterpreted life Experience record."""

    __tablename__ = "experiences"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True, index=True
    )
    source_message_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(
        SQLEnum(
            "CHAT", "FILE", "EMAIL", "CALENDAR", "GITHUB", "VOICE", "API", "MANUAL", "TOOL",
            name="experiencesource_enum",
            native_enum=True,
        ),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        SQLEnum(
            "RECEIVED", "PROCESSING", "PROCESSED", "FAILED",
            name="experiencestatus_enum",
            native_enum=True,
        ),
        nullable=False,
        default="RECEIVED",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False, index=True
    )


class ExperienceClassificationModel(Base):
    """ORM Model representing classification metrics and provenance produced by ExperienceClassifier."""

    __tablename__ = "experience_classifications"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_message_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    experience_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiences.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    is_experience: Mapped[bool] = mapped_column(nullable=False)
    type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    importance: Mapped[float] = mapped_column(nullable=False)
    confidence: Mapped[float] = mapped_column(nullable=False)
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False, index=True
    )

