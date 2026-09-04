from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Enum as SQLEnum, ForeignKey, Index, JSON, String, Text, UUID
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


class User(Base):
    """ORM Model representing an authenticated user identity."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )
    password_hash: Mapped[str] = mapped_column(
        String(255), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    conversations: Mapped[list["Conversation"]] = relationship(
        "Conversation", back_populates="user", cascade="all, delete-orphan"
    )
    experiences: Mapped[list["ExperienceModel"]] = relationship(
        "ExperienceModel", back_populates="user", cascade="all, delete-orphan"
    )


class Conversation(Base):
    """ORM Model representing a chat conversation thread."""

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    user: Mapped[Optional[User]] = relationship("User", back_populates="conversations")

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
    """ORM Model representing a raw or structured life Experience record."""

    __tablename__ = "experiences"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    source_message_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
        unique=True,
        index=True,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, index=True)
    domain: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    importance: Mapped[str] = mapped_column(String(20), nullable=False, default="MEDIUM", index=True)
    lifecycle: Mapped[str] = mapped_column(String(20), nullable=False, default="STABLE", index=True)
    lifecycle_status: Mapped[str] = mapped_column(String(20), nullable=False, default="ACTIVE", index=True)
    emotional_context: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    people_involved: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    temporal_context: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    evidence_level: Mapped[str] = mapped_column(String(30), nullable=False, default="EXTRACTED", index=True)
    extraction_confidence: Mapped[Optional[float]] = mapped_column(nullable=True)

    # Embedding & Vector Storage Fields (PR #10)
    embedding: Mapped[Optional[list[float]]] = mapped_column(
        Vector(1536).with_variant(JSON, "sqlite"), nullable=True
    )
    embedding_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    embedding_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="PENDING", index=True
    )
    embedded_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

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

    user: Mapped[Optional[User]] = relationship("User", back_populates="experiences")


class ExperienceRelationshipModel(Base):
    """ORM Model representing directional typed relationships between experiences."""

    __tablename__ = "experience_relationships"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_experience_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiences.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    target_experience_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiences.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    relationship_type: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(nullable=False)
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False, index=True
    )

    __table_args__ = (
        Index(
            "ix_experience_rel_source_target_type",
            "source_experience_id",
            "target_experience_id",
            "relationship_type",
            unique=True,
        ),
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
        unique=True,
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
