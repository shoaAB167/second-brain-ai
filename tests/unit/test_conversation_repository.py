from datetime import datetime, timezone
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from personal_ai.db.models import Base, MessageRole
from personal_ai.db.repositories import (
    ConversationRepository,
    SQLAlchemyConversationRepository,
)


@pytest_asyncio.fixture
async def db_session() -> AsyncSession:
    """Fixture providing an isolated in-memory SQLite async database session."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    await engine.dispose()


@pytest.mark.asyncio
async def test_repository_implements_abstract_interface(db_session: AsyncSession) -> None:
    """Verify SQLAlchemyConversationRepository implements ConversationRepository ABC."""
    repo = SQLAlchemyConversationRepository(session=db_session)
    assert isinstance(repo, ConversationRepository)


@pytest.mark.asyncio
async def test_create_and_get_conversation(db_session: AsyncSession) -> None:
    """Test creating and retrieving a conversation."""
    repo: ConversationRepository = SQLAlchemyConversationRepository(session=db_session)
    conversation = await repo.create_conversation()

    assert conversation.id is not None

    retrieved = await repo.get_conversation(conversation.id)
    assert retrieved is not None
    assert retrieved.id == conversation.id


@pytest.mark.asyncio
async def test_add_message_updates_conversation_updated_at(db_session: AsyncSession) -> None:
    """Test adding a message updates parent Conversation.updated_at timestamp."""
    repo: ConversationRepository = SQLAlchemyConversationRepository(session=db_session)
    conversation = await repo.create_conversation()

    initial_updated_at = conversation.updated_at

    # Small delay to ensure timestamp difference
    import asyncio
    await asyncio.sleep(0.01)

    await repo.add_message(conversation.id, role=MessageRole.USER, content="Hello")
    updated_conversation = await repo.get_conversation(conversation.id)

    assert updated_conversation is not None
    assert updated_conversation.updated_at.timestamp() > initial_updated_at.timestamp()


@pytest.mark.asyncio
async def test_add_and_retrieve_messages_chronological(db_session: AsyncSession) -> None:
    """Test adding user and assistant messages and retrieving them in chronological order."""
    repo: ConversationRepository = SQLAlchemyConversationRepository(session=db_session)
    conversation = await repo.create_conversation()

    await repo.add_message(conversation.id, role=MessageRole.USER, content="First message")
    await repo.add_message(conversation.id, role=MessageRole.ASSISTANT, content="Second message")
    await repo.add_message(conversation.id, role=MessageRole.USER, content="Third message")

    messages = await repo.get_conversation_messages(conversation.id)

    assert len(messages) == 3
    assert messages[0].content == "First message"
    assert messages[0].role == MessageRole.USER
    assert messages[1].content == "Second message"
    assert messages[1].role == MessageRole.ASSISTANT
    assert messages[2].content == "Third message"
    assert messages[2].role == MessageRole.USER
