import asyncio
from unittest.mock import AsyncMock, MagicMock
import uuid

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from personal_ai.application.experience.background_processor import (
    BackgroundExperienceProcessor,
)
from personal_ai.core.exceptions import AppException
from personal_ai.db.models import Base, MessageRole
from personal_ai.db.repositories import (
    ConversationRepository,
    SQLAlchemyConversationRepository,
)
from personal_ai.llm import (
    LLMAuthenticationException,
    LLMClient,
    LLMException,
    LLMMessage,
    LLMResponse,
    LLMStreamChunk,
)
from personal_ai.models.chat import ChatRequest
from personal_ai.services.chat_service import ChatService


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
async def test_chat_service_creates_conversation_when_id_absent(db_session: AsyncSession) -> None:
    """Verify ChatService creates a new conversation when conversation_id is None."""
    repo: ConversationRepository = SQLAlchemyConversationRepository(session=db_session)
    mock_llm_client = MagicMock(spec=LLMClient)
    mock_llm_client.generate_response = AsyncMock(
        return_value=LLMResponse(
            content="Hello back",
            provider="openai",
            model="gpt-4o-mini",
            latency_ms=50.0,
        )
    )

    service = ChatService(llm_client=mock_llm_client, conversation_repo=repo)
    user_id = uuid.uuid4()
    response = await service.process_chat(ChatRequest(message="Hello"), user_id=user_id)

    assert response.conversation_id is not None
    assert response.response == "Hello back"

    # Verify messages stored in DB and conversation belongs to user_id
    conversation = await repo.get_conversation(response.conversation_id, user_id=user_id)
    assert conversation is not None
    assert conversation.user_id == user_id

    messages = await repo.get_conversation_messages(response.conversation_id)
    assert len(messages) == 2
    assert messages[0].role == MessageRole.USER
    assert messages[0].content == "Hello"
    assert messages[1].role == MessageRole.ASSISTANT
    assert messages[1].content == "Hello back"


@pytest.mark.asyncio
async def test_chat_service_reuses_existing_conversation(db_session: AsyncSession) -> None:
    """Verify ChatService reuses existing conversation and passes history to LLM in order."""
    repo: ConversationRepository = SQLAlchemyConversationRepository(session=db_session)
    user_id = uuid.uuid4()
    conversation = await repo.create_conversation(user_id=user_id)

    # Pre-populate history
    await repo.add_message(conversation.id, role="user", content="Hi")
    await repo.add_message(conversation.id, role="assistant", content="Hello!")

    mock_llm_client = MagicMock(spec=LLMClient)
    mock_llm_client.generate_response = AsyncMock(
        return_value=LLMResponse(
            content="I am doing well",
            provider="openai",
            model="gpt-4o-mini",
            latency_ms=60.0,
        )
    )

    service = ChatService(llm_client=mock_llm_client, conversation_repo=repo)
    response = await service.process_chat(
        ChatRequest(conversation_id=conversation.id, message="How are you?"),
        user_id=user_id,
    )

    assert response.conversation_id == conversation.id
    assert response.response == "I am doing well"

    # Verify history passed to LLM was in chronological order: Hi -> Hello! -> How are you?
    mock_llm_client.generate_response.assert_called_once()
    passed_messages = mock_llm_client.generate_response.call_args.kwargs["messages"]

    assert len(passed_messages) == 3
    assert passed_messages[0] == LLMMessage(role="user", content="Hi")
    assert passed_messages[1] == LLMMessage(role="assistant", content="Hello!")
    assert passed_messages[2] == LLMMessage(role="user", content="How are you?")

    # Verify 4 messages in DB now
    messages = await repo.get_conversation_messages(conversation.id)
    assert len(messages) == 4


@pytest.mark.asyncio
async def test_sync_chat_does_not_wait_for_classification(db_session: AsyncSession) -> None:
    """Requirement A: Verify synchronous chat does NOT wait for background experience processor."""
    repo: ConversationRepository = SQLAlchemyConversationRepository(session=db_session)
    conversation = await repo.create_conversation()

    mock_llm_client = MagicMock(spec=LLMClient)
    mock_llm_client.generate_response = AsyncMock(
        return_value=LLMResponse(content="Sync chat response", provider="openai", model="gpt-4o-mini", latency_ms=10.0)
    )

    slow_bg_processor = MagicMock(spec=BackgroundExperienceProcessor)

    async def slow_bg_promo(*args, **kwargs):
        await asyncio.sleep(1.0)

    slow_bg_processor.process_background_promotion = AsyncMock(side_effect=slow_bg_promo)

    service = ChatService(
        llm_client=mock_llm_client,
        conversation_repo=repo,
        bg_processor=slow_bg_processor,
    )

    start_time = asyncio.get_event_loop().time()
    response = await service.process_chat(
        ChatRequest(conversation_id=conversation.id, message="Fast sync request")
    )
    elapsed = asyncio.get_event_loop().time() - start_time

    assert response.response == "Sync chat response"
    assert elapsed < 0.2  # Must finish immediately without waiting for slow_bg_promo (1.0s)


@pytest.mark.asyncio
async def test_chat_service_llm_failure_does_not_persist_assistant_message(
    db_session: AsyncSession,
) -> None:
    """Verify LLM failure leaves user message persisted but creates no assistant message."""
    repo: ConversationRepository = SQLAlchemyConversationRepository(session=db_session)
    conversation = await repo.create_conversation()

    mock_llm_client = MagicMock(spec=LLMClient)
    mock_llm_client.generate_response = AsyncMock(
        side_effect=LLMException("LLM error")
    )

    service = ChatService(llm_client=mock_llm_client, conversation_repo=repo)

    with pytest.raises(LLMException):
        await service.process_chat(
            ChatRequest(conversation_id=conversation.id, message="Attempted prompt")
        )

    # User message should be persisted, but NO assistant message created
    messages = await repo.get_conversation_messages(conversation.id)
    assert len(messages) == 1
    assert messages[0].role == MessageRole.USER
    assert messages[0].content == "Attempted prompt"


@pytest.mark.asyncio
async def test_chat_service_invalid_conversation_id_raises_404(
    db_session: AsyncSession,
) -> None:
    """Verify invalid/nonexistent conversation_id raises 404 AppException."""
    repo: ConversationRepository = SQLAlchemyConversationRepository(session=db_session)
    mock_llm_client = MagicMock(spec=LLMClient)

    service = ChatService(llm_client=mock_llm_client, conversation_repo=repo)
    non_existent_id = uuid.uuid4()

    with pytest.raises(AppException) as exc_info:
        await service.process_chat(
            ChatRequest(conversation_id=non_existent_id, message="Hello")
        )

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_chat_service_streaming_success(db_session: AsyncSession) -> None:
    """Verify process_chat_stream yields chunks in order, emits done event, and persists exactly ONE assistant message."""
    repo: ConversationRepository = SQLAlchemyConversationRepository(session=db_session)
    conversation = await repo.create_conversation()

    async def mock_stream_gen(*args, **kwargs):
        yield LLMStreamChunk(content="Hello")
        yield LLMStreamChunk(content=" world")

    mock_llm_client = MagicMock(spec=LLMClient)
    mock_llm_client.stream_response = MagicMock(side_effect=mock_stream_gen)

    service = ChatService(llm_client=mock_llm_client, conversation_repo=repo)
    events = [
        event
        async for event in service.process_chat_stream(
            ChatRequest(conversation_id=conversation.id, message="Hi")
        )
    ]

    assert len(events) == 3
    assert 'data: {"type":"token","content":"Hello"}' in events[0]
    assert 'data: {"type":"token","content":" world"}' in events[1]
    assert '"type":"done"' in events[2]
    assert str(conversation.id) in events[2]

    # Verify database persistence: User message + EXACTLY ONE assistant message containing accumulated content
    messages = await repo.get_conversation_messages(conversation.id)
    assert len(messages) == 2
    assert messages[0].role == MessageRole.USER
    assert messages[0].content == "Hi"
    assert messages[1].role == MessageRole.ASSISTANT
    assert messages[1].content == "Hello world"


@pytest.mark.asyncio
async def test_chat_service_streaming_failure_preserves_user_message_and_skips_assistant(
    db_session: AsyncSession,
) -> None:
    """Verify streaming failure yields error event, preserves user message, and persists NO assistant message."""
    repo: ConversationRepository = SQLAlchemyConversationRepository(session=db_session)
    conversation = await repo.create_conversation()

    async def mock_failed_stream_gen(*args, **kwargs):
        yield LLMStreamChunk(content="Partial text")
        raise LLMAuthenticationException("LLM provider authentication failed. Please check configured API key.")

    mock_llm_client = MagicMock(spec=LLMClient)
    mock_llm_client.stream_response = MagicMock(side_effect=mock_failed_stream_gen)

    service = ChatService(llm_client=mock_llm_client, conversation_repo=repo)
    events = [
        event
        async for event in service.process_chat_stream(
            ChatRequest(conversation_id=conversation.id, message="Failing stream prompt")
        )
    ]

    assert len(events) == 2
    assert 'data: {"type":"token","content":"Partial text"}' in events[0]
    assert '"type":"error"' in events[1]
    assert "LLM provider authentication failed" in events[1]

    # Verify DB persistence: User message is preserved, but NO assistant message was saved
    messages = await repo.get_conversation_messages(conversation.id)
    assert len(messages) == 1
    assert messages[0].role == MessageRole.USER
    assert messages[0].content == "Failing stream prompt"


@pytest.mark.asyncio
async def test_chat_service_streaming_503_connection_error_emits_sse_error(
    db_session: AsyncSession,
) -> None:
    """Verify 503/connection error during streaming emits clean user-friendly error and creates no assistant message."""
    repo: ConversationRepository = SQLAlchemyConversationRepository(session=db_session)
    conversation = await repo.create_conversation()

    async def mock_503_stream_gen(*args, **kwargs):
        raise LLMConnectionException("AI service is temporarily unavailable. Please try again.")
        yield LLMStreamChunk(content="Never reached")  # type: ignore

    mock_llm_client = MagicMock(spec=LLMClient)
    mock_llm_client.stream_response = MagicMock(side_effect=mock_503_stream_gen)

    service = ChatService(llm_client=mock_llm_client, conversation_repo=repo)
    events = [
        event
        async for event in service.process_chat_stream(
            ChatRequest(conversation_id=conversation.id, message="503 stream prompt")
        )
    ]

    assert len(events) == 1
    assert '"type":"error"' in events[0]
    assert "AI service is temporarily unavailable. Please try again." in events[0]

    # Verify DB persistence: User message is preserved, but NO assistant message was saved
    messages = await repo.get_conversation_messages(conversation.id)
    assert len(messages) == 1
    assert messages[0].role == MessageRole.USER
    assert messages[0].content == "503 stream prompt"


@pytest.mark.asyncio
async def test_chat_service_streaming_mid_stream_chunk_timeout_emits_error_and_does_not_persist_assistant(
    db_session: AsyncSession,
) -> None:
    """Verify mid-stream timeout emits token chunks then error event, and does NOT persist partial assistant message."""
    from personal_ai.llm.exceptions import LLMTimeoutException

    repo: ConversationRepository = SQLAlchemyConversationRepository(session=db_session)
    conversation = await repo.create_conversation()

    async def mock_partial_then_timeout(*args, **kwargs):
        yield LLMStreamChunk(content="Partial before timeout")
        raise LLMTimeoutException("AI service is temporarily unavailable. Please try again.")

    mock_llm_client = MagicMock(spec=LLMClient)
    mock_llm_client.stream_response = MagicMock(side_effect=mock_partial_then_timeout)

    service = ChatService(llm_client=mock_llm_client, conversation_repo=repo)
    events = [
        event
        async for event in service.process_chat_stream(
            ChatRequest(conversation_id=conversation.id, message="Timeout prompt")
        )
    ]

    assert len(events) == 2
    assert 'data: {"type":"token","content":"Partial before timeout"}' in events[0]
    assert '"type":"error"' in events[1]
    assert "AI service is temporarily unavailable. Please try again." in events[1]

    # Verify DB persistence: User message is preserved, but NO assistant message was saved
    messages = await repo.get_conversation_messages(conversation.id)
    assert len(messages) == 1
    assert messages[0].role == MessageRole.USER
    assert messages[0].content == "Timeout prompt"
