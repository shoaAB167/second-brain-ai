import uuid
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from personal_ai.api.dependencies import get_current_user_id, get_db_session, get_llm_client
from personal_ai.api.routers.memories import get_memory_retrieval_service
from personal_ai.application.experience import BackgroundExperienceProcessor
from personal_ai.application.memory import MemoryRetrievalService
from personal_ai.db.repositories import (
    ConversationRepository,
    SQLAlchemyConversationRepository,
)
from personal_ai.db.session import AsyncSessionFactory
from personal_ai.infrastructure.experience import SQLAlchemyBackgroundExperienceProcessor
from personal_ai.llm import LLMClient
from personal_ai.models.chat import ChatRequest, ChatResponse
from personal_ai.services.chat_service import ChatService

router = APIRouter()


def get_conversation_repository(
    session: AsyncSession = Depends(get_db_session),
) -> ConversationRepository:
    """Dependency provider constructing concrete SQLAlchemyConversationRepository as ConversationRepository abstraction."""
    return SQLAlchemyConversationRepository(session=session)


def get_background_experience_processor(
    llm_client: LLMClient = Depends(get_llm_client),
) -> BackgroundExperienceProcessor:
    """Dependency provider constructing SQLAlchemyBackgroundExperienceProcessor abstraction."""
    return SQLAlchemyBackgroundExperienceProcessor(
        session_factory=AsyncSessionFactory,
        llm_client=llm_client,
    )


def get_chat_service(
    llm_client: LLMClient = Depends(get_llm_client),
    conversation_repo: ConversationRepository = Depends(get_conversation_repository),
    bg_processor: BackgroundExperienceProcessor = Depends(get_background_experience_processor),
    retrieval_service: MemoryRetrievalService = Depends(get_memory_retrieval_service),
) -> ChatService:
    """Dependency provider for ChatService, injecting repository, LLM, background processor, and retrieval service."""
    return ChatService(
        llm_client=llm_client,
        conversation_repo=conversation_repo,
        bg_processor=bg_processor,
        retrieval_service=retrieval_service,
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send Chat Message",
    description="Processes a prompt message with conversation history synchronously for the authenticated user.",
)
async def chat(
    request: ChatRequest,
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Execute synchronous chat completion request through ChatService."""
    return await chat_service.process_chat(request, user_id=current_user_id)


@router.post(
    "/chat/stream",
    summary="Send Chat Message (Streaming)",
    description="Streams chat completions using Server-Sent Events (SSE) in real time for the authenticated user.",
)
async def chat_stream(
    request: ChatRequest,
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """Execute streaming chat completion request returning Server-Sent Events (SSE)."""
    return StreamingResponse(
        chat_service.process_chat_stream(request, user_id=current_user_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
