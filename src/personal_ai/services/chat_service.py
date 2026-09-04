import asyncio
from typing import Any, AsyncGenerator, List, Optional, Union
import uuid

from personal_ai.application.experience.background_processor import (
    BackgroundExperienceProcessor,
)
from personal_ai.application.memory import (
    MemoryContextBuilder,
    MemoryRetrievalService,
    PersonalContextBuilder,
    PersonalContextRetrievalService,
)
from personal_ai.config.settings import get_settings
from personal_ai.core.exceptions import AppException
from personal_ai.core.logger import get_logger
from personal_ai.db.repositories.base import ConversationRepository
from personal_ai.llm.client import LLMClient
from personal_ai.llm.exceptions import (
    LLMAuthenticationException,
    LLMConnectionException,
    LLMException,
    LLMRateLimitException,
)
from personal_ai.llm.models import LLMMessage
from personal_ai.models.chat import (
    ChatRequest,
    ChatResponse,
    ChatStreamEvent,
    StreamEventType,
)

logger = get_logger(__name__)


class ChatService:
    """Business logic service managing chat completion, personal context retrieval augmentation, and history.

    Depends exclusively on abstract interfaces.
    Orchestrates personal context retrieval safely without failing chat upon retrieval issues.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        conversation_repo: ConversationRepository,
        bg_processor: Optional[BackgroundExperienceProcessor] = None,
        retrieval_service: Optional[MemoryRetrievalService] = None,
        personal_context_service: Optional[PersonalContextRetrievalService] = None,
        context_builder: Optional[Union[PersonalContextBuilder, MemoryContextBuilder]] = None,
    ) -> None:
        """Initialize ChatService with dependencies.

        Args:
            llm_client: Abstract LLM client interface.
            conversation_repo: Abstract conversation repository interface.
            bg_processor: Abstract background Experience processor interface.
            retrieval_service: Optional legacy MemoryRetrievalService.
            personal_context_service: Optional PersonalContextRetrievalService for dimension-aware context.
            context_builder: Optional context builder for formatting context into prompts.
        """
        self._llm_client = llm_client
        self._conversation_repo = conversation_repo
        self._bg_processor = bg_processor
        self._retrieval_service = retrieval_service
        self._personal_context_service = personal_context_service
        self._context_builder = context_builder or PersonalContextBuilder()

    async def _retrieve_memory_context(
        self,
        user_id: Optional[uuid.UUID],
        query: str,
        conversation_context: Optional[List[LLMMessage]] = None,
    ) -> Optional[str]:
        """Perform fail-safe user-scoped personal context retrieval and format into structured prompt context."""
        settings = get_settings()
        if not user_id or not settings.memory_retrieval_enabled:
            return None

        try:
            if self._personal_context_service:
                logger.info("Starting personal context retrieval for chat [user_id=%s]", user_id)
                personal_context = await self._personal_context_service.retrieve_context(
                    user_id=user_id,
                    query=query,
                    conversation_context=conversation_context,
                )
                logger.info(
                    "Personal context retrieval completed [user_id=%s, count=%d]",
                    user_id,
                    len(personal_context.items),
                )
                if not personal_context.is_empty:
                    return self._context_builder.build_context(personal_context)
                return None
            elif self._retrieval_service:
                logger.info("Starting memory retrieval for chat [user_id=%s]", user_id)
                memories = await self._retrieval_service.search(
                    user_id=user_id,
                    query=query,
                    limit=settings.memory_retrieval_limit,
                )
                logger.info(
                    "Memory retrieval for chat completed [user_id=%s, count=%d]",
                    user_id,
                    len(memories),
                )
                if memories:
                    return self._context_builder.build_context(memories)
                return None
            return None
        except Exception as exc:
            logger.warning(
                "Personal context retrieval failed safely, proceeding with unaugmented chat [user_id=%s]: %s",
                user_id,
                exc,
            )
            return None

    def _build_llm_messages(
        self,
        stored_messages: List[Any],
        current_message: str,
        system_prompt: Optional[str] = None,
        memory_context: Optional[str] = None,
    ) -> List[LLMMessage]:
        """Build the full LLM message history: System Instructions + Memory Context + History + Current Message."""
        messages: List[LLMMessage] = []

        # 1. System instructions + Memory context
        system_parts: List[str] = []
        if system_prompt and system_prompt.strip():
            system_parts.append(system_prompt.strip())
        if memory_context and memory_context.strip():
            system_parts.append(memory_context.strip())

        if system_parts:
            messages.append(
                LLMMessage(role="system", content="\n\n".join(system_parts))
            )

        # 2. Conversation history (short-term)
        for msg in stored_messages:
            messages.append(
                LLMMessage(
                    role=msg.role.value if hasattr(msg.role, "value") else str(msg.role),
                    content=msg.content,
                )
            )

        # 3. Current user message
        messages.append(LLMMessage(role="user", content=current_message))

        return messages

    async def process_chat(
        self,
        request: ChatRequest,
        user_id: Optional[uuid.UUID] = None,
    ) -> ChatResponse:
        """Process an incoming chat request, augmenting context with retrieved personal memories synchronously."""
        # 1. Resolve or create conversation thread with user ownership validation
        if request.conversation_id:
            conversation = await self._conversation_repo.get_conversation(
                request.conversation_id,
                user_id=user_id,
            )
            if not conversation:
                logger.warning(
                    "Requested conversation not found or access denied [conversation_id=%s, user_id=%s]",
                    request.conversation_id,
                    user_id,
                )
                raise AppException(
                    message=f"Conversation '{request.conversation_id}' not found.",
                    status_code=404,
                )
            conv_id = conversation.id
        else:
            conversation = await self._conversation_repo.create_conversation(user_id=user_id)
            conv_id = conversation.id
            logger.info(
                "Created new conversation [conversation_id=%s, user_id=%s]",
                conv_id,
                user_id,
            )

        # 2. Retrieve previous conversation messages in chronological order
        stored_messages = await self._conversation_repo.get_conversation_messages(conv_id)
        conv_llm_messages = [
            LLMMessage(
                role=msg.role.value if hasattr(msg.role, "value") else str(msg.role),
                content=msg.content,
            )
            for msg in stored_messages
        ]

        # 3. Retrieve relevant personal context memories (fail-safe enhancement)
        memory_context = await self._retrieve_memory_context(
            user_id=user_id,
            query=request.message,
            conversation_context=conv_llm_messages,
        )

        # 4. Construct complete LLM messages structure
        history = self._build_llm_messages(
            stored_messages=stored_messages,
            current_message=request.message,
            system_prompt=request.system_prompt,
            memory_context=memory_context,
        )

        # 5. Persist user message to conversation history
        user_message = await self._conversation_repo.add_message(
            conversation_id=conv_id,
            role="user",
            content=request.message,
        )

        # 6. Non-blocking asynchronous Experience classification
        if self._bg_processor:
            asyncio.create_task(
                self._bg_processor.process_background_promotion(user_message, user_id=user_id)
            )

        # 7. Execute LLM completion request
        llm_response = await self._llm_client.generate_response(messages=history)

        # 8. Persist assistant response
        await self._conversation_repo.add_message(
            conversation_id=conv_id,
            role="assistant",
            content=llm_response.content,
        )

        return ChatResponse(
            conversation_id=conv_id,
            response=llm_response.content,
            provider=llm_response.provider,
            model=llm_response.model,
            latency_ms=llm_response.latency_ms,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            total_tokens=llm_response.total_tokens,
        )

    async def process_chat_stream(
        self,
        request: ChatRequest,
        user_id: Optional[uuid.UUID] = None,
    ) -> AsyncGenerator[str, None]:
        """Process an incoming chat request using Server-Sent Events (SSE) streaming with memory augmentation."""
        # 1. Resolve or create conversation thread with strict user_id ownership check
        if request.conversation_id:
            conversation = await self._conversation_repo.get_conversation(
                request.conversation_id,
                user_id=user_id,
            )
            if not conversation:
                logger.warning(
                    "Requested conversation not found for streaming [conversation_id=%s, user_id=%s]",
                    request.conversation_id,
                    user_id,
                )
                yield ChatStreamEvent(
                    type=StreamEventType.ERROR,
                    message=f"Conversation '{request.conversation_id}' not found.",
                ).to_sse()
                return
            conv_id = conversation.id
        else:
            conversation = await self._conversation_repo.create_conversation(user_id=user_id)
            conv_id = conversation.id
            logger.info(
                "Created new conversation thread for streaming [conversation_id=%s, user_id=%s]",
                conv_id,
                user_id,
            )

        # 2. Retrieve previous conversation messages in chronological order
        stored_messages = await self._conversation_repo.get_conversation_messages(conv_id)
        conv_llm_messages = [
            LLMMessage(
                role=msg.role.value if hasattr(msg.role, "value") else str(msg.role),
                content=msg.content,
            )
            for msg in stored_messages
        ]

        # 3. Retrieve relevant personal context memories (fail-safe enhancement)
        memory_context = await self._retrieve_memory_context(
            user_id=user_id,
            query=request.message,
            conversation_context=conv_llm_messages,
        )

        # 4. Construct complete LLM messages structure
        history = self._build_llm_messages(
            stored_messages=stored_messages,
            current_message=request.message,
            system_prompt=request.system_prompt,
            memory_context=memory_context,
        )

        # 5. Persist user message to conversation history
        user_message = await self._conversation_repo.add_message(
            conversation_id=conv_id,
            role="user",
            content=request.message,
        )

        # 6. Non-blocking asynchronous Experience classification
        if self._bg_processor:
            asyncio.create_task(
                self._bg_processor.process_background_promotion(user_message, user_id=user_id)
            )

        accumulated_chunks: List[str] = []

        # 7. Stream tokens directly from LLM
        try:
            stream_gen = self._llm_client.stream_response(messages=history)
            async for chunk in stream_gen:
                if chunk.content:
                    accumulated_chunks.append(chunk.content)
                    yield ChatStreamEvent(
                        type=StreamEventType.TOKEN,
                        content=chunk.content,
                    ).to_sse()

            # Stream completed successfully: persist exactly ONE assistant message if non-empty
            full_response_text = "".join(accumulated_chunks)
            if full_response_text.strip():
                await self._conversation_repo.add_message(
                    conversation_id=conv_id,
                    role="assistant",
                    content=full_response_text,
                )

            # Emit done event containing conversation_id
            yield ChatStreamEvent(
                type=StreamEventType.DONE,
                conversation_id=conv_id,
            ).to_sse()

        except LLMAuthenticationException as exc:
            logger.error("LLM authentication failed during stream [conversation_id=%s]: %s", conv_id, exc)
            yield ChatStreamEvent(
                type=StreamEventType.ERROR,
                message=exc.message,
                conversation_id=conv_id,
            ).to_sse()

        except LLMRateLimitException as exc:
            logger.error("LLM rate limit exceeded during stream [conversation_id=%s]: %s", conv_id, exc)
            yield ChatStreamEvent(
                type=StreamEventType.ERROR,
                message=exc.message,
                conversation_id=conv_id,
            ).to_sse()

        except LLMConnectionException as exc:
            logger.error("LLM connection error during stream [conversation_id=%s]: %s", conv_id, exc)
            yield ChatStreamEvent(
                type=StreamEventType.ERROR,
                message=exc.message,
                conversation_id=conv_id,
            ).to_sse()

        except LLMException as exc:
            logger.error("LLM domain exception during stream [conversation_id=%s]: %s", conv_id, exc)
            yield ChatStreamEvent(
                type=StreamEventType.ERROR,
                message=exc.message,
                conversation_id=conv_id,
            ).to_sse()

        except asyncio.CancelledError:
            logger.warning("Chat stream cancelled by client [conversation_id=%s]", conv_id)
            raise

        except Exception as exc:
            logger.error("Unexpected runtime error during chat stream [conversation_id=%s]: %s", conv_id, exc)
            yield ChatStreamEvent(
                type=StreamEventType.ERROR,
                message="AI service is temporarily unavailable. Please try again.",
                conversation_id=conv_id,
            ).to_sse()
