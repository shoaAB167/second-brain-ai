import asyncio
from typing import Any, AsyncGenerator, List, Optional, Union
import uuid

from personal_ai.agents.personal_agent import PersonalAgent
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
from personal_ai.domain.agent import AgentRequest
from personal_ai.domain.experience import PersonalContext, PersonalContextItem
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
    """Business logic service managing conversation lifecycle, experience classification, and agent orchestration.

    Depends exclusively on abstract interfaces.
    Coordinates between ConversationRepository, PersonalContextRetrievalService, and PersonalAgent.
    """

    def __init__(
        self,
        conversation_repo: ConversationRepository,
        llm_client: Optional[LLMClient] = None,
        personal_agent: Optional[PersonalAgent] = None,
        bg_processor: Optional[BackgroundExperienceProcessor] = None,
        retrieval_service: Optional[MemoryRetrievalService] = None,
        personal_context_service: Optional[PersonalContextRetrievalService] = None,
        context_builder: Optional[Union[PersonalContextBuilder, MemoryContextBuilder]] = None,
    ) -> None:
        """Initialize ChatService with dependencies.

        Args:
            conversation_repo: Abstract conversation repository interface.
            llm_client: Optional abstract LLM client interface (used to construct PersonalAgent if not provided).
            personal_agent: Optional PersonalAgent orchestration layer instance.
            bg_processor: Optional abstract background Experience processor interface.
            retrieval_service: Optional legacy MemoryRetrievalService.
            personal_context_service: Optional PersonalContextRetrievalService for dimension-aware context.
            context_builder: Optional context builder for formatting context into prompts.
        """
        self._conversation_repo = conversation_repo
        self._bg_processor = bg_processor
        self._retrieval_service = retrieval_service
        self._personal_context_service = personal_context_service

        builder = context_builder if isinstance(context_builder, PersonalContextBuilder) else PersonalContextBuilder()
        self._context_builder = builder

        if personal_agent is not None:
            self._personal_agent = personal_agent
        elif llm_client is not None:
            self._personal_agent = PersonalAgent(
                llm_client=llm_client,
                context_builder=builder,
            )
        else:
            raise ValueError("Either personal_agent or llm_client must be provided to ChatService.")

        self._llm_client = llm_client or getattr(self._personal_agent, "_llm_client", None)

    async def _retrieve_personal_context(
        self,
        user_id: Optional[uuid.UUID],
        query: str,
        conversation_context: Optional[List[LLMMessage]] = None,
    ) -> Optional[PersonalContext]:
        """Perform fail-safe user-scoped personal context retrieval."""
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
                return personal_context
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
                    items = [
                        PersonalContextItem(
                            experience_id=m.experience_id,
                            content=m.content,
                            type=m.type,
                            domain=m.domain,
                            score=m.similarity,
                            similarity=m.similarity,
                            created_at=m.created_at,
                        )
                        for m in memories
                    ]
                    return PersonalContext(
                        user_id=user_id,
                        query=query,
                        items=items,
                        total_candidates=len(items),
                    )
                return None
            return None
        except Exception as exc:
            logger.warning(
                "Personal context retrieval failed safely, proceeding with unaugmented chat [user_id=%s]: %s",
                user_id,
                exc,
            )
            return None

    async def process_chat(
        self,
        request: ChatRequest,
        user_id: Optional[uuid.UUID] = None,
    ) -> ChatResponse:
        """Process an incoming chat request through PersonalAgent orchestration."""
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
        personal_context = await self._retrieve_personal_context(
            user_id=user_id,
            query=request.message,
            conversation_context=conv_llm_messages,
        )

        # 4. Construct AgentRequest container
        agent_request = AgentRequest(
            current_message=request.message,
            user_id=user_id,
            conversation_history=conv_llm_messages,
            personal_context=personal_context,
            system_prompt=request.system_prompt,
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

        # 7. Execute PersonalAgent orchestration
        agent_decision = await self._personal_agent.generate_response(agent_request)

        # 8. Persist assistant response
        await self._conversation_repo.add_message(
            conversation_id=conv_id,
            role="assistant",
            content=agent_decision.content,
        )

        return ChatResponse(
            conversation_id=conv_id,
            response=agent_decision.content,
            provider=agent_decision.provider or "unknown",
            model=agent_decision.model or "unknown",
            latency_ms=agent_decision.latency_ms,
            prompt_tokens=agent_decision.prompt_tokens,
            completion_tokens=agent_decision.completion_tokens,
            total_tokens=agent_decision.total_tokens,
        )

    async def process_chat_stream(
        self,
        request: ChatRequest,
        user_id: Optional[uuid.UUID] = None,
    ) -> AsyncGenerator[str, None]:
        """Process an incoming chat request using Server-Sent Events (SSE) streaming through PersonalAgent."""
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
        personal_context = await self._retrieve_personal_context(
            user_id=user_id,
            query=request.message,
            conversation_context=conv_llm_messages,
        )

        # 4. Construct AgentRequest container
        agent_request = AgentRequest(
            current_message=request.message,
            user_id=user_id,
            conversation_history=conv_llm_messages,
            personal_context=personal_context,
            system_prompt=request.system_prompt,
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

        # 7. Stream tokens directly from PersonalAgent
        try:
            _, stream_gen = self._personal_agent.stream_response(agent_request)
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
