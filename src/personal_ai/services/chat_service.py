import asyncio
from typing import AsyncGenerator, List, Optional
import uuid

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from personal_ai.application.experience import (
    AIExperiencePromotionStrategy,
    ExperienceClassifier,
    ExperiencePromotionService,
    RecordExperience,
)
from personal_ai.core.exceptions import AppException
from personal_ai.core.logger import get_logger
from personal_ai.db.models import Message
from personal_ai.db.repositories import (
    SQLAlchemyExperienceClassificationRepository,
    SQLAlchemyExperienceRepository,
)
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
    """Business logic service for managing chat processing, conversation memory, and user isolation.

    Depends exclusively on abstract interfaces.
    Enforces conversation ownership and background asynchronous experience classification.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        conversation_repo: ConversationRepository,
        experience_promotion_service: Optional[ExperiencePromotionService] = None,
    ) -> None:
        """Initialize ChatService with abstract dependencies and optional promotion service.

        Args:
            llm_client: Abstract LLM client interface.
            conversation_repo: Abstract conversation repository interface.
            experience_promotion_service: Optional application service for Experience promotion.
        """
        self._llm_client = llm_client
        self._conversation_repo = conversation_repo
        self._experience_promotion_service = experience_promotion_service

    async def process_chat(
        self,
        request: ChatRequest,
        user_id: Optional[uuid.UUID] = None,
    ) -> ChatResponse:
        """Process an incoming chat request, maintaining conversation context synchronously."""
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
        else:
            conversation = await self._conversation_repo.create_conversation(user_id=user_id)
            logger.info(
                "Created new conversation [conversation_id=%s, user_id=%s]",
                conversation.id,
                user_id,
            )

        # 2. Retrieve previous conversation messages in chronological order
        stored_messages = await self._conversation_repo.get_conversation_messages(
            conversation.id
        )

        # 3. Convert stored messages to domain LLMMessage representation
        history: List[LLMMessage] = []

        if request.system_prompt:
            history.append(
                LLMMessage(role="system", content=request.system_prompt)
            )

        for msg in stored_messages:
            history.append(
                LLMMessage(
                    role=msg.role.value if hasattr(msg.role, "value") else str(msg.role),
                    content=msg.content,
                )
            )

        history.append(LLMMessage(role="user", content=request.message))

        # 4. Persist user message to conversation history
        user_message = await self._conversation_repo.add_message(
            conversation_id=conversation.id,
            role="user",
            content=request.message,
        )

        # 5. Experience classification execution (B1 & B2)
        if self._experience_promotion_service:
            await self._safe_bg_experience_promotion(user_message, user_id=user_id)

        # 6. Execute LLM completion request
        llm_response = await self._llm_client.generate_response(messages=history)

        # 7. Persist assistant response
        await self._conversation_repo.add_message(
            conversation_id=conversation.id,
            role="assistant",
            content=llm_response.content,
        )

        return ChatResponse(
            conversation_id=conversation.id,
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
        """Process an incoming chat request using Server-Sent Events (SSE) streaming.

        Execution Strategy:
        -------------------
        1. Resolve existing conversation or create new conversation matching user_id.
        2. Retrieve stored conversation messages in chronological order.
        3. Persist user prompt message to database history first.
        4. DECOUPLED CLASSIFICATION: Schedule background task for AI Experience classification.
           Does NOT block or delay the first streamed token!
        5. CALL STREAM: Invoke LLMClient.stream_response(messages) and stream tokens.
        6. PERSIST ASSISTANT RESPONSE: Save accumulated assistant text upon stream completion.
        """
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
        else:
            conversation = await self._conversation_repo.create_conversation(user_id=user_id)
            logger.info(
                "Created new conversation thread for streaming [conversation_id=%s, user_id=%s]",
                conversation.id,
                user_id,
            )

        # 2. Retrieve previous conversation messages in chronological order
        stored_messages = await self._conversation_repo.get_conversation_messages(
            conversation.id
        )

        # 3. Convert stored messages to domain LLMMessage representation
        history: List[LLMMessage] = []

        if request.system_prompt:
            history.append(
                LLMMessage(role="system", content=request.system_prompt)
            )

        for msg in stored_messages:
            history.append(
                LLMMessage(
                    role=msg.role.value if hasattr(msg.role, "value") else str(msg.role),
                    content=msg.content,
                )
            )

        history.append(LLMMessage(role="user", content=request.message))

        # 4. Persist user message to conversation history
        user_message = await self._conversation_repo.add_message(
            conversation_id=conversation.id,
            role="user",
            content=request.message,
        )

        # 5. Non-blocking asynchronous Experience classification (B1 & B2)
        # Background task runs concurrently without delaying the first token!
        if self._experience_promotion_service:
            asyncio.create_task(
                self._safe_bg_experience_promotion(user_message, user_id=user_id)
            )

        accumulated_chunks: List[str] = []

        # 6. Stream tokens directly from LLM
        try:
            stream_gen = self._llm_client.stream_response(messages=history)
            async for chunk in stream_gen:
                if chunk.content:
                    accumulated_chunks.append(chunk.content)
                    yield ChatStreamEvent(
                        type=StreamEventType.TOKEN,
                        content=chunk.content,
                    ).to_sse()

            # Stream completed successfully: persist exactly ONE assistant message
            full_response_text = "".join(accumulated_chunks)
            await self._conversation_repo.add_message(
                conversation_id=conversation.id,
                role="assistant",
                content=full_response_text,
            )

            # Emit done event containing conversation_id
            yield ChatStreamEvent(
                type=StreamEventType.DONE,
                conversation_id=conversation.id,
            ).to_sse()

        except LLMAuthenticationException as exc:
            logger.error("LLM authentication failed during stream [conversation_id=%s]: %s", conversation.id, exc)
            yield ChatStreamEvent(
                type=StreamEventType.ERROR,
                message=exc.message,
                conversation_id=conversation.id,
            ).to_sse()

        except LLMRateLimitException as exc:
            logger.error("LLM rate limit exceeded during stream [conversation_id=%s]: %s", conversation.id, exc)
            yield ChatStreamEvent(
                type=StreamEventType.ERROR,
                message=exc.message,
                conversation_id=conversation.id,
            ).to_sse()

        except LLMConnectionException as exc:
            logger.error("LLM connection error during stream [conversation_id=%s]: %s", conversation.id, exc)
            yield ChatStreamEvent(
                type=StreamEventType.ERROR,
                message=exc.message,
                conversation_id=conversation.id,
            ).to_sse()

        except LLMException as exc:
            logger.error("LLM domain exception during stream [conversation_id=%s]: %s", conversation.id, exc)
            yield ChatStreamEvent(
                type=StreamEventType.ERROR,
                message=exc.message,
                conversation_id=conversation.id,
            ).to_sse()

        except asyncio.CancelledError:
            logger.warning("Chat stream cancelled by client [conversation_id=%s]", conversation.id)
            raise

        except Exception as exc:
            logger.error("Unexpected runtime error during chat stream [conversation_id=%s]: %s", conversation.id, exc)
            yield ChatStreamEvent(
                type=StreamEventType.ERROR,
                message="An unexpected error occurred during chat streaming.",
                conversation_id=conversation.id,
            ).to_sse()

    async def _safe_bg_experience_promotion(
        self,
        message: Message,
        user_id: Optional[uuid.UUID] = None,
    ) -> None:
        """Safely execute background Experience promotion in an isolated DB session (B1 & B2)."""
        if not self._experience_promotion_service:
            return

        try:
            # Dynamically inspect active AsyncEngine from conversation_repo session
            session = getattr(self._conversation_repo, "_session", None)
            bind = getattr(session, "bind", None) if session else None

            if bind:
                async with AsyncSession(bind=bind, expire_on_commit=False) as bg_session:
                    exp_repo = SQLAlchemyExperienceRepository(session=bg_session)
                    record_exp = RecordExperience(repository=exp_repo)
                    classification_repo = SQLAlchemyExperienceClassificationRepository(session=bg_session)
                    classifier = ExperienceClassifier(llm_client=self._llm_client)

                    strategy = getattr(self._experience_promotion_service, "_strategy", None)
                    if not strategy or hasattr(strategy, "_classifier"):
                        strategy = AIExperiencePromotionStrategy(
                            classifier=classifier,
                            classification_repo=classification_repo,
                        )

                    bg_service = ExperiencePromotionService(
                        record_experience=record_exp,
                        strategy=strategy,
                        experience_repo=exp_repo,
                    )
                    res = await bg_service.promote_message(message=message, user_id=user_id)
                    if res.promoted:
                        logger.info(
                            "Background experience promoted successfully [message_id=%s, experience_id=%s, user_id=%s]",
                            message.id,
                            res.experience_id,
                            user_id,
                        )
                    return

            res = await self._experience_promotion_service.promote_message(
                message=message,
                user_id=user_id,
            )
            if res.promoted:
                logger.info(
                    "Background experience promoted successfully [message_id=%s, experience_id=%s, user_id=%s]",
                    message.id,
                    res.experience_id,
                    user_id,
                )
        except Exception as exc:
            logger.error(
                "Background experience promotion failed safely [message_id=%s, user_id=%s]: %s",
                message.id,
                user_id,
                exc,
            )
