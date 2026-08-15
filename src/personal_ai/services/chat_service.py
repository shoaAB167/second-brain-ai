import asyncio
from typing import AsyncGenerator, List

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
    """Business logic service for managing chat processing and conversation memory.

    Depends exclusively on LLMClient and ConversationRepository abstract interfaces.
    Does NOT depend on LiteLLM, SQLAlchemy, or concrete database implementations.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        conversation_repo: ConversationRepository,
    ) -> None:
        """Initialize ChatService with abstract dependencies.

        Args:
            llm_client: Abstract LLM client interface.
            conversation_repo: Abstract conversation repository interface.
        """
        self._llm_client = llm_client
        self._conversation_repo = conversation_repo

    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        """Process an incoming chat request, maintaining conversation context synchronously."""
        # 1. Resolve or create conversation thread
        if request.conversation_id:
            conversation = await self._conversation_repo.get_conversation(
                request.conversation_id
            )
            if not conversation:
                logger.warning(
                    "Requested conversation not found [conversation_id=%s]",
                    request.conversation_id,
                )
                raise AppException(
                    message=f"Conversation '{request.conversation_id}' not found.",
                    status_code=404,
                )
        else:
            conversation = await self._conversation_repo.create_conversation()
            logger.info(
                "Created new conversation [conversation_id=%s]",
                conversation.id,
            )

        # 2. Retrieve previous conversation messages in chronological order (oldest -> newest)
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

        # Append current user prompt message to history payload
        history.append(LLMMessage(role="user", content=request.message))

        # 4. Persist user message to conversation history before calling LLM
        await self._conversation_repo.add_message(
            conversation_id=conversation.id,
            role="user",
            content=request.message,
        )

        # 5. Execute LLM completion request with complete conversation context
        llm_response = await self._llm_client.generate_response(messages=history)

        # 6. Persist assistant response to conversation history after successful LLM call
        await self._conversation_repo.add_message(
            conversation_id=conversation.id,
            role="assistant",
            content=llm_response.content,
        )

        # 7. Return formatted ChatResponse containing conversation_id
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
        self, request: ChatRequest
    ) -> AsyncGenerator[str, None]:
        """Process an incoming chat request using Server-Sent Events (SSE) streaming.

        Streaming Execution Strategy:
        ------------------------------
        1. Resolve existing conversation or create a new conversation thread.
        2. Retrieve stored conversation messages in chronological order.
        3. Convert stored messages to domain LLMMessage format and append new user prompt.
        4. PERSIST USER MESSAGE: The user's prompt is committed to database history first.
        5. CALL STREAM: Invoke LLMClient.stream_response(messages).
        6. YIELD TOKENS: Stream partial token chunks to client as SSE data payloads as they arrive.
        7. ACCUMULATE RESPONSE: Collect partial token strings internally in memory.
        8. PERSIST ASSISTANT RESPONSE: Upon successful completion of stream, commit EXACTLY ONE
           assistant message with the full accumulated content to database history.
        9. FALLBACK ON FAILURE OR CANCELLATION:
           - User message remains recorded in database history.
           - NO assistant message is saved.
           - Emits sanitized SSE error payload for domain/runtime exceptions.
        """
        # 1. Resolve or create conversation thread
        if request.conversation_id:
            conversation = await self._conversation_repo.get_conversation(
                request.conversation_id
            )
            if not conversation:
                logger.warning(
                    "Requested conversation not found for streaming [conversation_id=%s]",
                    request.conversation_id,
                )
                yield ChatStreamEvent(
                    type=StreamEventType.ERROR,
                    message=f"Conversation '{request.conversation_id}' not found.",
                ).to_sse()
                return
        else:
            conversation = await self._conversation_repo.create_conversation()
            logger.info(
                "Created new conversation thread for streaming [conversation_id=%s]",
                conversation.id,
            )

        # 2. Retrieve previous conversation messages in chronological order (oldest -> newest)
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

        # Append current user prompt message to history payload
        history.append(LLMMessage(role="user", content=request.message))

        # 4. Persist user message to conversation history before invoking LLM stream
        await self._conversation_repo.add_message(
            conversation_id=conversation.id,
            role="user",
            content=request.message,
        )

        accumulated_chunks: List[str] = []

        # 5. Call LLMClient.stream_response() and stream tokens
        try:
            stream_gen = self._llm_client.stream_response(messages=history)
            async for chunk in stream_gen:
                if chunk.content:
                    accumulated_chunks.append(chunk.content)
                    yield ChatStreamEvent(
                        type=StreamEventType.TOKEN,
                        content=chunk.content,
                    ).to_sse()

            # 6. Stream completed successfully: persist exactly ONE assistant message
            full_response_text = "".join(accumulated_chunks)
            await self._conversation_repo.add_message(
                conversation_id=conversation.id,
                role="assistant",
                content=full_response_text,
            )

            # 7. Emit done event containing conversation_id
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
            # Re-raise cancellation without persisting assistant message
            raise

        except Exception as exc:
            logger.error("Unexpected runtime error during chat stream [conversation_id=%s]: %s", conversation.id, exc)
            yield ChatStreamEvent(
                type=StreamEventType.ERROR,
                message="An unexpected error occurred during chat streaming.",
                conversation_id=conversation.id,
            ).to_sse()
