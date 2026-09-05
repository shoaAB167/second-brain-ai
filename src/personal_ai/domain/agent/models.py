from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from personal_ai.domain.experience import PersonalContext
from personal_ai.llm.models import LLMMessage, LLMResponse


class ResponseMode(str, Enum):
    """Supported agent response modes."""

    DIRECT_ANSWER = "DIRECT_ANSWER"
    PERSONALIZED_RESPONSE = "PERSONALIZED_RESPONSE"
    CLARIFICATION = "CLARIFICATION"
    EMOTIONAL_SUPPORT = "EMOTIONAL_SUPPORT"
    DECISION_SUPPORT = "DECISION_SUPPORT"
    GENERAL_GUIDANCE = "GENERAL_GUIDANCE"


@dataclass(frozen=True)
class AgentRequest:
    """Domain request container passed to PersonalAgent for orchestration."""

    current_message: str
    user_id: Optional[uuid.UUID] = None
    conversation_history: List[LLMMessage] = field(default_factory=list)
    personal_context: Optional[PersonalContext] = None
    system_prompt: Optional[str] = None


@dataclass(frozen=True)
class AgentDecision:
    """Domain container representing the agent's decision and generated response."""

    response_mode: ResponseMode
    content: str
    provider: Optional[str] = None
    model: Optional[str] = None
    latency_ms: float = 0.0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    raw_response: Optional[LLMResponse] = None
