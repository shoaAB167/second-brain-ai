import json
import time
from typing import Any, Dict, List, Optional
from pydantic import ValidationError

from personal_ai.config.settings import get_settings
from personal_ai.core.logger import get_logger
from personal_ai.domain.experience import ClassificationResult
from personal_ai.llm.client import LLMClient
from personal_ai.llm.exceptions import LLMException
from personal_ai.llm.models import LLMMessage
from personal_ai.prompts import EXPERIENCE_CLASSIFIER_SYSTEM_PROMPT

logger = get_logger(__name__)


class ExperienceClassifier:
    """AI-powered Experience classifier depending on abstract LLMClient interface.

    Classifies user message content (with optional bounded conversation context)
    into structured ClassificationResult metrics without modifying or writing to the database directly.
    """

    def __init__(self, llm_client: LLMClient, model: Optional[str] = None) -> None:
        """Initialize classifier with abstract LLMClient and optional model override.

        Args:
            llm_client: Abstract LLM client interface.
            model: Optional model identifier override (defaults to Settings.experience_classifier_model).
        """
        settings = get_settings()
        self._llm_client = llm_client
        self._model = model or settings.experience_classifier_model

    async def classify(
        self,
        content: str,
        conversation_context: Optional[List[LLMMessage]] = None,
    ) -> ClassificationResult:
        """Classify a user message into structured ClassificationResult metrics.

        Args:
            content: Raw text content of the user message.
            conversation_context: Optional list of recent LLMMessage objects for contextual reference resolution.

        Returns:
            ClassificationResult: Structured classification result object.
        """
        if not content or not content.strip():
            return ClassificationResult(
                is_experience=False,
                type=None,
                importance=0.0,
                confidence=1.0,
                reasoning="Empty message content provided.",
                raw_model="none",
            )

        start_time = time.perf_counter()

        prompt_body = ""
        if conversation_context:
            context_lines = []
            for msg in conversation_context:
                role_label = msg.role.capitalize() if hasattr(msg.role, "capitalize") else str(msg.role)
                context_lines.append(f"{role_label}: \"{msg.content}\"")
            context_str = "\n".join(context_lines)
            prompt_body += (
                f"PRIOR CONVERSATION CONTEXT (FOR REFERENCE RESOLUTION ONLY):\n{context_str}\n\n"
                "IMPORTANT: Evaluate ONLY the target user message for experience extraction. "
                "Do NOT treat assistant responses as user experiences.\n\n"
            )

        prompt_body += f"TARGET USER MESSAGE TO CLASSIFY:\n\"\"\"{content}\"\"\""

        messages = [
            LLMMessage(role="system", content=EXPERIENCE_CLASSIFIER_SYSTEM_PROMPT),
            LLMMessage(role="user", content=prompt_body),
        ]

        kwargs: Dict[str, Any] = {"response_format": {"type": "json_object"}}
        if self._model:
            kwargs["model"] = self._model

        try:
            llm_response = await self._llm_client.generate_response(
                messages=messages,
                **kwargs,
            )

            raw_text = llm_response.content.strip()

            # Clean potential markdown block formatting if provider returns json in code blocks
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()

            data: Dict[str, Any] = json.loads(raw_text)

            # Construct and validate ClassificationResult
            result = ClassificationResult(
                is_experience=bool(data.get("is_experience", False)),
                type=data.get("type"),
                importance=float(data.get("importance", 0.0)),
                confidence=float(data.get("confidence", 0.0)),
                reasoning=data.get("reasoning"),
                raw_model=llm_response.model,
            )

            duration_ms = (time.perf_counter() - start_time) * 1000.0

            logger.info(
                "Experience classified [is_exp=%s, type=%s, importance=%.2f, confidence=%.2f, model=%s, duration_ms=%.1f]",
                result.is_experience,
                result.type,
                result.importance,
                result.confidence,
                result.raw_model,
                duration_ms,
            )
            return result

        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.warning(
                "Failed to parse structured LLM classification output [duration_ms=%.1f]: %s",
                duration_ms,
                exc,
            )
            return ClassificationResult(
                is_experience=False,
                type=None,
                importance=0.0,
                confidence=0.0,
                reasoning=f"Failed classification validation: {exc}",
                raw_model="fallback_parse_error",
            )

        except LLMException as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                "LLM provider exception during experience classification [duration_ms=%.1f]: %s",
                duration_ms,
                exc,
            )
            return ClassificationResult(
                is_experience=False,
                type=None,
                importance=0.0,
                confidence=0.0,
                reasoning=f"LLM provider error: {exc}",
                raw_model="fallback_llm_error",
            )

        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                "Unexpected error during experience classification [duration_ms=%.1f]: %s",
                duration_ms,
                exc,
            )
            return ClassificationResult(
                is_experience=False,
                type=None,
                importance=0.0,
                confidence=0.0,
                reasoning=f"Unexpected error: {exc}",
                raw_model="fallback_unexpected_error",
            )
