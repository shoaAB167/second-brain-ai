import json
import time
from typing import Any, Dict, List, Optional
from pydantic import ValidationError

from personal_ai.config.settings import get_settings
from personal_ai.core.logger import get_logger
from personal_ai.domain.experience import ClassificationResult
from personal_ai.domain.experience.extractor_models import ExperienceExtractionResult
from personal_ai.llm.client import LLMClient
from personal_ai.llm.exceptions import LLMException
from personal_ai.llm.models import LLMMessage
from personal_ai.prompts import EXPERIENCE_EXTRACTOR_SYSTEM_PROMPT

logger = get_logger(__name__)


class ExperienceExtractor:
    """AI-powered Experience extractor depending on abstract LLMClient interface.

    Extracts structured ExperienceExtractionResult metrics from classified user messages
    without modifying or writing to the database directly.
    """

    def __init__(self, llm_client: LLMClient, model: Optional[str] = None) -> None:
        """Initialize extractor with abstract LLMClient and optional model override.

        Args:
            llm_client: Abstract LLM client interface.
            model: Optional model identifier override (defaults to Settings.experience_extractor_model).
        """
        settings = get_settings()
        self._llm_client = llm_client
        self._model = model or settings.experience_extractor_model

    async def extract(
        self,
        content: str,
        classification: ClassificationResult,
        conversation_context: Optional[List[LLMMessage]] = None,
    ) -> ExperienceExtractionResult:
        """Extract structured ExperienceExtractionResult metrics from user message content.

        Args:
            content: Raw text content of the user message.
            classification: ClassificationResult produced by ExperienceClassifier.
            conversation_context: Optional list of recent LLMMessage objects for reference resolution.

        Returns:
            ExperienceExtractionResult: Validated extraction result object with explicit success flag.
        """
        if not content or not content.strip() or not classification or not classification.is_experience:
            return ExperienceExtractionResult(
                success=False,
                content=None,
                domain=None,
                status=None,
                confidence=0.0,
                reasoning="Message content empty or not classified as experience.",
                raw_model="none",
            )

        start_time = time.perf_counter()

        exp_type_str = classification.type.value if hasattr(classification.type, "value") else str(classification.type)
        prompt_body = f"CLASSIFIED EXPERIENCE TYPE: {exp_type_str}\n\n"

        if conversation_context:
            context_lines = []
            for msg in conversation_context:
                role_label = msg.role.capitalize() if hasattr(msg.role, "capitalize") else str(msg.role)
                context_lines.append(f"{role_label}: \"{msg.content}\"")
            context_str = "\n".join(context_lines)
            prompt_body += (
                f"PRIOR CONVERSATION CONTEXT (FOR REFERENCE RESOLUTION ONLY):\n{context_str}\n\n"
            )

        prompt_body += f"TARGET USER MESSAGE TO EXTRACT FROM:\n\"\"\"{content}\"\"\""

        messages = [
            LLMMessage(role="system", content=EXPERIENCE_EXTRACTOR_SYSTEM_PROMPT),
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

            # Clean potential markdown block formatting
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()

            data: Dict[str, Any] = json.loads(raw_text)

            # Construct and validate ExperienceExtractionResult with explicit success=True
            result = ExperienceExtractionResult(
                success=True,
                content=data.get("content"),
                type=data.get("type") or classification.type,
                domain=data.get("domain"),
                importance=data.get("importance"),
                lifecycle=data.get("lifecycle"),
                emotional_context=data.get("emotional_context"),
                people_involved=data.get("people_involved"),
                temporal_context=data.get("temporal_context"),
                evidence_level=data.get("evidence_level") or "EXTRACTED",
                status=data.get("status", "active"),
                confidence=data.get("confidence"),
                reasoning=data.get("reasoning"),
                raw_model=llm_response.model,
            )

            duration_ms = (time.perf_counter() - start_time) * 1000.0

            logger.info(
                "Experience extracted successfully [domain=%s, confidence=%.2f, model=%s, duration_ms=%.1f]",
                result.domain,
                result.confidence,
                result.raw_model,
                duration_ms,
            )
            return result

        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.warning(
                "Failed to parse structured LLM extraction output [duration_ms=%.1f]: %s",
                duration_ms,
                exc,
            )
            return ExperienceExtractionResult(
                success=False,
                content=None,
                domain=None,
                status=None,
                confidence=0.0,
                reasoning=f"Failed extraction validation: {exc}",
                raw_model="fallback_parse_error",
            )

        except LLMException as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                "LLM provider exception during experience extraction [duration_ms=%.1f]: %s",
                duration_ms,
                exc,
            )
            return ExperienceExtractionResult(
                success=False,
                content=None,
                domain=None,
                status=None,
                confidence=0.0,
                reasoning=f"LLM provider error: {exc}",
                raw_model="fallback_llm_error",
            )

        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                "Unexpected error during experience extraction [duration_ms=%.1f]: %s",
                duration_ms,
                exc,
            )
            return ExperienceExtractionResult(
                success=False,
                content=None,
                domain=None,
                status=None,
                confidence=0.0,
                reasoning=f"Unexpected error: {exc}",
                raw_model="fallback_unexpected_error",
            )
