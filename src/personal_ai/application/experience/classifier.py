import json
from typing import Any, Dict
from pydantic import ValidationError

from personal_ai.core.logger import get_logger
from personal_ai.domain.experience import ClassificationResult, ExperienceType
from personal_ai.llm.client import LLMClient
from personal_ai.llm.exceptions import LLMException
from personal_ai.llm.models import LLMMessage

logger = get_logger(__name__)

EXPERIENCE_CLASSIFIER_SYSTEM_PROMPT = """You are an AI Personal Experience Classifier for Second Brain AI.
Your task is to analyze a single user message and determine if it expresses meaningful, persistent personal information about the user.

TAXONOMY CATEGORIES (type):
- GOAL: Intentions, aspirations, objectives (e.g., "I want to become an AI engineer.")
- DECISION: Choices made, commitments (e.g., "I decided to stay at my current job.")
- PREFERENCE: Likes, dislikes, preferred methods (e.g., "I prefer learning by building projects.")
- FACT: Persistent factual information about user's life (e.g., "I live in Bangalore.")
- EVENT: Life events, interviews, scheduled occurrences (e.g., "I got promoted today.")
- RELATIONSHIP: Info about people in user's life (e.g., "My brother lives in Pune.")
- EMOTION_STATE: Feelings, mood, stress/burnout (e.g., "I feel overwhelmed with work.")
- HABIT: Routines, recurring behaviors (e.g., "I run 5 miles every morning.")
- PROJECT: Personal projects being built (e.g., "I'm working on Second Brain AI.")
- OTHER: Personal info not covered above.

NON-EXPERIENCES (is_experience = false):
- General knowledge questions ("What is cosine similarity?", "How does Kafka work?")
- Code debugging or syntax requests ("Fix this Python error", "Write a regex")
- Greetings, trivial chatter, command instructions.

CRITICAL RULES:
1. ONLY classify what the user actually expressed. Do NOT infer unsupported facts.
2. importance: Float between 0.0 (trivial) and 1.0 (vital for persistent user understanding).
3. confidence: Float between 0.0 (uncertain) and 1.0 (certain of classification).
4. If is_experience is false, type MUST be null.

Output strictly valid JSON matching this schema:
{
    "is_experience": boolean,
    "type": string or null,
    "importance": float (0.0 to 1.0),
    "confidence": float (0.0 to 1.0)
}"""


class ExperienceClassifier:
    """AI-powered Experience classifier depending on abstract LLMClient interface.

    Classifies user message content into structured ClassificationResult without
    modifying or writing to the database directly.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize classifier with abstract LLMClient.

        Args:
            llm_client: Abstract LLM client interface.
        """
        self._llm_client = llm_client

    async def classify(self, content: str) -> ClassificationResult:
        """Classify a user message into structured ClassificationResult metrics.

        Args:
            content: Raw text content of the user message.

        Returns:
            ClassificationResult: Structured classification result object.
        """
        if not content or not content.strip():
            return ClassificationResult(
                is_experience=False,
                type=None,
                importance=0.0,
                confidence=1.0,
                raw_model="none",
            )

        messages = [
            LLMMessage(role="system", content=EXPERIENCE_CLASSIFIER_SYSTEM_PROMPT),
            LLMMessage(role="user", content=f"Classify this message:\n\"\"\"{content}\"\"\""),
        ]

        try:
            llm_response = await self._llm_client.generate_response(
                messages=messages,
                response_format={"type": "json_object"},
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

            # Construct ClassificationResult with raw_model metadata
            result = ClassificationResult(
                is_experience=bool(data.get("is_experience", False)),
                type=data.get("type"),
                importance=float(data.get("importance", 0.0)),
                confidence=float(data.get("confidence", 0.0)),
                raw_model=llm_response.model,
            )

            logger.info(
                "Experience classified [is_exp=%s, type=%s, importance=%.2f, confidence=%.2f, model=%s]",
                result.is_experience,
                result.type,
                result.importance,
                result.confidence,
                result.raw_model,
            )
            return result

        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            logger.warning("Failed to parse structured LLM classification output: %s", exc)
            return ClassificationResult(
                is_experience=False,
                type=None,
                importance=0.0,
                confidence=0.0,
                raw_model="fallback_parse_error",
            )

        except LLMException as exc:
            logger.error("LLM provider exception during experience classification: %s", exc)
            return ClassificationResult(
                is_experience=False,
                type=None,
                importance=0.0,
                confidence=0.0,
                raw_model="fallback_llm_error",
            )

        except Exception as exc:
            logger.error("Unexpected error during experience classification: %s", exc)
            return ClassificationResult(
                is_experience=False,
                type=None,
                importance=0.0,
                confidence=0.0,
                raw_model="fallback_unexpected_error",
            )
