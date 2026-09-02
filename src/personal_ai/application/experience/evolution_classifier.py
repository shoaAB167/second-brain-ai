from dataclasses import dataclass
import json
import re
from typing import Optional

from personal_ai.core.logger import get_logger
from personal_ai.domain.experience.entity import Experience
from personal_ai.domain.experience.enums import ExperienceRelationshipType
from personal_ai.llm.client import LLMClient
from personal_ai.llm.models import LLMMessage

logger = get_logger(__name__)


@dataclass
class ExperienceEvolutionClassificationResult:
    """Structured result of evaluating the evolution relationship between two memories."""

    relationship: ExperienceRelationshipType
    confidence: float
    reason: Optional[str] = None


class ExperienceEvolutionClassifier:
    """Classifies semantic relationships between a new experience and an existing candidate memory.

    Model/provider agnostic using the domain LLMClient abstraction.
    Fails closed on invalid output.
    """

    SYSTEM_PROMPT = """You are a Memory Evolution Classifier for a Second Brain AI system.
Your task is to analyze a NEW observation about a user in relation to an EXISTING remembered experience and determine how the new observation relates to or evolves the existing memory.

Classification Taxonomy:
1. UPDATES: The new observation provides a direct correction, replacement, or change of state to the existing memory (e.g. "I go to gym at 7 PM" updates "I go to gym at 6 PM", or "My salary goal is now 30 LPA" updates "My salary goal is 20 LPA").
2. CONTRADICTS: The new observation directly contradicts the existing memory without explicitly updating a timeline or state change (e.g. "I want to become an AI engineer" contradicts "I don't want to work in AI").
3. REINFORCES: The new observation strengthens, confirms, or repeats the existing memory (e.g. "I really love volleyball" reinforces "I like volleyball").
4. SUPERSEDES: The new observation completely renders the old memory obsolete, expired, or superseded in context.
5. RELATED: The new observation is topically related to the existing memory but does not update, contradict, or reinforce it (e.g. "I played volleyball yesterday" is related to "I play volleyball").
6. UNRELATED: The two observations are about different topics or do not have a meaningful relationship.

Output Format:
You MUST return ONLY a valid JSON object with EXACTLY these keys:
{
  "relationship": "UPDATES" | "CONTRADICTS" | "REINFORCES" | "SUPERSEDES" | "RELATED" | "UNRELATED",
  "confidence": <float between 0.0 and 1.0>,
  "reason": "<one sentence explanation>"
}
Do not include any conversational preamble, explanation, or markdown backticks outside the JSON object."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize classifier with LLMClient instance."""
        self._llm_client = llm_client

    async def classify_relationship(
        self,
        new_experience: Experience,
        existing_experience: Experience,
    ) -> ExperienceEvolutionClassificationResult:
        """Classify the relationship between new_experience and existing_experience.

        Args:
            new_experience: The incoming/new Experience entity.
            existing_experience: The existing candidate Experience entity.

        Returns:
            ExperienceEvolutionClassificationResult: Structured classification result (fails closed).
        """
        user_prompt = f"""EXISTING MEMORY:
- Type: {existing_experience.type.value if existing_experience.type else "FACT"}
- Domain: {existing_experience.domain or "General"}
- Lifecycle Nature: {existing_experience.lifecycle.value if existing_experience.lifecycle else "STABLE"}
- Content: "{existing_experience.content}"

NEW OBSERVATION:
- Type: {new_experience.type.value if new_experience.type else "FACT"}
- Domain: {new_experience.domain or "General"}
- Content: "{new_experience.content}"

Analyze the relationship from the NEW observation to the EXISTING memory:"""

        messages = [
            LLMMessage(role="system", content=self.SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]

        try:
            response = await self._llm_client.generate_response(
                messages=messages,
                temperature=0.0,
            )
            return self._parse_response(response.content)
        except Exception as exc:
            logger.warning(
                "Evolution classification LLM call failed safely [new_id=%s, existing_id=%s]: %s",
                new_experience.id,
                existing_experience.id,
                exc,
            )
            return ExperienceEvolutionClassificationResult(
                relationship=ExperienceRelationshipType.UNRELATED,
                confidence=0.0,
                reason=f"Classification failed: {str(exc)}",
            )

    def _parse_response(self, content: str) -> ExperienceEvolutionClassificationResult:
        """Safely parse and validate LLM output. Fails closed on invalid schema or unknown enum."""
        if not content or not content.strip():
            return ExperienceEvolutionClassificationResult(
                relationship=ExperienceRelationshipType.UNRELATED,
                confidence=0.0,
                reason="Empty LLM response.",
            )

        cleaned = content.strip()
        # Strip markdown code fencing if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
            cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback regex extraction of JSON object
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    return ExperienceEvolutionClassificationResult(
                        relationship=ExperienceRelationshipType.UNRELATED,
                        confidence=0.0,
                        reason="Failed to parse JSON output.",
                    )
            else:
                return ExperienceEvolutionClassificationResult(
                    relationship=ExperienceRelationshipType.UNRELATED,
                    confidence=0.0,
                    reason="No JSON object found in response.",
                )

        if not isinstance(data, dict):
            return ExperienceEvolutionClassificationResult(
                relationship=ExperienceRelationshipType.UNRELATED,
                confidence=0.0,
                reason="Invalid JSON response structure (expected dictionary).",
            )

        raw_rel = data.get("relationship")
        if not isinstance(raw_rel, str):
            return ExperienceEvolutionClassificationResult(
                relationship=ExperienceRelationshipType.UNRELATED,
                confidence=0.0,
                reason="Missing or non-string relationship field.",
            )

        rel_upper = raw_rel.upper().strip()
        try:
            relationship = ExperienceRelationshipType(rel_upper)
        except ValueError:
            logger.warning("Unrecognized relationship string from LLM: '%s'", raw_rel)
            return ExperienceEvolutionClassificationResult(
                relationship=ExperienceRelationshipType.UNRELATED,
                confidence=0.0,
                reason=f"Unrecognized relationship type '{raw_rel}'.",
            )

        raw_conf = data.get("confidence", 0.0)
        try:
            confidence = float(raw_conf)
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 0.0

        reason = str(data.get("reason", "")).strip() or None

        return ExperienceEvolutionClassificationResult(
            relationship=relationship,
            confidence=round(confidence, 4),
            reason=reason,
        )
