from dataclasses import dataclass
import json
import re
from typing import Dict, List, Optional
import uuid

from personal_ai.core.logger import get_logger
from personal_ai.domain.experience.entity import Experience
from personal_ai.domain.experience.enums import ExperienceRelationshipType
from personal_ai.llm.client import LLMClient
from personal_ai.llm.models import LLMMessage

logger = get_logger(__name__)


@dataclass
class ExperienceEvolutionClassificationResult:
    """Structured result of evaluating the evolution relationship between two memories."""

    candidate_id: uuid.UUID
    relationship: ExperienceRelationshipType
    confidence: float
    reason: Optional[str] = None


class ExperienceEvolutionClassifier:
    """Classifies semantic relationships between a new experience and existing candidate memories in a single LLM call.

    Model/provider agnostic using the domain LLMClient abstraction.
    Fails closed on invalid output.
    """

    SYSTEM_PROMPT = """You are a Memory Evolution Classifier for a Second Brain AI system.
Your task is to analyze a NEW observation about a user in relation to a list of EXISTING candidate memories and determine how the new observation relates to or evolves each candidate memory.

Classification Taxonomy:
1. UPDATES: The new observation provides a direct correction, replacement, or change of state to the existing memory (e.g. "I go to gym at 7 PM" updates "I go to gym at 6 PM", or "My salary goal is now 30 LPA" updates "My salary goal is 20 LPA").
2. CONTRADICTS: The new observation directly conflicts or disagrees with the existing memory without explicitly updating a timeline or state change (e.g. "I want to become an AI engineer" contradicts "I don't want to work in AI").
3. REINFORCES: The new observation strengthens, confirms, or repeats the existing memory (e.g. "I really love volleyball" reinforces "I like volleyball").
4. RELATED: The new observation is topically connected to the existing memory but does not update, contradict, or reinforce it (e.g. "I played volleyball yesterday" is related to "I play volleyball").
5. UNRELATED: The two observations are about different topics or do not have a meaningful relationship.

Output Format:
You MUST return ONLY a valid JSON object with the key "relationships", containing an array evaluating every candidate:
{
  "relationships": [
    {
      "candidate_id": "<exact candidate id from list>",
      "relationship": "UPDATES" | "CONTRADICTS" | "REINFORCES" | "RELATED" | "UNRELATED",
      "confidence": <float between 0.0 and 1.0>,
      "reason": "<one sentence explanation>"
    }
  ]
}
Do not include any conversational preamble, explanation, or markdown backticks outside the JSON object."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize classifier with LLMClient instance."""
        self._llm_client = llm_client

    async def classify_relationships(
        self,
        new_experience: Experience,
        candidate_experiences: List[Experience],
    ) -> Dict[uuid.UUID, ExperienceEvolutionClassificationResult]:
        """Classify relationships between new_experience and all candidate_experiences in a single LLM call.

        Args:
            new_experience: The incoming/new Experience entity.
            candidate_experiences: List of existing candidate Experience entities (max 3).

        Returns:
            Dict[uuid.UUID, ExperienceEvolutionClassificationResult]: Map of candidate ID to classification result.
        """
        if not candidate_experiences:
            logger.debug(
                "No candidate memories provided for evolution classification [experience_id=%s]",
                new_experience.id,
            )
            return {}

        candidates_text_parts = []
        valid_candidate_ids = set()

        for idx, cand in enumerate(candidate_experiences, start=1):
            valid_candidate_ids.add(cand.id)
            cand_type = cand.type.value if cand.type else "FACT"
            cand_domain = cand.domain or "General"
            cand_lifecycle = cand.lifecycle.value if cand.lifecycle else "STABLE"
            candidates_text_parts.append(
                f"[Candidate {idx}]\n"
                f"- Candidate ID: {cand.id}\n"
                f"- Type: {cand_type}\n"
                f"- Domain: {cand_domain}\n"
                f"- Lifecycle Nature: {cand_lifecycle}\n"
                f'- Content: "{cand.content}"'
            )

        candidates_block = "\n\n".join(candidates_text_parts)

        user_prompt = f"""NEW OBSERVATION:
- Type: {new_experience.type.value if new_experience.type else "FACT"}
- Domain: {new_experience.domain or "General"}
- Content: "{new_experience.content}"

EXISTING CANDIDATE MEMORIES:
{candidates_block}

Analyze the relationship from the NEW observation to each EXISTING candidate memory:"""

        messages = [
            LLMMessage(role="system", content=self.SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]

        try:
            response = await self._llm_client.generate_response(
                messages=messages,
                temperature=0.0,
            )
            return self._parse_batch_response(response.content, valid_candidate_ids)
        except Exception as exc:
            logger.warning(
                "Batch evolution classification LLM call failed safely [new_id=%s, candidates_count=%d]: %s",
                new_experience.id,
                len(candidate_experiences),
                exc,
            )
            return {}

    def _parse_batch_response(
        self,
        content: str,
        valid_candidate_ids: set[uuid.UUID],
    ) -> Dict[uuid.UUID, ExperienceEvolutionClassificationResult]:
        """Safely parse and validate batch LLM output. Fails closed on invalid schema or unknown enum."""
        if not content or not content.strip():
            return {}

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
                    return {}
            else:
                return {}

        if not isinstance(data, dict):
            return {}

        relationships_list = data.get("relationships")
        if not isinstance(relationships_list, list):
            return {}

        results: Dict[uuid.UUID, ExperienceEvolutionClassificationResult] = {}

        for item in relationships_list:
            if not isinstance(item, dict):
                continue

            raw_cand_id = item.get("candidate_id")
            if not raw_cand_id:
                continue

            try:
                cand_uuid = uuid.UUID(str(raw_cand_id).strip())
            except (ValueError, TypeError):
                continue

            if cand_uuid not in valid_candidate_ids:
                logger.warning("Classifier returned unrecognized candidate_id: '%s'", raw_cand_id)
                continue

            raw_rel = item.get("relationship")
            if not isinstance(raw_rel, str):
                continue

            rel_upper = raw_rel.upper().strip()
            try:
                relationship = ExperienceRelationshipType(rel_upper)
            except ValueError:
                logger.warning("Unrecognized relationship string from LLM: '%s'", raw_rel)
                relationship = ExperienceRelationshipType.UNRELATED

            raw_conf = item.get("confidence", 0.0)
            try:
                confidence = float(raw_conf)
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError):
                confidence = 0.0

            reason = str(item.get("reason", "")).strip() or None

            results[cand_uuid] = ExperienceEvolutionClassificationResult(
                candidate_id=cand_uuid,
                relationship=relationship,
                confidence=round(confidence, 4),
                reason=reason,
            )

        return results
