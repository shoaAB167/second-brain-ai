import re
from typing import List, Optional, Set

from personal_ai.domain.experience import Experience, ExperienceType, RetrievalDimension
from personal_ai.llm.models import LLMMessage


class QueryDimensionAnalyzer:
    """Lightweight, high-precision deterministic query and context dimension analyzer.

    Uses high-confidence intent patterns to avoid false activations on ordinary language.
    When no dimension is confidently detected, returns an empty list, allowing semantic vector
    retrieval to serve as the primary mechanism.
    """

    _DIMENSION_PATTERNS = {
        RetrievalDimension.GOALS: [
            r"\b(my\s+)?(career\s+)?goals?\b",
            r"\bwant\s+to\s+(reach|achieve|earn|make)\b",
            r"\bplanning\s+to\s+(reach|achieve)\b",
            r"\baspire\s+to\b",
            r"\baim\s+to\b",
            r"\b(salary|lpa)\s+(goal|target|expectation|reach)\b",
            r"\bcareer\s+aspirations?\b",
        ],
        RetrievalDimension.PROJECTS: [
            r"\b(my\s+)?projects?\b",
            r"\bworking\s+on\s+(the\s+|my\s+)?(project|app|codebase|system)\b",
            r"\bsecond\s+brain\s+ai\b",
            r"\b(project|codebase|repo|repository)\s+status\b",
            r"\bbuilding\s+(the\s+|a\s+|my\s+)?(project|app|system|product)\b",
        ],
        RetrievalDimension.PREFERENCES: [
            r"\b(my\s+)?preferences?\b",
            r"\b(i\s+)?prefer\b",
            r"\bfavorite\b",
            r"\bfavourite\b",
            r"\bwhat\s+do\s+i\s+like\b",
            r"\bmy\s+favorite\b",
        ],
        RetrievalDimension.HABITS: [
            r"\b(my\s+)?habits?\b",
            r"\b(daily|weekly)\s+routine\b",
            r"\busually\s+(go|do|wake|sleep|eat|workout)\b",
            r"\bwhat\s+time\s+do\s+i\b",
            r"\b(my\s+)?schedule\b",
            r"\bat\s+what\s+time\b",
        ],
        RetrievalDimension.RELATIONSHIPS: [
            r"\b(my\s+)?(sister|brother|mother|father|mom|dad|wife|husband|partner|boss|manager|colleague|mentor|coworker|friend)s?\b",
            r"\bwho\s+is\s+my\b",
            r"\bwhere\s+does\s+my\s+(sister|brother|friend|mom|dad|family)\b",
        ],
        RetrievalDimension.EMOTIONS: [
            r"\b(i\s+)?(feel|feeling|felt)\s+(scared|afraid|anxious|nervous|stressed|overwhelmed|happy|sad|frustrated|upset|excited|doubtful|confident)\b",
            r"\b(my\s+)?(mood|emotions?)\b",
            r"\bwhy\s+am\s+i\s+(anxious|stressed|scared|upset|frustrated|overwhelmed)\b",
            r"\bhow\s+did\s+i\s+feel\b",
        ],
        RetrievalDimension.DECISIONS: [
            r"\bshould\s+i\s+(continue|quit|switch|stay|stop|start|proceed|choose|pick)\b",
            r"\b(my\s+)?decisions?\b",
            r"\bwhat\s+did\s+i\s+decide\b",
            r"\bwhether\s+to\s+(continue|quit|switch|stay)\b",
        ],
        RetrievalDimension.CURRENT_STATE: [
            r"\bhow\s+am\s+i\s+(feeling|doing)\s+today\b",
            r"\b(my\s+)?current\s+(state|condition|mood)\b",
            r"\b(am\s+i|feeling)\s+(tired|busy|exhausted)\s+today\b",
            r"\bhow\s+am\s+i\s+today\b",
        ],
        RetrievalDimension.CONSTRAINTS: [
            r"\b(my\s+)?(constraints?|limitations?)\b",
            r"\bwhat\s+are\s+my\s+(limits|boundaries|restrictions)\b",
            r"\bbudget\s+limit(ation)?\b",
        ],
        RetrievalDimension.PAST_EXPERIENCES: [
            r"\bwhat\s+happened\s+before\b",
            r"\bwhat\s+did\s+i\s+say\s+previously\s+about\b",
            r"\bremember\s+when\b",
            r"\bwhat\s+was\s+i\s+doing\s+last\s+year\b",
            r"\bin\s+the\s+past\b",
            r"\bwhat\s+did\s+i\s+previously\b",
            r"\bhistorically\b",
            r"\bwhat\s+did\s+i\s+used?\s+to\b",
            r"\bwhere\s+did\s+i\s+live\s+in\s+the\s+past\b",
            r"\bwhat\s+used\s+to\s+be\s+my\b",
        ],
        RetrievalDimension.PERSONALITY: [
            r"\bwho\s+am\s+i\b",
            r"\bwhat\s+is\s+my\s+name\b",
            r"\bwhere\s+do\s+i\s+live\b",
            r"\bwhere\s+do\s+i\s+currently\s+live\b",
            r"\babout\s+me\b",
            r"\bmy\s+identity\b",
            r"\bmy\s+background\b",
        ],
    }

    _HISTORICAL_HIGH_CONFIDENCE_PATTERNS = [
        r"\bwhat\s+happened\s+before\b",
        r"\bwhat\s+did\s+i\s+say\s+previously\s+about\b",
        r"\bremember\s+when\b",
        r"\bwhat\s+was\s+i\s+doing\s+last\s+year\b",
        r"\bin\s+the\s+past\b",
        r"\bwhat\s+did\s+i\s+previously\b",
        r"\bhistorically\b",
        r"\bwhat\s+did\s+i\s+used?\s+to\b",
        r"\bwhere\s+did\s+i\s+live\s+in\s+the\s+past\b",
        r"\bwhat\s+used\s+to\s+be\s+my\b",
    ]

    def analyze_query(
        self,
        query: str,
        conversation_context: Optional[List[LLMMessage]] = None,
    ) -> List[RetrievalDimension]:
        """Analyze query and optional recent conversation messages to detect relevant retrieval dimensions."""
        if not query or not query.strip():
            return []

        text_to_analyze = query.lower().strip()

        # Augment with last user/assistant message if available
        if conversation_context:
            for msg in reversed(conversation_context[-2:]):
                if msg.content:
                    text_to_analyze += " " + msg.content.lower().strip()

        detected: Set[RetrievalDimension] = set()

        for dim, patterns in self._DIMENSION_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, text_to_analyze):
                    detected.add(dim)
                    break

        # High-confidence multi-dimension activation for decision support queries
        if RetrievalDimension.DECISIONS in detected and ("project" in text_to_analyze or "working on" in text_to_analyze):
            detected.add(RetrievalDimension.PROJECTS)
            detected.add(RetrievalDimension.GOALS)

        return sorted(list(detected), key=lambda d: d.value)

    def is_historical_query(self, query: str) -> bool:
        """Conservatively check if a query is a high-confidence historical inquiry."""
        if not query or not query.strip():
            return False

        clean_text = query.lower().strip()
        return any(re.search(pat, clean_text) for pat in self._HISTORICAL_HIGH_CONFIDENCE_PATTERNS)

    def match_experience_dimensions(self, experience: Experience) -> List[RetrievalDimension]:
        """Map an Experience domain entity's attributes to matching RetrievalDimensions."""
        matched: Set[RetrievalDimension] = set()

        exp_type_val = (
            experience.type.value if hasattr(experience.type, "value") else str(experience.type or "")
        ).upper()

        if exp_type_val == ExperienceType.GOAL.value:
            matched.add(RetrievalDimension.GOALS)
        elif exp_type_val == ExperienceType.PROJECT.value:
            matched.add(RetrievalDimension.PROJECTS)
        elif exp_type_val == ExperienceType.PREFERENCE.value:
            matched.add(RetrievalDimension.PREFERENCES)
        elif exp_type_val == ExperienceType.HABIT.value:
            matched.add(RetrievalDimension.HABITS)
        elif exp_type_val == ExperienceType.RELATIONSHIP.value:
            matched.add(RetrievalDimension.RELATIONSHIPS)
        elif exp_type_val in (ExperienceType.STATE.value, "EMOTION", "EMOTION_STATE"):
            matched.add(RetrievalDimension.CURRENT_STATE)
            matched.add(RetrievalDimension.EMOTIONS)
        elif exp_type_val == ExperienceType.DECISION.value:
            matched.add(RetrievalDimension.DECISIONS)
        elif exp_type_val == ExperienceType.EVENT.value:
            matched.add(RetrievalDimension.PAST_EXPERIENCES)
        elif exp_type_val == ExperienceType.FACT.value:
            matched.add(RetrievalDimension.PERSONALITY)

        if experience.emotional_context:
            matched.add(RetrievalDimension.EMOTIONS)

        if experience.people_involved:
            matched.add(RetrievalDimension.RELATIONSHIPS)

        life_val = (
            experience.lifecycle.value if hasattr(experience.lifecycle, "value") else str(experience.lifecycle or "")
        ).upper()
        if life_val == "TEMPORARY":
            matched.add(RetrievalDimension.CURRENT_STATE)

        life_status_val = (
            experience.lifecycle_status.value
            if hasattr(experience.lifecycle_status, "value")
            else str(experience.lifecycle_status or "")
        ).upper()
        if life_status_val in ("SUPERSEDED", "EXPIRED"):
            matched.add(RetrievalDimension.PAST_EXPERIENCES)

        return sorted(list(matched), key=lambda d: d.value)
