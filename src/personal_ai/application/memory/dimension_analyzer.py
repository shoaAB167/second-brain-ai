import re
from typing import Any, List, Optional, Set

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

    _PATTERN_DOMAIN_DIMENSIONS = {
        "CAREER": [
            RetrievalDimension.GOALS,
            RetrievalDimension.PROJECTS,
            RetrievalDimension.DECISIONS,
            RetrievalDimension.PAST_EXPERIENCES,
        ],
        "FITNESS": [
            RetrievalDimension.HABITS,
            RetrievalDimension.CURRENT_STATE,
            RetrievalDimension.PAST_EXPERIENCES,
        ],
        "RELATIONSHIPS": [
            RetrievalDimension.RELATIONSHIPS,
            RetrievalDimension.EMOTIONS,
            RetrievalDimension.PAST_EXPERIENCES,
        ],
        "HEALTH": [
            RetrievalDimension.CURRENT_STATE,
            RetrievalDimension.HABITS,
            RetrievalDimension.EMOTIONS,
            RetrievalDimension.PAST_EXPERIENCES,
        ],
        "PROJECTS": [
            RetrievalDimension.PROJECTS,
            RetrievalDimension.GOALS,
            RetrievalDimension.DECISIONS,
            RetrievalDimension.PAST_EXPERIENCES,
        ],
        "LEARNING": [
            RetrievalDimension.GOALS,
            RetrievalDimension.PROJECTS,
            RetrievalDimension.HABITS,
            RetrievalDimension.PAST_EXPERIENCES,
        ],
        "FINANCE": [
            RetrievalDimension.GOALS,
            RetrievalDimension.CONSTRAINTS,
            RetrievalDimension.DECISIONS,
            RetrievalDimension.PAST_EXPERIENCES,
        ],
        "SOCIAL": [
            RetrievalDimension.RELATIONSHIPS,
            RetrievalDimension.EMOTIONS,
            RetrievalDimension.PAST_EXPERIENCES,
        ],
        "GENERAL": [
            RetrievalDimension.PERSONALITY,
            RetrievalDimension.PAST_EXPERIENCES,
            RetrievalDimension.CURRENT_STATE,
        ],
    }

    def match_pattern_dimensions(self, pattern: Any) -> List[RetrievalDimension]:
        """Map a PersonalPattern entity's domain to matching RetrievalDimensions."""
        dom_str = (
            pattern.domain.value if hasattr(pattern.domain, "value") else str(getattr(pattern, "domain", "GENERAL") or "GENERAL")
        ).upper().strip()

        matched = self._PATTERN_DOMAIN_DIMENSIONS.get(dom_str, [RetrievalDimension.PAST_EXPERIENCES])
        return sorted(list(set(matched)), key=lambda d: d.value)

    def calculate_pattern_query_relevance(
        self,
        query: str,
        pattern: Any,
        conversation_context: Optional[List[LLMMessage]] = None,
    ) -> float:
        """Calculate deterministic lexical and thematic relevance between user query and pattern."""
        if not query or not query.strip():
            return 0.0

        text = query.lower().strip()
        if conversation_context:
            for msg in reversed(conversation_context[-2:]):
                if msg.content:
                    text += " " + msg.content.lower().strip()

        # Extract words from query (length >= 3)
        query_words = set(re.findall(r"\b[a-z]{3,}\b", text))
        if not query_words:
            return 0.0

        # Pattern text and domain words
        pat_desc = (getattr(pattern, "description", "") or "").lower()
        pat_dom = (
            pattern.domain.value if hasattr(getattr(pattern, "domain", ""), "value") else str(getattr(pattern, "domain", "") or "")
        ).lower()
        pat_words = set(re.findall(r"\b[a-z]{3,}\b", f"{pat_desc} {pat_dom}"))

        # Domain-specific thematic indicator keywords in query
        domain_thematic_keywords = {
            "CAREER": {"career", "goal", "goals", "job", "work", "profession", "milestone", "stress", "salary", "promotion", "architect", "engineer", "interview"},
            "PROJECTS": {"project", "projects", "code", "coding", "deadline", "consistency", "inconsistent", "delay", "procrastinate", "momentum", "stalled", "surge", "intensity", "second", "brain", "app", "system"},
            "FITNESS": {"fitness", "gym", "workout", "exercise", "run", "running", "routine", "habit", "schedule", "lifting", "training", "streak", "cardio"},
            "HEALTH": {"health", "sleep", "slept", "insomnia", "tired", "energy", "exhausted", "fatigue", "drain", "groggy", "recovery", "rest"},
            "RELATIONSHIPS": {"relationship", "relationships", "family", "friend", "friends", "sister", "brother", "partner", "social", "boss", "colleague", "coworker"},
            "LEARNING": {"learning", "study", "studying", "focus", "morning", "learn", "course", "skill", "deep", "practice", "read", "reading"},
            "FINANCE": {"finance", "financial", "budget", "money", "spend", "saving", "salary", "expense", "cost"},
            "SOCIAL": {"social", "friends", "meetup", "people", "gather", "community"},
            "GENERAL": {"routine", "habit", "behavior", "tendency", "pattern"},
        }

        overlap = query_words.intersection(pat_words)
        thematic_kw = domain_thematic_keywords.get(pat_dom.upper(), set())
        thematic_overlap = query_words.intersection(thematic_kw)

        if not overlap and not thematic_overlap:
            return 0.0

        score = 0.0
        if thematic_overlap:
            score += min(len(thematic_overlap) * 0.35, 0.70)
        if overlap:
            score += min(len(overlap) * 0.20, 0.60)

        return round(min(score, 1.0), 4)
