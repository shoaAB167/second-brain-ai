import re
from typing import List, Optional, Set

from personal_ai.domain.experience import Experience, ExperienceType, RetrievalDimension
from personal_ai.domain.pattern.entity import PersonalPattern
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

        # Augment with last user/assistant message if available ONLY when current query is a continuation
        if self._is_continuation_query(text_to_analyze) and conversation_context:
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

    def match_pattern_dimensions(self, pattern: PersonalPattern) -> List[RetrievalDimension]:
        """Map a PersonalPattern entity's domain to matching RetrievalDimensions."""
        dom_str = (
            pattern.domain.value if hasattr(pattern.domain, "value") else str(getattr(pattern, "domain", "GENERAL") or "GENERAL")
        ).upper().strip()

        matched = self._PATTERN_DOMAIN_DIMENSIONS.get(dom_str, [RetrievalDimension.PAST_EXPERIENCES])
        return sorted(list(set(matched)), key=lambda d: d.value)

    _CONTINUATION_PATTERNS = [
        r"\b(get\s+(myself\s+)?back\s+on\s+track|fix\s+(that|this|it)|do\s+(that|it|this))\b",
        r"\b(why\s+does\s+that\s+happen|why\s+is\s+that|why\s+do\s+i\s+(do\s+this|struggle))\b",
        r"\b(tell\s+me\s+more|what\s+about\s+(that|this|it)|can\s+we\s+do\s+that|what\s+next|and\s+then)\b",
        r"\b(why\s+am\s+i\s+like\s+this|why\s+does\s+this\s+keep\s+happening)\b",
        r"\b(how\s+(can|do|to)\s+(i\s+)?(solve|improve|fix|handle|prepare|deal\s+with)\s*(this|that|it|better)?)\b",
    ]

    _GENERIC_STOP_WORDS = {
        "the", "and", "that", "have", "for", "not", "with", "you", "this", "but", "his",
        "from", "they", "say", "her", "she", "will", "one", "all", "would", "there", "their",
        "what", "out", "about", "who", "get", "which", "go", "when", "make", "can", "like",
        "time", "no", "just", "him", "know", "take", "people", "into", "year", "your", "good",
        "some", "could", "them", "see", "other", "than", "then", "now", "look", "only", "come",
        "its", "over", "think", "also", "back", "after", "use", "two", "how", "our", "work",
        "works", "working", "first", "well", "way", "even", "new", "want", "because", "any",
        "these", "give", "day", "most", "us", "is", "am", "are", "was", "were", "be", "been",
        "being", "do", "does", "did", "doing", "a", "an", "i", "me", "my", "myself", "we",
        "our", "ours", "it", "its", "tell", "say", "help", "solve", "question", "answer",
        "please", "should", "could",
        # Generic domain stop words that should not trigger false positives by themselves
        "project", "projects", "system", "systems", "app", "apps", "code", "coding", "task",
        "tasks", "goal", "goals", "habit", "habits", "routine", "routines", "state", "states",
        "pattern", "patterns", "thing", "things", "stress", "stressed", "stressing", "general",
        "activity", "activities", "reports", "appears", "tends", "tend", "report", "reported",
    }

    def _is_continuation_query(self, query: str) -> bool:
        """Check if the current query is an anaphoric continuation referring to preceding context."""
        clean = query.lower().strip()
        words = clean.split()
        if len(words) <= 8 and any(re.search(pat, clean) for pat in self._CONTINUATION_PATTERNS):
            return True
        if len(words) <= 4 and any(w in ("that", "this", "it", "track") for w in words):
            return True
        return False

    def calculate_pattern_query_relevance(
        self,
        query: str,
        pattern: PersonalPattern,
        conversation_context: Optional[List[LLMMessage]] = None,
    ) -> float:
        """Calculate conservative deterministic lexical and thematic relevance between user query and pattern.

        Invariants:
        1. Current user query receives the strongest weight / primary intent.
        2. Conversation history supplements query ONLY when current message is an anaphoric continuation.
        3. Generic domain words (work, goal, stress, routine, habit, project, system, pattern) do not trigger false positives.
        4. If there is no meaningful relationship between query and pattern, returns 0.0.
        """
        if not query or not query.strip():
            return 0.0

        current_query_text = query.lower().strip()

        # Step 1: Resolve context text: current message is primary; history only used on continuation
        if self._is_continuation_query(current_query_text) and conversation_context:
            context_snippets: List[str] = [current_query_text]
            for msg in reversed(conversation_context[-2:]):
                if msg.content:
                    context_snippets.append(msg.content.lower().strip())
            analyzed_text = " ".join(context_snippets)
        else:
            analyzed_text = current_query_text

        # Step 2: Extract meaningful content words (excluding generic stop words)
        raw_query_words = set(re.findall(r"\b[a-z]{3,}\b", analyzed_text))
        meaningful_query_words = {w for w in raw_query_words if w not in self._GENERIC_STOP_WORDS and len(w) >= 4}

        if not meaningful_query_words:
            return 0.0

        # Step 3: Extract meaningful content words from pattern description
        pat_desc = (getattr(pattern, "description", "") or "").lower()
        raw_pat_words = set(re.findall(r"\b[a-z]{3,}\b", pat_desc))
        meaningful_pat_words = {w for w in raw_pat_words if w not in self._GENERIC_STOP_WORDS and len(w) >= 4}

        # Step 4: Check exact phrase / multi-word overlap
        query_words_list = re.findall(r"\b[a-z]{3,}\b", analyzed_text)
        query_bigrams = {
            f"{query_words_list[i]} {query_words_list[i+1]}"
            for i in range(len(query_words_list) - 1)
            if query_words_list[i] not in self._GENERIC_STOP_WORDS or query_words_list[i+1] not in self._GENERIC_STOP_WORDS
        }
        has_bigram_match = any(bg in pat_desc for bg in query_bigrams)

        # Step 5: Check stem/word overlap
        overlap_count = 0
        for qw in meaningful_query_words:
            for pw in meaningful_pat_words:
                if qw == pw:
                    overlap_count += 1
                    break
                elif len(qw) >= 4 and len(pw) >= 4 and (qw[:4] == pw[:4] or qw[:4] in pw or pw[:4] in qw):
                    overlap_count += 1
                    break

        if overlap_count == 0 and not has_bigram_match:
            return 0.0

        score = 0.0
        if has_bigram_match:
            score += 0.50
        if overlap_count >= 2:
            score += 0.50
        elif overlap_count == 1:
            score += 0.35

        return round(min(score, 1.0), 4)

