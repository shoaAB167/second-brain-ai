import re
from typing import List, Optional, Set

from personal_ai.domain.experience import Experience, ExperienceType, RetrievalDimension
from personal_ai.llm.models import LLMMessage


class QueryDimensionAnalyzer:
    """Lightweight, deterministic query and context dimension analyzer.

    Identifies relevant retrieval dimensions from user queries and short-term conversation context,
    and maps domain Experience attributes to corresponding retrieval dimensions.
    """

    _DIMENSION_PATTERNS = {
        RetrievalDimension.GOALS: [
            r"\bgoal\b", r"\btarget\b", r"\baim\b", r"\baspire\b", r"\bambition\b",
            r"\bachieve\b", r"\breach\b", r"\bwant to\b", r"\bplanning to\b",
            r"\blpa\b", r"\bsalary\b", r"\baspiration\b", r"\bmilestone\b",
        ],
        RetrievalDimension.PROJECTS: [
            r"\bproject\b", r"\bbuild\b", r"\bbuilding\b", r"\bwork on\b", r"\bworking on\b",
            r"\bapp\b", r"\bcodebase\b", r"\brepository\b", r"\brepo\b", r"\barchitecture\b",
            r"\bfeature\b", r"\bsystem\b", r"\bsoftware\b", r"\bproduct\b",
        ],
        RetrievalDimension.PREFERENCES: [
            r"\bprefer\b", r"\bpreference\b", r"\blike\b", r"\bfavorite\b", r"\bfavourite\b",
            r"\bdislike\b", r"\bhate\b", r"\blove\b", r"\benjoy\b", r"\bopt for\b",
            r"\btaste\b", r"\bstyle\b", r"\bchoice of\b",
        ],
        RetrievalDimension.HABITS: [
            r"\bhabit\b", r"\broutine\b", r"\busually\b", r"\boften\b", r"\bschedule\b",
            r"\bdaily\b", r"\bgym\b", r"\btime do i\b", r"\bevery day\b", r"\balways\b",
            r"\bnormally\b", r"\btypically\b", r"\bfrequency\b", r"\bworkout\b",
        ],
        RetrievalDimension.RELATIONSHIPS: [
            r"\bfriend\b", r"\bwife\b", r"\bhusband\b", r"\bmom\b", r"\bdad\b",
            r"\bmother\b", r"\bfather\b", r"\bsister\b", r"\bbrother\b", r"\bcolleague\b",
            r"\bmanager\b", r"\bboss\b", r"\bpartner\b", r"\blead\b", r"\bmentor\b",
            r"\bfamily\b", r"\bteam\b", r"\bwho is\b", r"\bnames?\b",
        ],
        RetrievalDimension.EMOTIONS: [
            r"\bfeel\b", r"\bfeeling\b", r"\bafraid\b", r"\bscared\b", r"\bfear\b",
            r"\banxious\b", r"\banxiety\b", r"\bnervous\b", r"\bfrustrated\b", r"\bstress\b",
            r"\bstressed\b", r"\bexcited\b", r"\bhappy\b", r"\bsad\b", r"\bemotion\b",
            r"\bdoubt\b", r"\boverwhelmed\b", r"\bupset\b", r"\bproud\b", r"\bmood\b",
        ],
        RetrievalDimension.DECISIONS: [
            r"\bdecide\b", r"\bdecision\b", r"\bchoice\b", r"\bchose\b", r"\bchosen\b",
            r"\bshould i\b", r"\bwhether to\b", r"\bquit\b", r"\bswitch\b", r"\bopt\b",
            r"\bcontinue\b", r"\btradeoff\b", r"\bverdict\b", r"\bweigh\b",
        ],
        RetrievalDimension.CURRENT_STATE: [
            r"\bcurrently\b", r"\bnow\b", r"\btoday\b", r"\btired\b", r"\bbusy\b",
            r"\bexhausted\b", r"\bright now\b", r"\bhow am i\b", r"\bstate\b",
            r"\bpresently\b", r"\bat the moment\b",
        ],
        RetrievalDimension.CONSTRAINTS: [
            r"\bcannot\b", r"\bmust not\b", r"\bconstraint\b", r"\blimitation\b",
            r"\brestricted\b", r"\bbudget\b", r"\bdeadline\b", r"\bboundary\b",
            r"\brule\b", r"\bblocker\b", r"\blimit\b", r"\bcant\b",
        ],
        RetrievalDimension.PAST_EXPERIENCES: [
            r"\bpast\b", r"\bearlier\b", r"\bremember when\b", r"\bhistory\b",
            r"\bprevious\b", r"\bpreviously\b", r"\bused to\b", r"\bbefore\b",
            r"\byesterday\b", r"\blast year\b", r"\blast week\b", r"\bformer\b",
        ],
        RetrievalDimension.PERSONALITY: [
            r"\bwho am i\b", r"\bmy name\b", r"\bwhere do i live\b", r"\bidentity\b",
            r"\babout me\b", r"\bmy profile\b", r"\bmy background\b", r"\bwhere am i\b",
        ],
    }

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

        # Decision support queries naturally cross-cut into goals, projects, emotions, and constraints
        if RetrievalDimension.DECISIONS in detected:
            if "project" in text_to_analyze or "work" in text_to_analyze:
                detected.add(RetrievalDimension.PROJECTS)
                detected.add(RetrievalDimension.GOALS)
                detected.add(RetrievalDimension.CONSTRAINTS)
                detected.add(RetrievalDimension.EMOTIONS)

        return sorted(list(detected), key=lambda d: d.value)

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
