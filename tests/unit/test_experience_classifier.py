import json
from unittest.mock import AsyncMock, MagicMock
import pytest

from personal_ai.application.experience.classifier import ExperienceClassifier
from personal_ai.domain.experience import ClassificationResult, ExperienceType
from personal_ai.llm.client import LLMClient
from personal_ai.llm.exceptions import LLMRateLimitException
from personal_ai.llm.models import LLMMessage, LLMResponse


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "category_type",
    [
        ExperienceType.GOAL,
        ExperienceType.DECISION,
        ExperienceType.PREFERENCE,
        ExperienceType.FACT,
        ExperienceType.EVENT,
        ExperienceType.RELATIONSHIP,
        ExperienceType.EMOTION_STATE,
        ExperienceType.HABIT,
        ExperienceType.PROJECT,
        ExperienceType.OTHER,
    ],
)
async def test_experience_classifier_all_taxonomy_types(category_type: ExperienceType) -> None:
    """Verify ExperienceClassifier correctly parses all taxonomy categories."""
    mock_llm = MagicMock(spec=LLMClient)
    payload = {
        "is_experience": True,
        "type": category_type.value,
        "importance": 0.85,
        "confidence": 0.95,
        "reasoning": f"Extracted user {category_type.value}.",
    }
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(
            content=json.dumps(payload),
            provider="openai",
            model="gpt-4o-mini",
            latency_ms=10.0,
        )
    )

    classifier = ExperienceClassifier(llm_client=mock_llm)
    result = await classifier.classify("Sample text")

    assert isinstance(result, ClassificationResult)
    assert result.is_experience is True
    assert result.type == category_type
    assert result.importance == 0.85
    assert result.confidence == 0.95
    assert result.reasoning == f"Extracted user {category_type.value}."
    assert result.raw_model == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_case_a_personal_fact() -> None:
    """Requirement 11A: Personal fact -> is_experience=True, type=FACT."""
    mock_llm = MagicMock(spec=LLMClient)
    payload = {
        "is_experience": True,
        "type": "FACT",
        "importance": 0.80,
        "confidence": 0.95,
        "reasoning": "User explicitly states their tech stack context.",
    }
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content=json.dumps(payload), provider="openai", model="gpt-4o-mini", latency_ms=10.0)
    )
    classifier = ExperienceClassifier(llm_client=mock_llm)
    res = await classifier.classify("I work primarily with Java and Spring Boot.")

    assert res.is_experience is True
    assert res.type == ExperienceType.FACT
    assert res.importance == 0.80


@pytest.mark.asyncio
async def test_case_b_preference() -> None:
    """Requirement 11B: Preference -> is_experience=True, type=PREFERENCE."""
    mock_llm = MagicMock(spec=LLMClient)
    payload = {
        "is_experience": True,
        "type": "PREFERENCE",
        "importance": 0.75,
        "confidence": 0.90,
        "reasoning": "User states work environment preference.",
    }
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content=json.dumps(payload), provider="openai", model="gpt-4o-mini", latency_ms=10.0)
    )
    classifier = ExperienceClassifier(llm_client=mock_llm)
    res = await classifier.classify("I prefer working remotely.")

    assert res.is_experience is True
    assert res.type == ExperienceType.PREFERENCE


@pytest.mark.asyncio
async def test_case_c_goal() -> None:
    """Requirement 11C: Goal -> is_experience=True, type=GOAL."""
    mock_llm = MagicMock(spec=LLMClient)
    payload = {
        "is_experience": True,
        "type": "GOAL",
        "importance": 0.90,
        "confidence": 0.95,
        "reasoning": "User expresses salary aspiration.",
    }
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content=json.dumps(payload), provider="openai", model="gpt-4o-mini", latency_ms=10.0)
    )
    classifier = ExperienceClassifier(llm_client=mock_llm)
    res = await classifier.classify("I want to reach 30 LPA.")

    assert res.is_experience is True
    assert res.type == ExperienceType.GOAL


@pytest.mark.asyncio
async def test_case_d_habit() -> None:
    """Requirement 11D: Habit -> is_experience=True, type=HABIT."""
    mock_llm = MagicMock(spec=LLMClient)
    payload = {
        "is_experience": True,
        "type": "HABIT",
        "importance": 0.70,
        "confidence": 0.90,
        "reasoning": "User describes workout routine.",
    }
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content=json.dumps(payload), provider="openai", model="gpt-4o-mini", latency_ms=10.0)
    )
    classifier = ExperienceClassifier(llm_client=mock_llm)
    res = await classifier.classify("I work out five days a week.")

    assert res.is_experience is True
    assert res.type == ExperienceType.HABIT


@pytest.mark.asyncio
async def test_case_e_project() -> None:
    """Requirement 11E: Project -> is_experience=True, type=PROJECT."""
    mock_llm = MagicMock(spec=LLMClient)
    payload = {
        "is_experience": True,
        "type": "PROJECT",
        "importance": 0.85,
        "confidence": 0.92,
        "reasoning": "User describes software project being built.",
    }
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content=json.dumps(payload), provider="openai", model="gpt-4o-mini", latency_ms=10.0)
    )
    classifier = ExperienceClassifier(llm_client=mock_llm)
    res = await classifier.classify("I'm building a personal AI assistant.")

    assert res.is_experience is True
    assert res.type == ExperienceType.PROJECT


@pytest.mark.asyncio
async def test_case_f_general_question() -> None:
    """Requirement 11F: General question -> is_experience=False."""
    mock_llm = MagicMock(spec=LLMClient)
    payload = {
        "is_experience": False,
        "type": None,
        "importance": 0.10,
        "confidence": 0.99,
        "reasoning": "General technical knowledge query.",
    }
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content=json.dumps(payload), provider="openai", model="gpt-4o-mini", latency_ms=10.0)
    )
    classifier = ExperienceClassifier(llm_client=mock_llm)
    res = await classifier.classify("What is dependency injection?")

    assert res.is_experience is False
    assert res.type is None


@pytest.mark.asyncio
async def test_case_g_technical_question() -> None:
    """Requirement 11G: Technical question -> is_experience=False."""
    mock_llm = MagicMock(spec=LLMClient)
    payload = {
        "is_experience": False,
        "type": None,
        "importance": 0.15,
        "confidence": 0.98,
        "reasoning": "Code implementation query.",
    }
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content=json.dumps(payload), provider="openai", model="gpt-4o-mini", latency_ms=10.0)
    )
    classifier = ExperienceClassifier(llm_client=mock_llm)
    res = await classifier.classify("How do I implement Kafka consumers in Spring Boot?")

    assert res.is_experience is False
    assert res.type is None


@pytest.mark.asyncio
async def test_case_h_hypothetical() -> None:
    """Requirement 11H: Hypothetical statement -> is_experience=False."""
    mock_llm = MagicMock(spec=LLMClient)
    payload = {
        "is_experience": False,
        "type": None,
        "importance": 0.20,
        "confidence": 0.95,
        "reasoning": "Hypothetical architecture scenario.",
    }
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content=json.dumps(payload), provider="openai", model="gpt-4o-mini", latency_ms=10.0)
    )
    classifier = ExperienceClassifier(llm_client=mock_llm)
    res = await classifier.classify("If I were building a banking system, I would use Kafka.")

    assert res.is_experience is False
    assert res.type is None


@pytest.mark.asyncio
async def test_case_i_invalid_llm_json() -> None:
    """Requirement 11I: Invalid LLM JSON returns safe fallback (is_experience=False)."""
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content="Malformed json } string", provider="openai", model="gpt-4o-mini", latency_ms=10.0)
    )
    classifier = ExperienceClassifier(llm_client=mock_llm)
    res = await classifier.classify("Sample text")

    assert res.is_experience is False
    assert res.type is None
    assert res.confidence == 0.0
    assert res.raw_model == "fallback_parse_error"


@pytest.mark.asyncio
async def test_case_j_invalid_enum() -> None:
    """Requirement 11J: Invalid enum type returns safe fallback (is_experience=False)."""
    mock_llm = MagicMock(spec=LLMClient)
    payload = {
        "is_experience": True,
        "type": "SUPER_INVALID_TAXONOMY_CATEGORY",
        "importance": 0.8,
        "confidence": 0.8,
    }
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content=json.dumps(payload), provider="openai", model="gpt-4o-mini", latency_ms=10.0)
    )
    classifier = ExperienceClassifier(llm_client=mock_llm)
    res = await classifier.classify("Sample text")

    assert res.is_experience is False
    assert res.type is None
    assert res.raw_model == "fallback_parse_error"


@pytest.mark.asyncio
async def test_case_k_confidence_outside_range() -> None:
    """Requirement 11K: Confidence outside 0..1 returns safe fallback (is_experience=False)."""
    mock_llm = MagicMock(spec=LLMClient)
    payload = {
        "is_experience": True,
        "type": "GOAL",
        "importance": 0.8,
        "confidence": 1.5,
    }
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content=json.dumps(payload), provider="openai", model="gpt-4o-mini", latency_ms=10.0)
    )
    classifier = ExperienceClassifier(llm_client=mock_llm)
    res = await classifier.classify("Sample text")

    assert res.is_experience is False
    assert res.type is None


@pytest.mark.asyncio
async def test_case_l_importance_outside_range() -> None:
    """Requirement 11L: Importance outside 0..1 returns safe fallback (is_experience=False)."""
    mock_llm = MagicMock(spec=LLMClient)
    payload = {
        "is_experience": True,
        "type": "GOAL",
        "importance": -0.5,
        "confidence": 0.8,
    }
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content=json.dumps(payload), provider="openai", model="gpt-4o-mini", latency_ms=10.0)
    )
    classifier = ExperienceClassifier(llm_client=mock_llm)
    res = await classifier.classify("Sample text")

    assert res.is_experience is False
    assert res.type is None


@pytest.mark.asyncio
async def test_case_m_n_conversation_context_resolution() -> None:
    """Requirements 11M & 11N: Verify conversation context resolves references without treating assistant text as user experience."""
    mock_llm = MagicMock(spec=LLMClient)
    payload = {
        "is_experience": True,
        "type": "GOAL",
        "importance": 0.88,
        "confidence": 0.94,
        "reasoning": "Context confirms user goal to move into AI engineering.",
    }
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content=json.dumps(payload), provider="openai", model="gpt-4o-mini", latency_ms=10.0)
    )
    classifier = ExperienceClassifier(llm_client=mock_llm)

    context = [
        LLMMessage(role="user", content="I'm planning to leave my current job."),
        LLMMessage(role="assistant", content="Why?"),
    ]
    res = await classifier.classify("Because I want to move into AI engineering.", conversation_context=context)

    assert res.is_experience is True
    assert res.type == ExperienceType.GOAL

    # Inspect call args to ensure context was passed to LLM
    call_args = mock_llm.generate_response.call_args[1]
    prompt_sent = call_args["messages"][1].content
    assert "PRIOR CONVERSATION CONTEXT" in prompt_sent
    assert "I'm planning to leave my current job." in prompt_sent
    assert "Do NOT treat assistant responses as user experiences." in prompt_sent


def test_classification_result_type_null_when_is_experience_false() -> None:
    """Verify is_experience=False requires type=None."""
    res = ClassificationResult(is_experience=False, type=None, importance=0.1, confidence=0.9)
    assert res.is_experience is False
    assert res.type is None

    with pytest.raises(ValueError, match="is_experience is False"):
        ClassificationResult(is_experience=False, type=ExperienceType.GOAL, importance=0.1, confidence=0.9)


def test_classification_result_score_range_validation() -> None:
    """Verify invalid importance or confidence scores raise ValueError."""
    res = ClassificationResult(is_experience=True, type=ExperienceType.GOAL, importance=0.0, confidence=1.0)
    assert res.importance == 0.0
    assert res.confidence == 1.0

    with pytest.raises(ValueError):
        ClassificationResult(is_experience=True, type=ExperienceType.GOAL, importance=1.5, confidence=0.8)

    with pytest.raises(ValueError):
        ClassificationResult(is_experience=True, type=ExperienceType.GOAL, importance=0.5, confidence=-0.1)


@pytest.mark.asyncio
async def test_case_o_classifier_failure_safe_fallback() -> None:
    """Requirement 11O: Verify ExperienceClassifier returns safe fallback on LLMException without raising."""
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_response = AsyncMock(
        side_effect=LLMRateLimitException("Rate limit exceeded")
    )

    classifier = ExperienceClassifier(llm_client=mock_llm)
    result = await classifier.classify("Some message")

    assert result.is_experience is False
    assert result.type is None
    assert result.importance == 0.0
    assert result.confidence == 0.0
    assert result.raw_model == "fallback_llm_error"
