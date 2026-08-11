from typing import Any, Dict, Optional, Union

from personal_ai.core.exceptions import ExternalServiceException


class LLMException(ExternalServiceException):
    """Base exception for all LLM integration failures."""

    def __init__(
        self,
        message: str = "LLM provider operation failed.",
        status_code: int = 502,
        details: Optional[Union[Dict[str, Any], list]] = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            details=details,
        )


class LLMAuthenticationException(LLMException):
    """Raised when authentication with the LLM provider fails."""

    def __init__(
        self,
        message: str = "LLM authentication failed. Check API key configuration.",
        details: Optional[Union[Dict[str, Any], list]] = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=401,
            details=details,
        )


class LLMRateLimitException(LLMException):
    """Raised when provider rate limit or quota is exceeded."""

    def __init__(
        self,
        message: str = "LLM rate limit or quota exceeded.",
        details: Optional[Union[Dict[str, Any], list]] = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=429,
            details=details,
        )


class LLMConnectionException(LLMException):
    """Raised when network connection or timeout issues occur while reaching the provider."""

    def __init__(
        self,
        message: str = "Failed to connect to LLM provider.",
        details: Optional[Union[Dict[str, Any], list]] = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=503,
            details=details,
        )
