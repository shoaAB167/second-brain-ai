from typing import Any, Dict, Optional, Union

from personal_ai.core.exceptions import ExternalServiceException


class LLMException(ExternalServiceException):
    """Base exception for all LLM integration failures."""

    def __init__(
        self,
        message: str = "An error occurred while processing the LLM request.",
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
        message: str = "LLM provider authentication failed. Please check configured credentials.",
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
        message: str = "LLM provider rate limit exceeded. Please try again later.",
        details: Optional[Union[Dict[str, Any], list]] = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=429,
            details=details,
        )


class LLMConnectionException(LLMException):
    """Raised when network connection, timeout, or service unavailability issues occur while reaching the provider."""

    def __init__(
        self,
        message: str = "AI service is temporarily unavailable. Please try again.",
        status_code: int = 503,
        details: Optional[Union[Dict[str, Any], list]] = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            details=details,
        )


class LLMTimeoutException(LLMConnectionException):
    """Raised when an LLM stream or request exceeds its bounded timeout."""

    def __init__(
        self,
        message: str = "AI service is temporarily unavailable. Please try again.",
        details: Optional[Union[Dict[str, Any], list]] = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=504,
            details=details,
        )


class LLMServiceUnavailableException(LLMConnectionException):
    """Raised when LLM provider returns 503 / high demand / service overloaded."""

    def __init__(
        self,
        message: str = "AI service is temporarily unavailable. Please try again.",
        details: Optional[Union[Dict[str, Any], list]] = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=503,
            details=details,
        )
