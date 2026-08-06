from typing import Any, Dict, Optional, Union

from fastapi import Request, status
from fastapi.responses import JSONResponse

from personal_ai.core.logger import get_logger

logger = get_logger(__name__)


class AppException(Exception):
    """Base application exception for Second Brain AI."""

    def __init__(
        self,
        message: str = "An unexpected application error occurred.",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Union[Dict[str, Any], list]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}

    def __str__(self) -> str:
        return f"[{self.__class__.__name__}] {self.message} (status_code={self.status_code})"


class ConfigurationException(AppException):
    """Raised when there is a configuration error."""

    def __init__(
        self,
        message: str = "Invalid application configuration.",
        details: Optional[Union[Dict[str, Any], list]] = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
        )


class ExternalServiceException(AppException):
    """Raised when an interaction with an external service fails."""

    def __init__(
        self,
        message: str = "External service integration failure.",
        status_code: int = status.HTTP_502_BAD_GATEWAY,
        details: Optional[Union[Dict[str, Any], list]] = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            details=details,
        )


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """FastAPI Exception Handler for custom application exceptions."""
    logger.error("Application error handling request %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": exc.__class__.__name__,
                "message": exc.message,
                "details": exc.details,
            }
        },
    )
