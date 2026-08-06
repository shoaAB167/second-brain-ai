from fastapi import status

from personal_ai.config.settings import get_settings
from personal_ai.core.exceptions import (
    AppException,
    ConfigurationException,
    ExternalServiceException,
)
from personal_ai.core.logger import get_logger


def test_settings() -> None:
    """Test setting loading and default fields."""
    settings = get_settings()
    assert settings.app_name is not None
    assert settings.api_v1_str == "/api/v1"


def test_logger() -> None:
    """Test logger initialization and idempotency."""
    logger1 = get_logger("test_logger")
    logger2 = get_logger("test_logger")
    assert logger1.name == "test_logger"
    assert len(logger1.handlers) == 1
    assert logger1 is logger2


def test_exceptions() -> None:
    """Test custom exception hierarchy and status codes."""
    exc = AppException("Test error", status_code=400, details={"foo": "bar"})
    assert exc.status_code == 400
    assert exc.message == "Test error"
    assert exc.details == {"foo": "bar"}

    config_exc = ConfigurationException("Config error")
    assert config_exc.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    ext_exc = ExternalServiceException("External service failure")
    assert ext_exc.status_code == status.HTTP_502_BAD_GATEWAY
