from fastapi import status
import pytest

from personal_ai.config.settings import Settings, get_settings
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


def test_production_jwt_secret_validation() -> None:
    """Item 1: Verify production mode fails startup if jwt_secret_key is insecure or default."""
    # Default dev secret in production -> raises ValueError
    with pytest.raises(ValueError) as exc_info:
        Settings(app_env="production")
    assert "jwt_secret_key" in str(exc_info.value)

    # Short secret in production -> raises ValueError
    with pytest.raises(ValueError) as exc_info_short:
        Settings(app_env="production", jwt_secret_key="short-key")
    assert "jwt_secret_key" in str(exc_info_short.value)

    # Valid >=32 char secret in production -> succeeds
    prod_settings = Settings(
        app_env="production",
        jwt_secret_key="a-very-long-and-secure-production-jwt-signing-key-12345",
    )
    assert prod_settings.app_env == "production"


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
