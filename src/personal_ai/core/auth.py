from datetime import datetime, timedelta, timezone
from typing import Optional
import uuid

import bcrypt
import jwt

from personal_ai.config.settings import settings
from personal_ai.core.exceptions import AppException
from personal_ai.core.logger import get_logger

logger = get_logger(__name__)


def hash_password(password: str) -> str:
    """Hash plaintext password using bcrypt."""
    pwd_bytes = password.encode("utf-8")[:72]
    return bcrypt.hashpw(pwd_bytes, bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plaintext password against bcrypt hash."""
    try:
        pwd_bytes = plain_password.encode("utf-8")[:72]
        return bcrypt.checkpw(pwd_bytes, hashed_password.encode("utf-8"))
    except Exception:
        return False


def create_access_token(user_id: uuid.UUID, expires_delta: Optional[timedelta] = None) -> str:
    """Generate a signed JWT access token for user_id.

    Args:
        user_id: The authenticated user's unique UUID.
        expires_delta: Optional custom expiration duration.

    Returns:
        Encoded JWT token string.
    """
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.jwt_access_token_expire_minutes)

    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": expire,
    }

    encoded_token = jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )
    return encoded_token


def decode_access_token(token: str) -> uuid.UUID:
    """Decode and validate a JWT access token.

    Args:
        token: The Bearer JWT token string.

    Returns:
        The authenticated user_id as UUID.

    Raises:
        AppException(401): If token is invalid, expired, or malformed.
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        sub: Optional[str] = payload.get("sub")
        if not sub:
            raise AppException(
                message="Invalid authentication credentials.",
                status_code=401,
            )
        return uuid.UUID(sub)
    except jwt.ExpiredSignatureError as exc:
        logger.warning("Expired JWT access token presented")
        raise AppException(
            message="Access token has expired.",
            status_code=401,
        ) from exc
    except (jwt.PyJWTError, ValueError) as exc:
        logger.warning("Invalid JWT token format: %s", exc)
        raise AppException(
            message="Invalid authentication credentials.",
            status_code=401,
        ) from exc
