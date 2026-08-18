from personal_ai.core.auth import create_access_token, hash_password, verify_password
from personal_ai.core.exceptions import AppException
from personal_ai.core.logger import get_logger
from personal_ai.db.repositories.base import UserRepository
from personal_ai.domain.user.models import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)

logger = get_logger(__name__)


class AuthService:
    """Application use-case service for User Registration and Authentication."""

    def __init__(self, user_repo: UserRepository) -> None:
        """Initialize AuthService with UserRepository interface."""
        self._user_repo = user_repo

    async def register_user(self, request: RegisterRequest) -> UserResponse:
        """Register a new user identity.

        Args:
            request: Registration payload containing email and password.

        Returns:
            UserResponse: Safe response representation of created user.

        Raises:
            AppException(400): If email is registered, password < 8 chars, or password > 72 bytes.
        """
        normalized_email = request.email.strip().lower()
        if len(request.password) < 8:
            raise AppException(
                message="Password must be at least 8 characters long.",
                status_code=400,
            )

        pwd_bytes = request.password.encode("utf-8")
        if len(pwd_bytes) > 72:
            raise AppException(
                message="Password exceeds maximum allowed length of 72 bytes.",
                status_code=400,
            )

        existing_user = await self._user_repo.get_user_by_email(normalized_email)
        if existing_user:
            logger.warning("User registration attempted with existing email [email=%s]", normalized_email)
            raise AppException(
                message="User with this email already exists.",
                status_code=400,
            )

        hashed_password = hash_password(request.password)
        user = await self._user_repo.create_user(
            email=normalized_email,
            password_hash=hashed_password,
        )

        return UserResponse(
            id=user.id,
            email=user.email,
            created_at=user.created_at,
        )

    async def login_user(self, request: LoginRequest) -> TokenResponse:
        """Authenticate user credentials and generate JWT access token.

        Args:
            request: Credentials payload containing email and password.

        Returns:
            TokenResponse: JWT access token and token type.

        Raises:
            AppException(401): Generic authentication failure if credentials match no user.
        """
        normalized_email = request.email.strip().lower()
        user = await self._user_repo.get_user_by_email(normalized_email)
        if not user or not verify_password(request.password, user.password_hash):
            logger.warning("Authentication failed for email [email=%s]", normalized_email)
            raise AppException(
                message="Invalid email or password.",
                status_code=401,
            )

        access_token = create_access_token(user_id=user.id)
        logger.info("User authenticated successfully [user_id=%s]", user.id)

        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
        )
