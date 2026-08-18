from fastapi import APIRouter, Depends, status

from personal_ai.api.dependencies import get_auth_service
from personal_ai.application.auth.auth_service import AuthService
from personal_ai.domain.user.models import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="Register a new user identity with email and password.",
)
async def register(
    request: RegisterRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> UserResponse:
    """Register a new user and return the safe user representation."""
    return await auth_service.register_user(request)


@router.post(
    "/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="User login",
    description="Authenticate user credentials and issue a Bearer JWT access token.",
)
async def login(
    request: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> TokenResponse:
    """Authenticate credentials and issue a Bearer JWT token."""
    return await auth_service.login_user(request)
