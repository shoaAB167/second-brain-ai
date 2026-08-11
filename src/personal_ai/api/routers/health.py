from fastapi import APIRouter
from pydantic import BaseModel, Field

from personal_ai.config.settings import settings

router = APIRouter()


class HealthCheckResponse(BaseModel):
    """Schema for health check endpoint response."""

    status: str = Field(default="healthy", description="Status of the application service")
    app_name: str = Field(..., description="Application name")
    environment: str = Field(..., description="Current deployment environment")


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Application Health Check",
    description="Returns current operational status and environment details of the service.",
)
async def health_check() -> HealthCheckResponse:
    """Execute health check endpoint."""
    return HealthCheckResponse(
        status="healthy",
        app_name=settings.app_name,
        environment=settings.app_env,
    )
