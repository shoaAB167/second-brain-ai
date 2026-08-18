from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from personal_ai.api.router import api_router
from personal_ai.config.settings import settings
from personal_ai.core.exceptions import AppException, app_exception_handler
from personal_ai.core.lifespan import lifespan


def create_app() -> FastAPI:
    """Factory to create and configure the FastAPI application instance."""
    application = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        debug=settings.debug,
        lifespan=lifespan,
    )

    # Configure CORS middleware
    origins = settings.cors_origins or ["*"]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins if "*" not in origins else ["*"],
        allow_credentials=True if "*" not in origins else False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register custom exception handlers
    application.add_exception_handler(AppException, app_exception_handler)  # type: ignore[arg-type]

    # Include top-level API router
    application.include_router(api_router)

    return application


app = create_app()
