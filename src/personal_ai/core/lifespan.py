from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from personal_ai.config.settings import settings
from personal_ai.core.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI application lifespan manager for startup and shutdown hooks."""
    logger.info("Starting up application: %s [%s]", settings.app_name, settings.app_env)
    
    # Startup tasks (e.g., DB connections, caches, external clients) can be initialized here in future sprints.
    
    yield
    
    # Shutdown tasks (e.g., closing DB pools, flushing metrics, cleanup) can be added here in future sprints.
    logger.info("Shutting down application: %s", settings.app_name)
