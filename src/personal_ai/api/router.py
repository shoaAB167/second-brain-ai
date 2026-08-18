from fastapi import APIRouter

from personal_ai.api.routers import auth, chat, experiences, health
from personal_ai.config.settings import settings

api_router = APIRouter(prefix=settings.api_v1_str)
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(auth.router)
api_router.include_router(chat.router, tags=["Chat"])
api_router.include_router(experiences.router, tags=["Experiences"])
