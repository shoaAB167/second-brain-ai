from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class UserEntity:
    """Domain representation of an authenticated User."""

    id: uuid.UUID
    email: str
    password_hash: str
    created_at: datetime
    updated_at: datetime
