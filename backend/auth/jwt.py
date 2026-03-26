"""
auth/jwt.py
-----------
JWT token creation and verification.

FIX: datetime.utcnow() deprecated in Python 3.12+ —
     replaced with datetime.now(timezone.utc)
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from backend.config import settings

ALGORITHM     = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()

    # FIX: timezone-aware datetime (utcnow() is deprecated in 3.12+)
    expire = datetime.now(timezone.utc) + (
        expires_delta if expires_delta else timedelta(hours=settings.JWT_EXPIRE_HOURS)
    )

    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=ALGORITHM)


def verify_token(token: str) -> dict:
    credentials_exception = HTTPException(
        status_code = status.HTTP_401_UNAUTHORIZED,
        detail      = "Could not validate credentials",
        headers     = {"WWW-Authenticate": "Bearer"},
    )
    try:
        payload    = jwt.decode(token, settings.JWT_SECRET, algorithms=[ALGORITHM])
        username:   str = payload.get("sub")
        factory_id: str = payload.get("factory_id")

        if username is None or factory_id is None:
            raise credentials_exception

        return {"username": username, "factory_id": factory_id}

    except JWTError:
        raise credentials_exception


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    return verify_token(token)


async def get_factory_id(current_user: dict = Depends(get_current_user)) -> str:
    """Dependency to automatically scope routes to the user's factory."""
    return current_user.get("factory_id")