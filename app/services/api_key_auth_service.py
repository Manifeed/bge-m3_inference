from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.domain.config import require_embedding_api_key


def require_embedding_api_key_auth(authorization: str | None = Header(default=None)) -> None:
    expected_api_key = require_embedding_api_key()
    expected_header = f"Bearer {expected_api_key}"
    if authorization != expected_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
