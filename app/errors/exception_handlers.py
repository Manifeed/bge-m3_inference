from __future__ import annotations

from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import JSONResponse

from .custom_exceptions import EmbeddingServiceError


def embedding_service_error_handler(_: Request, exception: EmbeddingServiceError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "code": exception.code,
            "message": str(exception),
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(EmbeddingServiceError, embedding_service_error_handler)
