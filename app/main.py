from __future__ import annotations

from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import JSONResponse

from app.clients.flag_embedding_client import EmbeddingModelError
from app.domain.config import EmbeddingServiceConfigurationError, require_embedding_api_key
from app.routers.embedding_router import embedding_router
from app.schemas.embedding_schema import InternalServiceHealthRead
from app.services.embedding_service import check_embedding_model_ready


def create_app() -> FastAPI:
    app = FastAPI(title="Manifeed Embedding Service")
    app.include_router(embedding_router)

    @app.exception_handler(EmbeddingModelError)
    def embedding_model_error_handler(_: Request, exception: EmbeddingModelError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "code": "embedding_model_unavailable",
                "message": str(exception),
            },
        )

    @app.exception_handler(EmbeddingServiceConfigurationError)
    def embedding_configuration_error_handler(
        _: Request,
        exception: EmbeddingServiceConfigurationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "code": "embedding_service_misconfigured",
                "message": str(exception),
            },
        )

    @app.get("/internal/health", response_model=InternalServiceHealthRead)
    def read_internal_health() -> InternalServiceHealthRead:
        return InternalServiceHealthRead(service="embedding-service", status="ok")

    @app.get("/internal/ready", response_model=InternalServiceHealthRead)
    def read_internal_ready() -> InternalServiceHealthRead:
        require_embedding_api_key()
        check_embedding_model_ready()
        return InternalServiceHealthRead(service="embedding-service", status="ready")

    return app


app = create_app()
