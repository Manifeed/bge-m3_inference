from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.clients.flag_embedding_client import get_flag_embedding_client
from app.domain.config import CANONICAL_SERVICE_NAME
from app.domain.config import require_embedding_api_key
from app.errors import register_exception_handlers
from app.routers.embedding_router import embedding_router
from app.schemas.embedding_schema import InternalServiceHealthRead
from app.services.embedding_service import check_embedding_model_ready


@asynccontextmanager
async def _app_lifespan(_: FastAPI):
    client = get_flag_embedding_client()
    client.start_warmup()
    try:
        yield
    finally:
        client.stop()


def create_app() -> FastAPI:
    app = FastAPI(title="Manifeed BGE-M3 Inference Service", lifespan=_app_lifespan)
    app.include_router(embedding_router)
    register_exception_handlers(app)

    @app.get("/internal/health", response_model=InternalServiceHealthRead)
    def read_internal_health() -> InternalServiceHealthRead:
        return InternalServiceHealthRead(service=CANONICAL_SERVICE_NAME, status="ok")

    @app.get("/internal/ready", response_model=InternalServiceHealthRead)
    def read_internal_ready() -> InternalServiceHealthRead:
        require_embedding_api_key()
        check_embedding_model_ready()
        return InternalServiceHealthRead(service=CANONICAL_SERVICE_NAME, status="ready")

    return app


app = create_app()
