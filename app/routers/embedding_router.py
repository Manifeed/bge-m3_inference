from __future__ import annotations

from fastapi import APIRouter, Depends

from app.schemas.embedding_schema import EmbeddingRequestSchema, EmbeddingResponseRead
from app.services.api_key_auth_service import require_embedding_api_key_auth
from app.services.embedding_service import create_embeddings


embedding_router = APIRouter(prefix="/v1", tags=["embeddings"])


@embedding_router.post(
    "/embeddings",
    response_model=EmbeddingResponseRead,
    dependencies=[Depends(require_embedding_api_key_auth)],
)
def embed_texts(payload: EmbeddingRequestSchema) -> EmbeddingResponseRead:
    return create_embeddings(payload)
