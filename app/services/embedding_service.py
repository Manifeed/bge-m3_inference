from __future__ import annotations

from app.clients.flag_embedding_client import get_flag_embedding_client
from app.schemas.embedding_schema import (
    EmbeddingItemRead,
    EmbeddingRequestSchema,
    EmbeddingResponseRead,
)


def create_embeddings(payload: EmbeddingRequestSchema) -> EmbeddingResponseRead:
    client = get_flag_embedding_client()
    texts = payload.input if isinstance(payload.input, list) else [payload.input]
    dense_vectors, sparse_vectors = client.encode(
        texts=texts,
        dense=payload.dense,
        sparse=payload.sparse,
    )
    return EmbeddingResponseRead(
        data=[
            EmbeddingItemRead(
                index=index,
                embedding=dense if payload.dense else None,
                sparse_embedding=sparse if payload.sparse else None,
            )
            for index, (dense, sparse) in enumerate(zip(dense_vectors, sparse_vectors, strict=True))
        ],
    )


def check_embedding_model_ready() -> None:
    get_flag_embedding_client().check_ready()
