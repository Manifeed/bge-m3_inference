from __future__ import annotations

from dataclasses import dataclass

from app.domain.config import (
    CANONICAL_BGE_M3_MODEL_NAME,
    resolve_batch_max_items,
    resolve_batch_max_tokens,
    resolve_batch_max_wait_ms,
    resolve_max_length,
    resolve_model_source,
    resolve_queue_max_items,
    resolve_request_timeout_seconds,
    resolve_shutdown_grace_seconds,
    resolve_use_fp16,
)


@dataclass
class FlagEmbeddingClientSettings:
    model_name: str
    model_source: str
    max_length: int
    use_fp16: bool
    batch_max_items: int
    batch_max_tokens: int
    batch_max_wait_ms: int
    queue_max_items: int
    request_timeout_seconds: float
    shutdown_grace_seconds: float


def build_flag_embedding_client_settings(
    model_name: str = CANONICAL_BGE_M3_MODEL_NAME,
) -> FlagEmbeddingClientSettings:
    return FlagEmbeddingClientSettings(
        model_name=model_name,
        model_source=resolve_model_source(),
        max_length=resolve_max_length(),
        use_fp16=resolve_use_fp16(),
        batch_max_items=resolve_batch_max_items(),
        batch_max_tokens=resolve_batch_max_tokens(),
        batch_max_wait_ms=resolve_batch_max_wait_ms(),
        queue_max_items=resolve_queue_max_items(),
        request_timeout_seconds=resolve_request_timeout_seconds(),
        shutdown_grace_seconds=resolve_shutdown_grace_seconds(),
    )
